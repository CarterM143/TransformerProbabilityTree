import math
import os
import tkinter as tk
from tkinter import ttk
from dotenv import load_dotenv
from openai import OpenAI

# Load your API key from .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    raise ValueError("Please set your OPENAI_API_KEY in a .env file.")

client = OpenAI(api_key=api_key)

########################################################################
# 1. API CALL & TREE BUILDING FUNCTIONS
########################################################################

def fetch_completion(prompt, temp=0.7, max_tokens=50, logprobs=True):
    """
    Calls the API with the given prompt.
    Returns:
      tokens: a list of token objects (each with a .token attribute)
      top_logprobs: a list (one per generated token) of lists of alternative objects.
         (Each alternative object should have attributes .token and .logprob.)
    """
    response = client.chat.completions.create(
        model="gpt-4",   # Change if needed.
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        logprobs=logprobs,  # to get probabilities.
        temperature=temp,
        top_logprobs=3
    )
    # In this example we assume the response returns an object .logprobs.content
    # which is a list of token objects (each with a .token attribute and a .top_logprobs field)
    tokens = response.choices[0].logprobs.content
    # For each token, we assume token.top_logprobs is a list of alternative candidate objects.
    top_logprobs = []
    for token_obj in tokens:
        if not token_obj.top_logprobs:
            raise ValueError(f"Token '{token_obj.token}' has no alternatives!")
        top_logprobs.append(token_obj.top_logprobs)
    return tokens, top_logprobs


class TreeNode:
    """
    Represents one token-generation event.
      - token: the token text (a string)
      - logprob: log probability (a float)
      - chosen: True if this token is the one that was generated in the “chosen” branch.
      - parent: pointer to parent TreeNode (or None for the root)
      - children: list of child TreeNodes (each representing one candidate for the next token)
      - expanded: whether we have already fetched a continuation from this node.
      - depth: for bookkeeping (0 for the dummy root)
    """
    def __init__(self, token, logprob, parent=None, chosen=False, depth=0):
        self.token = token
        self.logprob = logprob
        self.parent = parent
        self.chosen = chosen
        self.children = []      # Will hold TreeNode children (for the next generation event)
        self.expanded = False   # Has this node been expanded (i.e. continuation fetched) yet?
        self.depth = depth

    def path_tokens(self):
        """Return a list of tokens from the root (excluding the dummy root) to this node."""
        tokens = []
        node = self
        while node.parent is not None:
            tokens.append(node.token)
            node = node.parent
        tokens.reverse()
        return tokens


def build_branch_from_completion(parent_node, tokens, top_logprobs):
    """
    Given a parent TreeNode and an API response (tokens and alternatives) for its continuation,
    build a branch: at each generation event attach all candidate tokens as children,
    marking the one that equals the API’s chosen token with chosen=True.
    Returns the last (chosen) TreeNode in the branch.
    """
    current_parent = parent_node
    for token_obj, alternatives in zip(tokens, top_logprobs):
        chosen_token = getattr(token_obj, 'token')
        # Find the chosen candidate’s logprob (assumed to be among the alternatives)
        chosen_logprob = None
        children = []
        for cand in alternatives:
            cand_token = getattr(cand, 'token')
            cand_logprob = getattr(cand, 'logprob', float("-inf"))
            is_chosen = (cand_token == chosen_token)
            node = TreeNode(token=cand_token, logprob=cand_logprob,
                            parent=current_parent,
                            chosen=is_chosen,
                            depth=current_parent.depth + 1)
            children.append(node)
            if is_chosen:
                chosen_node = node
                chosen_logprob = cand_logprob
        # Attach all candidate nodes as children of current_parent.
        current_parent.children = children
        current_parent.expanded = True
        # Move to the chosen branch for the next generation event.
        current_parent = chosen_node
    # Mark the final node as expanded (even if it has no children yet).
    current_parent.expanded = True
    return current_parent


def expand_node(node, initial_prompt):
    """
    Expand a branch that has not been expanded yet.
    Build the prompt by concatenating the initial prompt and the branch tokens.
    Fetch a new continuation and attach it as children to the given node.
    Returns the new chosen (continuation) node.
    """
    branch_text = "".join(node.path_tokens())
    new_prompt = initial_prompt + branch_text
    tokens, top_logprobs = fetch_completion(new_prompt)
    chosen_end_node = build_branch_from_completion(node, tokens, top_logprobs)
    return chosen_end_node

########################################################################
# 2. THE UI: AN INTERACTIVE TREEVIEW
########################################################################

class InteractiveTreeLLM:
    """
    Displays the branching tree of completions using Tkinter’s Treeview.
    - The top “Current Branch” area shows the tokens (from the prompt) along the current path.
    - The Treeview (on the lower portion) shows the entire tree starting at the prompt.
    • Double-clicking on a node that is not yet expanded will trigger a new API call to expand it.
    • Selecting a node (or clicking “Back”) updates the current branch display.
    """
    def __init__(self, initial_prompt, root_node, current_node):
        self.initial_prompt = initial_prompt
        self.root_node = root_node    # Dummy root node
        self.current_node = current_node  # The current branch endpoint

        # Set up the main window.
        self.root = tk.Tk()
        self.root.title("Interactive LLM Tree Explorer")
        self.root.geometry("1000x700")

        # Top frame: display the current branch and a Back button.
        top_frame = tk.Frame(self.root)
        top_frame.pack(fill=tk.X, padx=10, pady=10)
        tk.Label(top_frame, text="Current Branch:", font=("Helvetica", 16)).pack(side=tk.LEFT)
        self.branch_text = tk.Label(top_frame, text="", font=("Helvetica", 14), fg="blue")
        self.branch_text.pack(side=tk.LEFT, padx=10)
        tk.Button(top_frame, text="Back", command=self.go_back).pack(side=tk.RIGHT)

        # Middle frame: Treeview widget to display the entire tree.
        tree_frame = tk.Frame(self.root)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.tree = ttk.Treeview(tree_frame)
        self.tree.pack(fill=tk.BOTH, expand=True)
        self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)
        self.tree.bind("<Double-1>", self.on_tree_double_click)

        # We keep mappings from TreeNode objects to Treeview item IDs.
        self.node_to_id = {}
        self.id_to_node = {}

        # Initially populate the tree from the root.
        self.populate_treeview()
        self.update_branch_display()

    def populate_treeview(self):
        """Clear and repopulate the Treeview from the tree structure."""
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.node_to_id = {}
        self.id_to_node = {}
        # Insert a root item that shows the prompt.
        root_id = self.tree.insert("", "end", text="Prompt", open=True)
        self.node_to_id[self.root_node] = root_id
        self.id_to_node[root_id] = self.root_node
        # Recursively insert the children.
        self.insert_children(self.root_node, root_id)

    def insert_children(self, node, parent_id):
        """Recursively insert node.children into the Treeview."""
        for child in node.children:
            # Show the token and its probability (converted from logprob).
            display_text = f"{child.token} (p={math.exp(child.logprob):.4f})"
            item_id = self.tree.insert(parent_id, "end", text=display_text, open=True)
            self.node_to_id[child] = item_id
            self.id_to_node[item_id] = child
            # If this child has been expanded, insert its children.
            if child.expanded and child.children:
                self.insert_children(child, item_id)

    def update_branch_display(self):
        """Update the 'Current Branch' area to show the tokens along the current path."""
        tokens = self.current_node.path_tokens()
        branch_str = " ".join(tokens)
        self.branch_text.config(text=branch_str)

    def on_tree_select(self, event):
        """When a node is selected, update the current branch display."""
        selected_item = self.tree.selection()
        if selected_item:
            node = self.id_to_node.get(selected_item[0])
            if node:
                self.current_node = node
                self.update_branch_display()

    def on_tree_double_click(self, event):
        """
        On double-click, if the node is not yet expanded, then expand it
        (i.e. call the API to get its continuation) and refresh the tree.
        """
        item_id = self.tree.focus()
        if item_id:
            node = self.id_to_node.get(item_id)
            if node and not node.expanded:
                self.expand_and_refresh(node)

    def expand_and_refresh(self, node):
        """Expand a node (if not already) and repopulate the tree view."""
        self.root.config(cursor="wait")
        self.root.update()
        try:
            # This call attaches new children to the node.
            expand_node(node, self.initial_prompt)
        except Exception as e:
            print("Error expanding node:", e)
        self.root.config(cursor="")
        self.populate_treeview()

    def go_back(self):
        """Go back one token (if available) along the current branch."""
        if self.current_node.parent is not None:
            self.current_node = self.current_node.parent
            self.update_branch_display()
            item_id = self.node_to_id.get(self.current_node)
            if item_id:
                self.tree.selection_set(item_id)

    def run(self):
        self.root.mainloop()

########################################################################
# 3. MAIN: BUILD INITIAL TREE AND RUN THE UI
########################################################################

def main():
    # Get prompt from the user.
    initial_prompt = input("Enter your prompt: ")
    print("Fetching initial completion from GPT-4...")
    try:
        tokens, top_logprobs = fetch_completion(initial_prompt)
    except Exception as e:
        print("An error occurred while fetching the completion:")
        print(e)
        return

    # Create a dummy root node (representing the prompt).
    root_node = TreeNode(token="(Prompt)", logprob=0.0, parent=None, chosen=True, depth=0)
    # Build the initial branch from the API response and attach it to the root.
    current_node = build_branch_from_completion(root_node, tokens, top_logprobs)
    print("Initial branch built. Launching interactive tree explorer...")
    ui = InteractiveTreeLLM(initial_prompt, root_node, current_node)
    ui.run()


if __name__ == "__main__":
    main()
