from openai import OpenAI
import tkinter as tk
from tkinter import scrolledtext
import math
import os
from dotenv import load_dotenv


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    raise ValueError("Please set your OPENAI_API_KEY in a .env file.")


client = OpenAI(api_key=api_key)

def fetch_completion(prompt, temp=0.7, max_tokens=100, logprobs=True):
    """
    Calls the OpenAI API with the given prompt.
    Requests a completion with up to max_tokens and with logprobs enabled.
    Returns a tuple (tokens, top_logprobs) where:
      - tokens is a list of the tokens in the generated text.
      - top_logprobs is a list of dictionaries for each token,
        mapping alternative tokens to their log probabilities.
    """
    response = client.chat.completions.create(
        model="gpt-4",   # Change this to your desired model if needed.
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        logprobs=logprobs, # this is what will track the probabilities.
        temperature=temp,
        top_logprobs=3
    )
    # The 'logprobs' field contains:
    #   - tokens: list of generated token strings.
    #   - top_logprobs: list of dicts mapping candidate tokens to log probability.
    tokens = response.choices[0].logprobs.content
    top_logprobs = []
    for logprobs in tokens:
        if logprobs.top_logprobs == []: # check there are alternatives
            print(f"Token {logprobs.token} does not have alternatives available. Check the model tested.")
            raise ValueError 
        top_logprobs.append(logprobs.top_logprobs)
    return tokens, top_logprobs

class InteractiveLLM:
    """
    A simple Tkinter GUI that displays the generated text one token at a time.
    For each token revealed, it shows the top three alternative tokens (and their probabilities)
    that the model considered.
    """
    def __init__(self, tokens, top_logprobs):
        self.tokens = tokens
        self.top_logprobs = top_logprobs
        self.current_index = 0

        # Set up the main window.
        self.root = tk.Tk()
        self.root.title("Interactive LLM Token Viewer")

        # Create a scrolled text widget to display the generated text.
        self.text_area = scrolledtext.ScrolledText(
            self.root, wrap=tk.WORD, width=60, height=10, font=("Helvetica", 14)
        )
        self.text_area.pack(padx=10, pady=10)

        # Label for the alternatives.
        self.alt_label = tk.Label(
            self.root,
            text="Top alternatives for current token:",
            font=("Helvetica", 12)
        )
        self.alt_label.pack()

        # A text widget to show the alternatives.
        self.alt_text = tk.Text(
            self.root, wrap=tk.WORD, width=60, height=5, font=("Helvetica", 12)
        )
        self.alt_text.pack(padx=10, pady=10)

        # A button to show the next token.
        self.next_button = tk.Button(
            self.root, text="Next Word", command=self.show_next_token, font=("Helvetica", 14)
        )
        self.next_button.pack(pady=10)

    def show_next_token(self):
        """
        When the button is pressed, this method appends the next token
        to the displayed text and updates the alternatives list.
        """
        if self.current_index < len(self.tokens):
            # Get the current token.
            token = self.tokens[self.current_index]
            # Append the token to the text area.
            #current_token_name = str(getattr(token, 'token'))
            self.text_area.insert(tk.END, str(getattr(token, 'token')))
            self.text_area.see(tk.END)

            # Clear the alternatives widget.
            self.alt_text.delete("1.0", tk.END)

            # Retrieve the dictionary of candidate tokens and their log probabilities.
            token_logprobs = self.top_logprobs[self.current_index]
            # Sort candidates by log probability (highest first).
            sorted_alts = sorted(token_logprobs, key=lambda obj: getattr(obj, "logprob", float("-inf")), reverse=True) # the -inf will be sorted last if no num
            # Take the top 3 alternatives.
            top3 = sorted_alts[:3]

            # Build a string to display each alternative with its probability.
            alt_display = ""
            for alt_obj in top3:
                # Convert log probability to probability.
                alt_prob = math.exp(getattr(alt_obj, "logprob", float("-inf")))
                alt_word = str(getattr(alt_obj, 'token'))
                alt_display += f"Token: \"{alt_word}\" - Probability: {alt_prob:.4f}\n"
            self.alt_text.insert(tk.END, alt_display)

            self.current_index += 1
        else:
            # When finished, disable the button.
            self.next_button.config(state=tk.DISABLED)
            self.alt_text.delete("1.0", tk.END)
            self.alt_text.insert(tk.END, "End of response.")

    def run(self):
        self.root.mainloop()

def main():
    # Get the prompt from the user.
    prompt = input("Enter your prompt: ")
    print("Fetching completion from GPT-4o-mini...")

    try:
        tokens, top_logprobs = fetch_completion(prompt)
    except Exception as e:
        print("An error occurred while fetching the completion:")
        print(e)
        return

    print("Completion fetched. Starting interactive session.")
    # Start the interactive GUI.
    interactive_session = InteractiveLLM(tokens, top_logprobs)
    interactive_session.run()

if __name__ == "__main__":
    main()
