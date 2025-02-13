from openai import OpenAI
import tkinter as tk
from tkinter import scrolledtext, ttk
import math
import os
from dotenv import load_dotenv

# Load API key from .env file
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
      - tokens is a list of token objects (each with a .token attribute).
      - top_logprobs is a list (one per token) where each element is a list
        of alternative objects (each with .token and .logprob attributes).
    """
    response = client.chat.completions.create(
        model="gpt-4",   # Change this to your desired model if needed.
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        logprobs=logprobs,  # this is what will track the probabilities.
        temperature=temp,
        top_logprobs=3
    )
    # Assume the API response returns a .logprobs.content list of token objects.
    tokens = response.choices[0].logprobs.content
    top_logprobs = []
    for token_obj in tokens:
        if token_obj.top_logprobs == []:  # check there are alternatives
            print(f"Token {token_obj.token} does not have alternatives available. Check the model tested.")
            raise ValueError("No alternatives available for token.")
        top_logprobs.append(token_obj.top_logprobs)
    return tokens, top_logprobs

class InteractiveLLM:
    """
    A Tkinter GUI that shows the generated text and for each token
    visualizes the top 3 alternative tokens (and their probabilities)
    as horizontal bars. You can use the slider to jump to any token,
    or click "Next Word" to advance.
    """
    def __init__(self, tokens, top_logprobs):
        self.tokens = tokens
        self.top_logprobs = top_logprobs
        self.num_tokens = len(tokens)
        self.current_index = 0

        # Create main window
        self.root = tk.Tk()
        self.root.title("Interactive LLM Token Viewer")
        self.root.geometry("900x600")  # Adjust size as needed

        # --- Top Frame with Text and Alternatives ---
        top_frame = ttk.Frame(self.root)
        top_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left frame for generated text
        text_frame = ttk.Frame(top_frame)
        text_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        top_frame.columnconfigure(0, weight=3)
        top_frame.rowconfigure(0, weight=1)

        text_label = ttk.Label(text_frame, text="Generated Text:", font=("Helvetica", 14))
        text_label.pack(anchor="w")
        self.text_area = scrolledtext.ScrolledText(text_frame, wrap=tk.WORD, width=50, height=20, font=("Helvetica", 14))
        self.text_area.pack(fill=tk.BOTH, expand=True)

        # Right frame for alternatives visualization
        alt_frame = ttk.Frame(top_frame)
        alt_frame.grid(row=0, column=1, sticky="nsew")
        top_frame.columnconfigure(1, weight=2)

        alt_label = ttk.Label(alt_frame, text="Alternative Tokens", font=("Helvetica", 14))
        alt_label.pack(anchor="w")
        # Create a canvas that will display the alternative bars.
        self.alt_canvas = tk.Canvas(alt_frame, bg="white", height=200)
        self.alt_canvas.pack(fill=tk.BOTH, expand=True, pady=(5,0))

        # --- Bottom Frame with Controls (Slider and Button) ---
        bottom_frame = ttk.Frame(self.root)
        bottom_frame.pack(fill=tk.X, padx=10, pady=10)

        self.token_slider = ttk.Scale(bottom_frame, from_=0, to=self.num_tokens-1, orient=tk.HORIZONTAL, command=self.slider_changed)
        self.token_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.slider_label = ttk.Label(bottom_frame, text=f"Token: 0/{self.num_tokens-1}", font=("Helvetica", 12))
        self.slider_label.pack(side=tk.LEFT)

        self.next_button = ttk.Button(bottom_frame, text="Next Word", command=self.next_word)
        self.next_button.pack(side=tk.LEFT, padx=(10,0))

        # Make sure canvas redraws after window layout is complete.
        self.root.after(100, lambda: self.update_display(0))

    def update_display(self, index):
        """Update the generated text and alternative token display for token index 'index'."""
        self.current_index = int(index)
        # Update slider label.
        self.slider_label.config(text=f"Token: {self.current_index}/{self.num_tokens-1}")

        # Update the text area with tokens 0 to current index.
        generated_text = "".join(str(getattr(token_obj, 'token')) for token_obj in self.tokens[:self.current_index+1])
        self.text_area.delete("1.0", tk.END)
        self.text_area.insert(tk.END, generated_text)

        # Update the alternatives display for the current token.
        if self.current_index < len(self.top_logprobs):
            token_alternatives = self.top_logprobs[self.current_index]
            # Sort alternatives by logprob (largest first)
            sorted_alts = sorted(token_alternatives, key=lambda alt: getattr(alt, "logprob", float("-inf")), reverse=True)
            top3 = sorted_alts[:3]

            # Clear canvas
            self.alt_canvas.delete("all")
            self.alt_canvas.update_idletasks()  # ensure we have updated dimensions
            canvas_width = self.alt_canvas.winfo_width() or 300
            canvas_height = self.alt_canvas.winfo_height() or 200

            # Determine bar heights and spacing
            gap = 10
            bar_height = (canvas_height - (len(top3)+1)*gap) / len(top3)
            # Compute probabilities and scale factors
            # (we use math.exp to convert logprob)
            probs = [math.exp(getattr(alt, "logprob", float("-inf"))) for alt in top3]
            max_prob = max(probs) if probs else 1.0

            for i, alt in enumerate(top3):
                alt_token = str(getattr(alt, "token"))
                alt_logprob = getattr(alt, "logprob", float("-inf"))
                alt_prob = math.exp(alt_logprob)
                # Compute width proportional to probability (reserve 150 px for text)
                bar_max_width = canvas_width - 150
                fraction = alt_prob / max_prob if max_prob > 0 else 0
                bar_width = fraction * bar_max_width

                # Coordinates for this bar
                y_top = gap + i * (bar_height + gap)
                y_bottom = y_top + bar_height
                x_left = 10
                x_right = x_left + bar_width

                # Draw rectangle (bar)
                self.alt_canvas.create_rectangle(x_left, y_top, x_right, y_bottom, fill="skyblue", outline="black")
                # Draw text with token and probability next to the bar.
                text_x = x_right + 10
                text_y = (y_top + y_bottom) / 2
                alt_text = f'"{alt_token}"  {alt_prob:.4f}'
                self.alt_canvas.create_text(text_x, text_y, anchor="w", font=("Helvetica", 12), text=alt_text)
        else:
            self.alt_canvas.delete("all")

    def slider_changed(self, event):
        """Callback for when the slider is moved."""
        # event is a string representing the current slider value.
        self.update_display(self.token_slider.get())

    def next_word(self):
        """Advance to the next token (if available) and update the slider."""
        next_index = self.current_index + 1
        if next_index < self.num_tokens:
            # Set the slider (this will trigger slider_changed)
            self.token_slider.set(next_index)
        else:
            self.next_button.config(state=tk.DISABLED)

    def run(self):
        self.root.mainloop()

def main():
    prompt = input("Enter your prompt: ")
    print("Fetching completion from GPT-4...")
    try:
        tokens, top_logprobs = fetch_completion(prompt)
    except Exception as e:
        print("An error occurred while fetching the completion:")
        print(e)
        return

    print("Completion fetched. Starting interactive session.")
    interactive_session = InteractiveLLM(tokens, top_logprobs)
    interactive_session.run()

if __name__ == "__main__":
    main()
