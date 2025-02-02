from transformers import GPT2Tokenizer

# Load the GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Ensure the correct file name
file_path = "cleaned_war_and_peace.txt"  # Fixed file extension

try:
    # Open the text file and read its content
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Tokenize the text
    tokens = tokenizer.encode(text, add_special_tokens=True)

    # Save the tokenized output
    with open("tokenized_war_and_peace.txt", "w", encoding="utf-8") as f:
        f.write(" ".join(map(str, tokens)))

    print(f"Tokenization complete! Tokenized data saved to 'tokenized_war_and_peace.txt'")

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Make sure the file exists in the directory.")
