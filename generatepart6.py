import torch
from transformers import GPT2Tokenizer
from gpt_model import MiniGPT  # Ensure MiniGPT is correctly implemented

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Model parameters (must match training settings)
vocab_size = tokenizer.vocab_size
d_model = 64   # Ensure same as training
n_heads = 4
n_layers = 4

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MiniGPT(vocab_size, d_model, n_heads, n_layers).to(device)
model.load_state_dict(torch.load("trained_gpt.pth", map_location=device))
model.eval()  # Set to evaluation mode

# Function to generate text
def generate_text(prompt, max_length=50):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            next_token = outputs[:, -1, :].argmax(dim=-1).unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Stop if EOS token is generated
            if next_token.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(input_ids.squeeze(), skip_special_tokens=True)

# Example prompt
prompt = "Once upon a time"
generated_text = generate_text(prompt)
print("Generated Text:\n", generated_text)



    