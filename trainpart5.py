import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2Tokenizer
from gpt_model import MiniGPT  # Ensure gpt_model.py contains the MiniGPT class

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Load tokenized data
with open("tokenized_war_and_peace.txt", "r") as f:
    tokenized_text = list(map(int, f.read().split()))

# Convert to tensor (reduce batch size for memory efficiency)
inputs_ids = torch.tensor(tokenized_text).unsqueeze(0)  # Shape: (1, sequence_length)

# Model parameters (reduce size to avoid memory issues)
vocab_size = tokenizer.vocab_size
d_model = 64   # Reduced from 768
n_heads = 4     # Reduced from 12
n_layers = 4    # Reduced from 12

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MiniGPT(vocab_size, d_model, n_heads, n_layers).to(device)

# Optimizer & Loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# Training loop (with memory optimizations)
epochs = 5
batch_size = 256  # Reduce if still running out of memory
accumulation_steps = 4  # Gradient accumulation for efficiency
num_batches = len(inputs_ids[0]) // batch_size

for epoch in range(epochs):
    total_loss = 0
    optimizer.zero_grad()

    for i in range(num_batches):
        # Get batch
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(inputs_ids[0]))
        batch_inputs = inputs_ids[:, start_idx:end_idx].to(device)

        # Forward pass
        outputs = model(batch_inputs)

        # Compute loss
        loss = loss_fn(outputs.view(-1, vocab_size), batch_inputs.view(-1))
        loss.backward()

        # Accumulate gradients every few steps
        if (i + 1) % accumulation_steps == 0 or i == num_batches - 1:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()

        # Free up memory (especially useful for GPUs)
        del batch_inputs, outputs
        torch.cuda.empty_cache()

        # Print progress every 10 batches
        if i % 10 == 0:
            print(f"Epoch {epoch+1}, Batch {i}/{num_batches}, Loss: {loss.item():.4f}")

    print(f"Epoch {epoch+1} completed. Avg Loss: {total_loss/num_batches:.4f}")

# Save trained model
torch.save(model.state_dict(), "trained_gpt.pth")
print("Training Complete: Model saved as trained_gpt.pth")
