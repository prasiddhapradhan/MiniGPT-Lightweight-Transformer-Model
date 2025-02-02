import torch
import torch.nn as nn

# Define your MiniGPT class using nn.Module
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers):
        super(MiniGPT, self).__init__()
        
        # Define layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads) for _ in range(n_layers)
        ])
        self.fc_out = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        
        # Pass through transformer blocks
        for layer in self.transformer_blocks:
            x = layer(x)
        
        # Output layer
        x = self.fc_out(x)
        return x

# Example usage:
vocab_size = 10000  # Example vocab size
d_model = 512  # Example model dimension
n_heads = 8  # Example number of attention heads
n_layers = 6  # Example number of transformer layers

# Create the model
model = MiniGPT(vocab_size, d_model, n_heads, n_layers)

# Example input (batch size = 2, sequence length = 5)
input_data = torch.randint(0, vocab_size, (5, 2))

# Forward pass
output = model(input_data)

print(output)

    
    