from datasets import load_dataset

dataset = load_dataset("imdb")
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
from torch.utils.data import DataLoader

# Use a small batch size to reduce memory usage
train_dataset = tokenized_datasets['train'].with_format('torch')
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
import torch
from transformers import BertForSequenceClassification
from torch.nn.utils import parameters_to_vector

model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
model.eval()  # Ensure the model is in evaluation mode

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Fisher Information storage
fisher_info = [torch.zeros_like(p) for p in model.parameters()]

# Compute Fisher Information Matrix
for batch in train_loader:
    import torch

    # Assuming 'batch' is a dictionary of lists
    batch = {k: torch.tensor(v) if isinstance(v, list) else v for k, v in batch.items()}

    # Now convert to the desired device
    device = 'cuda'  # or 'cpu', depending on your setup
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    outputs = model(**batch)
    loss = outputs.loss
    model.zero_grad()
    loss.backward()

    # Accumulate squared gradients (Fisher Information approximation)
    with torch.no_grad():
        for i, p in enumerate(model.parameters()):
            fisher_info[i] += (p.grad ** 2)

# Normalize by number of batches
fisher_info = [f / len(train_loader) for f in fisher_info]

# Compute eigenvalues for each layer
eigenvalues_per_layer = []
for idx, layer in enumerate(model.bert.encoder.layer):
    layer_params = list(layer.parameters())
    layer_indices = [i for i, p in enumerate(model.parameters()) if p in layer_params]
    layer_fisher_info = torch.cat([fisher_info[i].flatten() for i in layer_indices])
    cov_matrix = layer_fisher_info.reshape(int(torch.sqrt(layer_fisher_info.numel())), -1)
    eigenvalues = torch.linalg.eigvals(cov_matrix).abs()
    eigenvalues_per_layer.append(eigenvalues.mean().real.item())

print(eigenvalues_per_layer)
