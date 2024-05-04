import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset

def compute_fisher_information_diagonal(model, data_loader):
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    fisher_information_matrix = torch.zeros(sum(p.numel() for p in params), device=model.device)

    model.zero_grad()
    for batch in data_loader:
        # Ensure input tensors are on the correct device
        inputs = {k: v.to(model.device) for k, v in batch.items() if k in ['input_ids', 'attention_mask', 'labels']}
        outputs = model(**inputs)
        loss = outputs.loss

        model.zero_grad()  # Clear previous gradients
        loss.backward()

        # Collect gradients for all trainable parameters
        grads = [p.grad.flatten() for p in params if p.grad is not None]
        grad_vec = torch.cat(grads)
        fisher_information_matrix += grad_vec ** 2

    fisher_information_matrix /= len(data_loader)

    # Decompose the FIM to extract the diagonal for each parameter
    fisher_diag = {}
    index = 0
    for name, param in model.named_parameters():
        num_param = param.numel()
        fisher_diag[name] = fisher_information_matrix[index:index + num_param].reshape(param.shape)
        index += num_param

    return fisher_diag

# Calculate mean eigenvalue for each layer type
def mean_eigenvalues_by_layer(fisher_diagonals):
    layer_types = {'embedding': [], 'linear': [], 'layernorm': []}
    for name, diag in fisher_diagonals.items():
        mean_val = torch.mean(diag).item()
        if "embedding" in name:
            layer_types['embedding'].append(mean_val)
        elif "linear" in name:
            layer_types['linear'].append(mean_val)
        elif "layernorm" in name or "LayerNorm" in name:
            layer_types['layernorm'].append(mean_val)

    for layer_type, values in layer_types.items():
        if values:
            print(f"Mean Eigenvalue for {layer_type} layers: {sum(values) / len(values)}")
        else:
            print(f"No data for {layer_type} layers")

# Main execution
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Load and preprocess the IMDb dataset
dataset = load_dataset("imdb")
train_dataset = dataset['train'].map(lambda examples: tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512), batched=True)
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

# Compute the Fisher Information Matrix diagonal
fisher_diagonal = compute_fisher_information_diagonal(model, train_loader)

# Calculate mean eigenvalues by layer
mean_eigenvalues_by_layer(fisher_diagonal)
