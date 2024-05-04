import torch
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch.nn as nn

def compute_fim(model, dataloader, device):
    model.eval()
    fim = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            # Create FIM placeholders for each layer on the same device as the parameter
            fim[name] = torch.zeros(parameter.numel(), parameter.numel()).to(device)

    for batch in dataloader:
        inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
        labels = batch['labels'].to(device)
        outputs = model(**inputs)
        loss = nn.CrossEntropyLoss()(outputs.logits, labels)
        model.zero_grad()
        loss.backward()

        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                grad = parameter.grad.data.flatten()
                fim_update = torch.ger(grad, grad)
                fim[name] += fim_update

    # Normalize FIM by the number of data samples and compute the mean eigenvalues
    mean_eigenvalues = {}
    for name, matrix in fim.items():
        matrix /= len(dataloader.dataset)
        u, s, v = torch.linalg.svd(matrix.cpu())  # SVD on CPU due to potential size
        mean_eigenvalue = torch.mean(s).item()
        mean_eigenvalues[name] = mean_eigenvalue
        print(f"Layer {name}: Mean Eigenvalue = {mean_eigenvalue}")

    return mean_eigenvalues

def main():
    model_name = 'bert-base-uncased'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

    # Load and prepare IMDb dataset
    dataset = load_dataset("imdb")
    dataset = dataset.map(lambda examples: tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128), batched=True)
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    dataloader = DataLoader(dataset['test'], batch_size=32, shuffle=False)

    # Compute Fisher Information Matrix
    fim_eigenvalues = compute_fim(model, dataloader, device)

if __name__ == "__main__":
    main()
