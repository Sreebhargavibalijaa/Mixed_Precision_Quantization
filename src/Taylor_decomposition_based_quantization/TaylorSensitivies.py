import torch
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
from torch.utils.data import DataLoader
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
dataset = load_dataset("imdb", split='test[:10%]')
def encode(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512, return_tensors='pt')
dataset = dataset.map(encode, batched=True)
dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
dataloader = DataLoader(dataset, batch_size=10)
def taylor_sensitivity(model, dataloader, device):
    sensitivities = []
    model.eval()
    for index, layer in enumerate(model.bert.encoder.layer):
        # Store original state
        original_state_dict = {name: param.clone() for name, param in layer.named_parameters()}

        # Generate perturbation and apply it to the first parameter
        first_param_name, first_param = next(iter(layer.named_parameters()))
        perturbation = torch.randn_like(first_param, device=device)
        first_param.data.add_(perturbation)  # Apply perturbation in-place

        total_variation = 0.0
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
            with torch.no_grad():
                outputs = model(**inputs)
                loss = outputs.loss if outputs.loss is not None else outputs.logits.sum()
            total_variation += loss.item()

        # Restore original layer state
        layer.load_state_dict(original_state_dict)

        # Calculate normalized variation
        normalized_variation = total_variation / len(dataloader.dataset)
        sensitivities.append((index, normalized_variation))

    return sensitivities
sensitivity_scores = taylor_sensitivity(model, dataloader, device)
layer_sensitivitites = {}
for layer_idx, sensitivity in sensitivity_scores:
    print(f"Layer {layer_idx}: Sensitivity = {sensitivity}")
    layer_sensitivitites["layer_"+str(layer_idx)]=sensitivity
