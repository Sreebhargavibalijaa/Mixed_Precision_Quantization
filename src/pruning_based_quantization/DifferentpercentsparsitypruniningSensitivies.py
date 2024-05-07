import torch
from transformers import BertForSequenceClassification, BertTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

# Load model and tokenizer
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load IMDB dataset
dataset = load_dataset('imdb')
test_dataset = dataset['test'].map(
    lambda e: tokenizer(e['text'], truncation=True, padding='max_length', max_length=512), batched=True)
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
test_loader = DataLoader(test_dataset, batch_size=16)

# Function to introduce sparsity to a given layer
def apply_sparsity_to_layer(layer, sparsity_level):
    with torch.no_grad():
        for param in layer.parameters():
            flat_param = param.data.view(-1)
            threshold = torch.quantile(torch.abs(flat_param), sparsity_level)
            mask = torch.abs(flat_param) > threshold
            param.data *= mask.float().view(param.data.shape)

# Function to evaluate the model's accuracy
def evaluate_model(model):
    model.eval()
    total_accuracy = 0
    for batch in tqdm(test_loader, desc="Evaluating"):
        inputs = {k: v.to(device) for k, v in batch.items() if k != 'label'}
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
            total_accuracy += (predictions == batch['label'].to(device)).float().mean().item()
    return total_accuracy / len(test_loader)

# Main loop to apply sparsity and evaluate
layer_sensitivity = {}
base_accuracy = evaluate_model(model)
print(f"Base Accuracy: {base_accuracy}")
sparsity_levels = [0.3]
for s in sparsity_levels:
    print("Sparsity level is "+ str(s))
    sensitivities = []
    for layer_num, layer in enumerate(model.bert.encoder.layer):
        layer_name = f"Encoder Layer {layer_num}"
        print(f"Evaluating {layer_name}")
        # Evaluate with increasing sparsity levels
            # Clone the original model for each experiment
        sparse_model = BertForSequenceClassification.from_pretrained('bert-base-uncased').to(device)
        sparse_model.load_state_dict(model.state_dict())

        # Apply sparsity
        apply_sparsity_to_layer(sparse_model.bert.encoder.layer[layer_num], s)

        # Evaluate
        accuracy = evaluate_model(sparse_model)
        sensitivity = base_accuracy - accuracy
        sensitivities.append(sensitivity)
#         print(f"Sparsity Level: {s}, Accuracy: {accuracy}, Sensitivity: {sensitivity}")
        # Clean up to free GPU memory
        del sparse_model
        torch.cuda.empty_cache()
    layer_sensitivity[layer_name] = sensitivities
    print("Layer Sensitivity Analysis Completed:")
    for layer, sensitivity in layer_sensitivity.items():
        print(f"{layer}: {sensitivity}")
layer_sensitivitites = {}
layer_idx =0
for sensitivity in list(layer_sensitivity.items()):
    layer_sensitivitites["layer_"+str(layer_idx)] = sensitivity
    layer_idx+=1

