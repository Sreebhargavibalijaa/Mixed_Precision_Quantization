import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.cluster import KMeans
from correlation_based_quantization.canonical_analysis_sensitivites import layer_sensitivitites

class LinearLSQ(nn.Module):
    def __init__(self, original_linear, nbits_w):
        super(LinearLSQ, self).__init__()
        self.original_linear = original_linear
        self.nbits_w = nbits_w
        self.original_weight = original_linear.weight.detach().clone()
    def calculate_memory_reduction(self):
        original_mem = self.original_weight.nelement() * 32  # Assuming original weights are 32-bit floats
        quantized_mem = self.original_weight.nelement() * self.nbits_w
        reduction_percent = 100 * (1 - quantized_mem / original_mem)
        compression_ratio = original_mem / quantized_mem
        return original_mem, quantized_mem, reduction_percent, compression_ratio

    def quantize(self, x, nbits):
        qmin = -(2 ** (nbits - 1))
        qmax = (2 ** (nbits - 1)) - 1
        min_val, max_val = x.min(), x.max()
        scale = (max_val - min_val) / (qmax - qmin)
        scale = max(scale, 1e-8)
        zero_point = qmin - min_val / scale
        q_x = torch.round(x / scale + zero_point)
        q_x.clamp_(qmin, qmax)
        q_x = (q_x - zero_point) * scale
        return q_x

    def forward(self, x):
        quantized_weight = self.quantize(self.original_linear.weight, self.nbits_w)
        self.original_linear.weight = nn.Parameter(quantized_weight)
        output = self.original_linear(x)
        self.original_linear.weight = nn.Parameter(self.original_weight)
        return output
def evaluate_model(model, tokenizer, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)
            outputs = model(**inputs)
            _, predicted = torch.max(outputs.logits, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    return accuracy
def replace_layers_and_calculate_memory(model, layer_precision_map):
    total_original_mem = 0
    total_quantized_mem = 0
    # First, collect all layers in a list to avoid mutating the OrderedDict during iteration
    layers = [(name, module) for name, module in model.named_modules()]

    for name, module in layers:
        if isinstance(module, nn.Linear):
            bits = layer_precision_map.get(name, 8)  # Default to 32 bits if not specified in the map
            quant_layer = LinearLSQ(module, bits)
            setattr(model, name, quant_layer)  # Replace the original layer with the quantized version
            model._modules[name] = quant_layer  # Ensure the module is updated in the model's dictionary of modules
            
            # After replacing the layer, calculate the memory change
            orig_mem, quant_mem, reduction, compression_ratio = quant_layer.calculate_memory_reduction()
            total_original_mem += orig_mem
            total_quantized_mem += quant_mem
            print(f"Layer {name}: Original Memory = {orig_mem} bits, Quantized Memory = {quant_mem} bits, Reduction = {reduction:.2f}%")
        if isinstance(module, nn.LayerNorm):
            bits = layer_precision_map.get(name, 8)  # Default to 32 bits if not specified in the map
            quant_layer = LinearLSQ(module, bits)
            setattr(model, name, quant_layer)  # Replace the original layer with the quantized version
            model._modules[name] = quant_layer  # Ensure the module is updated in the model's dictionary of modules
            
            # After replacing the layer, calculate the memory change
            orig_mem, quant_mem, reduction, compression_ratio = quant_layer.calculate_memory_reduction()
            total_original_mem += orig_mem
            total_quantized_mem += quant_mem
            print(f"Layer {name}: Original Memory = {orig_mem} bits, Quantized Memory = {quant_mem} bits, Reduction = {reduction:.2f}%")

    return total_original_mem, total_quantized_mem, compression_ratio


def main():
    # Initial setup
    model_name = 'bert-base-uncased'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
    dataset = load_dataset("imdb")
    dataset = dataset.map(lambda examples: tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128), batched=True)
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    dataloader = DataLoader(dataset['test'], batch_size=32)
    accuracy_before = evaluate_model(model, tokenizer, dataloader, device)
    print(f"Accuracy before quantization: {accuracy_before:.2f}")

    # Placeholder for calculating layer sensitivity and clustering
    layer_sensitivity = layer_sensitivitites#{'layer_0': 0.1, 'layer_1': 0.5, 'layer_2': 0.9}  # Example sensitivities
    print("Sree")
    print(layer_sensitivity)
    sensitivities = np.array(list(layer_sensitivity.values())).reshape(-1, 1)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(sensitivities)
    clusters = kmeans.labels_

    layer_precision_map = {}
    for i, (layer_name, _) in enumerate(layer_sensitivity.items()):
        if clusters[i] == 0:
            layer_precision_map[layer_name] = 32  # fp32 for most sensitive
        elif clusters[i] == 1:
            layer_precision_map[layer_name] = 16  # int16 for medium
        else:
            layer_precision_map[layer_name] = 8   # int8 for least sensitive

    # Apply quantization and print results
    
    total_orig_mem, total_quant_mem, compression_ratio = replace_layers_and_calculate_memory(model, layer_precision_map)
    total_reduction_percent = 100 * (1 - total_quant_mem / total_orig_mem)
    print(f"Total Memory Reduction: Original Memory = {total_orig_mem} bits, Quantized Memory = {total_quant_mem} bits, Reduction = {total_reduction_percent:.2f}%")
    print(f"Compression ratio = {compression_ratio:.2f}")
    # Evaluate after quantization
    accuracy_after = evaluate_model(model, tokenizer, dataloader, device)
    print(f"Accuracy after quantization: {accuracy_after:.2f}")

if __name__ == "__main__":
    main()
