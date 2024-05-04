import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch.nn.utils.prune as prune

def train_model():
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    dataset = load_dataset("imdb")
    encoded_dataset = dataset.map(lambda x: tokenizer(x['text'], padding='max_length', truncation=True, max_length=512), batched=True)
    encoded_dataset = encoded_dataset.rename_column("label", "labels")
    encoded_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    train_dataset = encoded_dataset['train']
    validation_dataset = encoded_dataset['test']

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset
    )

    trainer.train()
    return model, validation_dataset

def prune_and_evaluate(model, validation_dataset, layer_to_prune, amount=0.2):
    parameters_to_prune = [(module, 'weight') for name, module in model.named_modules() if isinstance(module, torch.nn.Linear) and layer_to_prune in name]
    for module, param in parameters_to_prune:
        prune.l1_unstructured(module, name='weight', amount=amount)
    
    eval_result = Trainer(model=model).evaluate(eval_dataset=validation_dataset)
    return eval_result['eval_loss']

def main():
    model, validation_dataset = train_model()
    original_loss = Trainer(model=model).evaluate(eval_dataset=validation_dataset)['eval_loss']
    sensitivities = {}
    for i in range(12):  # BERT Base has 12 transformer layers
        layer_name = f'bert.encoder.layer.{i}.intermediate.dense'
        pruned_loss = prune_and_evaluate(model, validation_dataset, layer_name)
        sensitivity = (pruned_loss - original_loss) / original_loss
        sensitivities[layer_name] = sensitivity
        print(f'Layer {i+1} sensitivity: {sensitivity}')

if __name__ == "__main__":
    main()
