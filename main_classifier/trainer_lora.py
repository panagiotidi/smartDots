import os
import numpy as np
import torch.nn as nn
from accelerate import DataLoaderConfiguration
from numpy import nan
import evaluate
from transformers import TrainingArguments, Trainer, AutoImageProcessor, ViTForImageClassification, EvalPrediction, \
    PretrainedConfig
from sklearn.metrics import classification_report
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
import torch
from torcheval.metrics import MulticlassConfusionMatrix
from torchvision.transforms import ToTensor, Compose, Resize
from dataloader.FishLoader import FishDataset

from config import BATCH_SIZE, epochs, clean_data_path, subsample_fraction, device, learning_rate, weights, \
    filter_species, model_name, regression, total_classes, weight_decay, metric_max_diff
from utils import compute_max_diff, print_separate_confusions
from peft import LoraConfig, get_peft_model


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )


def collate_fn(examples):
    pixel_values = torch.stack([example[0] for example in examples])
    labels = torch.stack([example[1] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def compute_metrics(eval_pred: EvalPrediction):
    """Computes accuracy on a batch of predictions"""
    if regression == 'ordinal':
        predictions = np.argmax(eval_pred.predictions, axis=1)
        abs_labels = np.sum(eval_pred.label_ids, axis=1, dtype=np.int64) - 1
    elif regression == 'categorical_prob' or regression == 'categorical_abs':
        predictions = np.argmax(eval_pred.predictions, axis=1)
        abs_labels = np.argmax(eval_pred.label_ids, axis=1)
    else:
        exit('Continuous regression not implemented!')

    # print('predictions:', eval_pred.label_ids, eval_pred.predictions)
    # print('targets:', predictions, abs_labels)

    print(classification_report(abs_labels, predictions, zero_division=nan))
    loss = criterion(torch.tensor(eval_pred.predictions.copy()).to(device),
                     torch.tensor(eval_pred.label_ids.copy()).to(device))
    print('crossentropy:', loss)
    max_diff = compute_max_diff(predictions, abs_labels, metric_max_diff)
    print('max_diff accuracy:', max_diff)

    metric_conf_matrix.reset()
    metric_conf_matrix.update(torch.tensor(predictions.copy()).cpu(), torch.tensor(abs_labels.copy()).cpu())
    print('Confusion matrix\n', metric_conf_matrix.compute())

    print_separate_confusions(valDataset.dataset, predictions, abs_labels)

    return metric.compute(predictions=predictions, references=abs_labels)


class weighted_RMSELoss(nn.Module):
    def __init__(self, weight):
        super().__init__()

    def forward(self, inputs, targets):
        # print(inputs, targets, self.weight)
        mse_loss = torch.sqrt(torch.sum(((inputs - targets) ** 2) * class_weights))
        # print('mse_loss:', mse_loss)
        return mse_loss


class TrainerCustomLoss(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        # print('inputs:', inputs['pixel_values'].shape)
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        # print('logits:', logits)
        # print('labels:', labels)
        loss = criterion(logits, labels)
        return (loss, outputs) if return_outputs else loss


if __name__ == '__main__':

    #################### Model ###############################3
    # lo:PretrainedConfig

    if model_name != 'ViT':
        exit('Only ViT model with Lora adapter for now. Sorry..')
    else:
        # model = str_to_class(model_name)().to(device)
        image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", num_labels=total_classes,
                                                          ignore_mismatched_sizes=True).to(device)

    print('Model: ', model.__class__)

    #################### Define preprocess ###############################3

    if model_name != 'ViT':
        exit('Only ViT model with Lora adapter for now. Sorry..')
    else:
        transforms = Compose([
            Resize(image_processor.size["height"]),
            ToTensor()
        ])

    #################### Data preparation ###############################3

    trainDataset = FishDataset(os.path.join(clean_data_path, 'train'), preprocess=transforms,
                               fraction=subsample_fraction, filter_species=filter_species)
    valDataset = FishDataset(os.path.join(clean_data_path, 'val'), preprocess=transforms, fraction=subsample_fraction,
                             filter_species=filter_species)

    # # create training and validation set dataloaders
    trainDataLoader = DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=True)
    print('Train DataLoader length:', len(trainDataLoader))
    valDataLoader = DataLoader(valDataset, batch_size=BATCH_SIZE)
    print('Val DataLoader length:', len(valDataLoader))

    #################### Define class weights ###############################3

    if weights == 'inverse_quantities':
        class_weights = trainDataset.getClassWeights()
    else:
        class_weights = torch.Tensor(total_classes * [1.0])

    print('class weights:', class_weights)

    #################### Loss ###############################3
    print('Regression type: ', regression)

    if regression == 'continuous' or regression == 'ordinal':
        criterion = weighted_RMSELoss(class_weights).to(device)
        # criterion = nn.CrossEntropyLoss(reduction='mean')
    else:
        # criterion = nn.CrossEntropyLoss(reduction='mean')
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).to(device), reduction='mean')

    print('Loss function: ', criterion)

    metric_conf_matrix = MulticlassConfusionMatrix(num_classes=total_classes)

    metric = evaluate.load("accuracy")

    #################### Optimizer, Scheduler ###############################3

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # optimizer = SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)

    scheduler = LinearLR(optimizer, total_iters=epochs)
    print('Weight decay:', weight_decay)
    print('Learning rate:', learning_rate)
    #################### Train process ###############################3

    config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="none",
        modules_to_save=["classifier"],
    )
    lora_model = get_peft_model(model, config)
    print_trainable_parameters(lora_model)
    "trainable params: 667493 || all params: 86466149 || trainable%: 0.77"

    model_name_part = model_name.split("/")[-1]

    data_loader_config = DataLoaderConfiguration(dispatch_batches=None, split_batches=False)

    args = TrainingArguments(
        f"{model_name_part}-finetuned-lora-otoliths",
        remove_unused_columns=False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        # eval_steps=1,
        learning_rate=learning_rate,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        per_device_eval_batch_size=BATCH_SIZE,
        fp16=False,
        bf16=False,
        num_train_epochs=epochs,
        logging_strategy='epoch',
        # logging_steps=1,
        load_best_model_at_end=True,
        do_train=True,
        do_eval=True,
        metric_for_best_model="accuracy",
        # push_to_hub=True,
        label_names=["labels"],
    )

    trainer = Trainer(
        lora_model,
        args,
        train_dataset=trainDataset,
        eval_dataset=valDataset,
        tokenizer=image_processor,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )

    train_results = trainer.train()
    # print(train_results)
    # print(trainer.state.log_history)
    # trainer.evaluate(eval_dataset=valDataset)
