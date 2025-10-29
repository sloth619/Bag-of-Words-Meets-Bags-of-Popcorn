import os

os.environ["UNSLOTH_DISABLE_STATS"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
import torch.nn as nn
import sys
import logging
import evaluate
import pandas as pd
import numpy as np
import gc
from typing import Optional, Tuple, Union
from dataclasses import dataclass

from unsloth import FastModel
from transformers import (
    TrainingArguments,
    Trainer,
    AutoModel,
    AutoConfig,
    PreTrainedModel,
    training_args
)
from transformers.modeling_outputs import SequenceClassifierOutput
from datasets import Dataset
from sklearn.model_selection import train_test_split


# ===================== SupConLoss =====================
class SupConLoss(nn.Module):
    """Supervised Contrastive Learning Loss"""

    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None):
        """
        Args:
            features: [bsz, hidden_dim]
            labels: [bsz]
        """
        device = features.device
        features = nn.functional.normalize(features, dim=1)

        batch_size = features.shape[0]
        if batch_size < 2:
            return torch.tensor(0.0, device=device)

        if labels is not None:
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)

        # Compute similarity
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Mask out self-contrast
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # Compute loss
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        mask_sum = torch.clamp(mask.sum(1), min=1.0)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum

        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        return loss.mean()


# ===================== 自定义模型类（继承HuggingFace） =====================
class BertForSequenceClassificationWithSCL(PreTrainedModel):
    """
    继承自 HuggingFace 的 PreTrainedModel
    在 forward 中实现 SCL loss
    """

    def __init__(self, config, alpha=0.2):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.alpha = alpha

        # 加载 BERT backbone
        self.bert = AutoModel.from_config(config)

        # Classifier
        classifier_dropout = (
            config.classifier_dropout
            if hasattr(config, 'classifier_dropout') and config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # SCL loss
        self.scl_loss_fct = SupConLoss()

        # Initialize weights
        self.post_init()

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        """
        重写 forward 方法，在这里实现 SCL loss
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # BERT forward - 只在训练时输出hidden_states
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=self.training,  # 关键改动：只在训练时输出
            return_dict=return_dict,
        )

        # 获取 [CLS] token 的表示
        pooled_output = outputs[1]  # pooler_output

        # Dropout + Classifier
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # 计算 loss
        loss = None
        if labels is not None:
            # Cross-entropy loss
            loss_fct = nn.CrossEntropyLoss()
            ce_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            # 只在训练时计算 Supervised contrastive loss
            if self.training and labels.size(0) >= 2:
                # 使用最后一层的 [CLS] token hidden state
                last_hidden_state = outputs.hidden_states[-1]  # [bsz, seq_len, hidden_dim]
                cls_features = last_hidden_state[:, 0, :]  # [bsz, hidden_dim]

                scl_loss = self.scl_loss_fct(cls_features, labels)

                # Combined loss
                loss = ce_loss + self.alpha * scl_loss
            else:
                loss = ce_loss

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if self.training else None,
            attentions=outputs.attentions,
        )


# ===================== 主程序 =====================
if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(
        format='%(asctime)s: %(levelname)s: %(message)s',
        level=logging.INFO
    )
    logger.info(r"running %s" % ''.join(sys.argv))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        logger.info(f"GPU: {gpu_name}")
        logger.info(f"Total GPU Memory: {gpu_mem_total:.2f}GB")

    logger.info("Loading data...")
    train = pd.read_csv("../tutorialData/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
    test = pd.read_csv("../tutorialData/testData.tsv", header=0, delimiter="\t", quoting=3)

    train, val = train_test_split(train, test_size=.2, random_state=42)

    train_dict = {'label': train["sentiment"].values, 'text': train['review'].values}
    val_dict = {'label': val["sentiment"].values, 'text': val['review'].values}
    test_dict = {"text": test['review'].values}

    train_dataset = Dataset.from_dict(train_dict)
    val_dataset = Dataset.from_dict(val_dict)
    test_dataset = Dataset.from_dict(test_dict)

    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # ===================== 使用自定义模型类 =====================
    model_name = 'bert-base-uncased'
    NUM_CLASSES = 2

    logger.info("Loading config and creating custom model...")

    # 1. 加载配置
    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = NUM_CLASSES

    # 2. 创建自定义模型实例
    base_model = BertForSequenceClassificationWithSCL(config, alpha=0.2)

    # 3. 加载预训练权重到 BERT backbone
    pretrained_bert = AutoModel.from_pretrained(model_name)
    base_model.bert = pretrained_bert

    logger.info("Model structure created, now applying Unsloth optimization...")

    # 4. 使用 Unsloth 优化
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        model = base_model

        logger.info("Using custom model directly")

        try:
            from peft import get_peft_model, LoraConfig, TaskType

            peft_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                inference_mode=False,
                r=8,
                lora_alpha=16,
                lora_dropout=0.1,
                target_modules=["query", "key", "value", "dense"],  # BERT 的注意力层
            )

            model = get_peft_model(model, peft_config)
            logger.info("LoRA applied successfully")
            model.print_trainable_parameters()

        except ImportError:
            logger.warning("PEFT not available, using full fine-tuning")

    except Exception as e:
        logger.warning(f"Could not use FastModel wrapper: {e}")
        logger.info("Proceeding with standard HuggingFace model")
        model = base_model
        tokenizer = AutoTokenizer.from_pretrained(model_name)


    # Tokenization
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            max_length=512,
            truncation=True,
            padding=False
        )


    logger.info("Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    # Metrics
    metric = evaluate.load("accuracy")


    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)


    # Training arguments
    logger.info("Setting up training...")
    training_args = TrainingArguments(
        output_dir='./checkpoint_scl_custom',
        per_device_train_batch_size=8,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=500,
        num_train_epochs=3,
        learning_rate=2e-5,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=100,
        logging_dir='./logs_scl_custom',
        save_strategy="epoch",
        save_total_limit=2,
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        seed=3407,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train
    logger.info("=" * 50)
    logger.info("Starting training with custom model...")
    logger.info("=" * 50)

    try:
        trainer_stats = trainer.train()
        logger.info(f"Training completed!")

        # Evaluate
        eval_results = trainer.evaluate()
        logger.info(f"Validation results: {eval_results}")

        # Predict
        logger.info("Predicting on test set...")

        model.eval()
        test_predictions = []

        from torch.utils.data import DataLoader

        test_dataloader = DataLoader(
            test_dataset.remove_columns(['text']),
            batch_size=16,
            collate_fn=lambda x: tokenizer.pad(
                {k: [d[k] for d in x] for k in x[0].keys()},
                return_tensors='pt'
            )
        )

        with torch.no_grad():
            for batch in test_dataloader:
                batch = {k: v.to(model.device) for k, v in batch.items()}
                outputs = model(**batch)
                predictions = torch.argmax(outputs.logits, dim=-1)
                test_predictions.extend(predictions.cpu().numpy())

        test_pred = np.array(test_predictions)

        os.makedirs("../results", exist_ok=True)
        result_output = pd.DataFrame(data={"id": test["id"], "sentiment": test_pred})
        result_output.to_csv("../results/bert_scl_custom_model.csv", index=False, quoting=3)
        logger.info('Results saved!')

        model.save_pretrained("./bert_scl_custom_final")
        tokenizer.save_pretrained("./bert_scl_custom_final")
        logger.info('Model saved!')

    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()