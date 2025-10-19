import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer,AutoModelForSequenceClassification, TrainingArguments, Trainer
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 加载 TSV 数据集
train = pd.read_csv("../../tutorialData/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv("../../tutorialData/testData.tsv", header=0, delimiter="\t", quoting=3)
train_datasets =Dataset.from_pandas(train)  #将pandas.read_csv() 读出来的 DataFrame 转成 Hugging Face 框架专用的 datasets.Dataset 对象
test_datasets =Dataset.from_pandas(test)
train_datasets = train_datasets.rename_column("sentiment", "labels")
dataset = train_datasets.train_test_split(test_size=0.2,seed=42) #划分出 20% 的样本作为验证集,seed为了复现

#加载 Tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
def tokenize_function(examples):
    return tokenizer(examples["review"],padding="max_length",truncation=True)
tokenized_datasets = dataset.map(tokenize_function)

#加载模型
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

#评估函数
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels)
    return {"accuracy": accuracy["accuracy"], "f1": f1["f1"]}

# 训练参数
train_args = TrainingArguments(output_dir="./checkpoints",      # 输出文件夹
                               per_device_train_batch_size=16,  # 训练时的batch_size
                               per_device_eval_batch_size=32,   # 验证时的batch_size
                               num_train_epochs=3,              # 训练轮次
                               fp16=True,                       # 半精度
                               eval_strategy="epoch",           # 评估策略
                               save_strategy="epoch",           # 保存策略
                               save_total_limit=3,              # 最大保存数
                               learning_rate=2e-5,              # 学习率
                               weight_decay=0.01,               # weight_decay
                               metric_for_best_model="f1",      # 设定评估指标
                               load_best_model_at_end=True)     # 训练结束后自动加载最佳模型

#创建 Trainer
trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

#训练模型
trainer.train()
train_results = trainer.evaluate(eval_dataset=tokenized_datasets["train"])
print("训练集准确率:", train_results["eval_accuracy"])
val_results = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
print("验证集准确率:", val_results["eval_accuracy"])

# 加载最佳模型
best_model_path = trainer.state.best_model_checkpoint
if best_model_path:
    model = AutoModelForSequenceClassification.from_pretrained(best_model_path)

# 将模型移到GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()  # 设置为评估模式

# 测试单条影评
test_text = "The movie was really good!"
inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True)
inputs = {k: v.to(device) for k, v in inputs.items()}
with torch.no_grad():
    outputs = model(**inputs)
prediction = outputs.logits.argmax(-1).item()
print("情感预测结果:", "正面" if prediction == 1 else "负面")