## 文件夹内容如下：
**tutorialCode**: kaggle教程练习代码

**TutorialData**: 数据集

**results**: 各模型跑出的预测数据

**JupyterNotebook**: 学习过程notebook

**imdb_sentiment_analysis_torch**:各模型运行代码

**myCode**:自己编写的模型代码

**imdb_deberta**:deberta微调代码

**imdb_bert**:bert相关代码

## 以下结果通过RTX 5080 Laptop GPU得出：
| 模型                 | 训练准确率 | 验证准确率 | 总时间 | 备注                                       |
|--------------------|-----------|-----------|--------|------------------------------------------|
| **imdb models**    |
| Attention_LSTM     | 91% | 88% | 80秒 | 速度较快,准确率较高                               |
| BERT_Native        | 96% | 92% | 1500秒 | 速度较慢,显存占用高,准确率高                          |
| BERT_Scratch       | 93% | 93% | 1500秒 | 速度较慢,显存占用高,修正前向传播总是测试算loss问题             |
| BERT_Trainer       | 94% | 94% | 1500秒 | 速度较慢,显存占用高,准确率略较其他BERT高                  |
| BERT_RDrop         | 92.92% | 92.92%| 43分53秒| 速度慢，效果好                    |
| BERT_SCL           | 91.64% | 91.64%| 27分19秒| 速度慢，效果好|
| Capsule_LSTM       | 97% | 90% | 150秒 | 学习率lr=0.01时准确率显著高于lr=0.1                 |
| CNN                | 87% | 82% | 10秒 | 速度非常快,准确率一般                              |
| CNN_LSTM           | 93% | 85% | 15秒 | 速度非常快,比纯CNN效果有一定提升                       |
| DistilBERT_Native  | 97% | 91% | 750秒 | 速度较慢,显存占用非常高,比起BERT更轻量化但时效果不输            |
| DistilBERT_Trainer | 93% | 93% | 750秒 | 速度较慢,显存占用非常高,比起BERT更轻量化但时效果不输            |
| GRU                | 89% | 86% | 50秒 | 速度较快,准确率还行                               |
| LSTM               | 81% | 79% | 80秒 | 速度较快,准确率一般                               |
| RoBERTa_Trainer    | 94% | 94% | 1500秒 | 准确率高,显存占用高                               |
| Transformer        | 96% | 88% | 500秒 | 显存需求大,使用GloVe前准确率只有50%,使用后提升明显           |
| Deberta_prompt     | 71.42% | 71.28% | 2小时18分 | kaggle运行，速度慢，效果一般，训练参数是原模型的0.0024%       |
| Deberta_LoRA       | 96.02% | 96.88% | 2小时20分 | kaggle运行，速度慢，效果全场最佳，训练参数是原模型的0.1809%     |
| Deberta_ptuning    | 58.22% | 57.96% | 2小时10分 | kaggle运行，速度慢，效果差，训练参数是原模型的0.0694%        |
| Deberta_prefix     | / | / | /| 模型不支持 Prefix Tuning                      |
| Deberta_unsloth    | 96.42% | 96.42% | 4小时18分| 速度慢，训练效果最好，相比其他微调方法，显存占用更小，训练参数是原模型的0.6% |
| **tutorial models**     |
| BOW                     | / | 84% | 42秒 | 速度较快,准确率还行                         |
| Word2Vec+AverageVectors | / | 83% | 282秒 | 速度较快,准确率还行                         |
| Word2Vec+Kmeans         | / | 85% | | 速度一般,准确率还行                         |

## Kaggle提交结果

| 模型                      | Private Score | Public Score | 备注              |
|-------------------------|--------------|--------------|-----------------|
| BagOfCentroids          | 0.84696 | 0.84696 | -               |
| Bag_of_Words_model      | 0.84404 | 0.84404 | -               |
| Attention_LSTM          | 0.83892 | 0.83892 | -               |
| Deberta_ptuning         | 0.58144 | 0.58144 | 效果较差            |
| Deberta_prompt          | 0.70604 | 0.70604 | -               |
| Deberta_LoRA            | 0.96768 | 0.96768 | **最高分**         |
| Deberta_unsloth         | 0.81584 | 0.81584 | 训练效果好但得分低，疑似过拟合 |
| Word2Vec_AverageVectors | 0.82864 | 0.82864 | -               |
| Transformer_GloVe       | 0.86288 | 0.86288 | -               |
| Transformer             | 0.57952 | 0.57952 | 未使用GloVe，效果差    |
| RoBERTa_Trainer         | 0.95264 | 0.95264 | -               |
| LSTM                    | 0.78880 | 0.78880 | -               |
| GRU                     | 0.81976 | 0.81976 | -               |
| DistilBERT_Trainer      | 0.92984 | 0.92984 | -               |
| DistilBERT_Native       | 0.91340 | 0.91340 | -               |
| CNN_LSTM                | 0.84332 | 0.84332 | -               |
| CNN                     | 0.69860 | 0.69860 | -               |
| Capsule_LSTM            | 0.88368 | 0.88368 | -               |
| BERT_Trainer            | 0.93900 | 0.93900 | -               |
| BERT_Scratch            | 0.93452 | 0.93452 | -               |
| BERT_Native             | 0.91116 | 0.91116 | -               |
| BERT_RDrop              | 0.93776 | 0.93776 | -               |
| BERT_SCL                | 0.91620 | 0.91620 | -               |
