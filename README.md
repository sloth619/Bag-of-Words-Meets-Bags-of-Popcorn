## 文件夹内容如下：
**tutorialCode**: kaggle教程练习代码

**TutorialData**: 数据集

**results**: 各模型跑出的预测数据

**JupyterNotebook**: 学习过程notebook

**imdb_sentiment_analysis_torch**:各模型运行代码

**myCode**:自己编写的模型代码

**imdb_deberta**:deberta微调代码

## 以下结果通过RTX 5080 Laptop GPU得出：
| 模型 | 训练准确率 | 验证准确率 | 总时间 | 备注                                 |
|------|-----------|-----------|--------|------------------------------------|
| **imdb models** |
| Attention_LSTM | 91% | 88% | 80秒 | 速度较快,准确率较高                         |
| BERT_Native | 96% | 92% | 1500秒 | 速度非常慢,显存占用高,准确率高                   |
| BERT_Scratch | 93% | 93% | 1500秒 | 速度非常慢,显存占用高,修正前向传播总是测试算loss问题      |
| BERT_Trainer | 94% | 94% | 1500秒 | 速度非常慢,显存占用高,准确率略较其他BERT高           |
| Capsule_LSTM | 97% | 90% | 150秒 | 学习率lr=0.01时准确率显著高于lr=0.1           |
| CNN | 87% | 82% | 10秒 | 速度非常快,准确率一般                        |
| CNN_LSTM | 93% | 85% | 15秒 | 速度非常快,比纯CNN效果有一定提升                 |
| DistilBERT_Native | 97% | 91% | 750秒 | 速度较慢,显存占用非常高,比起BERT更轻量化但时效果不输      |
| DistilBERT_Trainer | 93% | 93% | 750秒 | 速度较慢,显存占用非常高,比起BERT更轻量化但时效果不输      |
| GRU | 89% | 86% | 50秒 | 速度较快,准确率还行                         |
| LSTM | 81% | 79% | 80秒 | 速度较快,准确率一般                         |
| RoBERTa_Trainer | 94% | 94% | 1500秒 | 准确率高,显存占用高                         |
| Transformer | 96% | 88% | 500秒 | 显存需求大,使用GloVe前准确率只有50%,使用后提升明显     |
| Deberta_prompt | 71.42% | 71.28% | 2小时18分 | kaggle运行，速度慢，效果一般，训练参数是原模型的0.0024% |
| Deberta_LoRA | 96.02% | 96.88% | 2小时20分 | kaggle运行，速度慢，效果全场最佳，训练参数是原模型的0.1809% |
| Deberta_ptuning | 58.22% | 57.96% | 2小时10分 | kaggle运行，速度慢，效果差，训练参数是原模型的0.0694% |
| Deberta_prefix | / | / | /| 模型不支持 Prefix Tuning |
| **tutorial models** |
| BOW | / | 84% | 42秒 | 速度较快,准确率还行                         |
| Word2Vec+AverageVectors | / | 83% | 282秒 | 速度较快,准确率还行                         |
| Word2Vec+Kmeans | / | 85% | | 速度一般,准确率还行                         |
