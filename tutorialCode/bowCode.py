import re
import pandas as pd
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer

# 将原始评论转换为单词字符串
def review_to_words(raw_review):
    review_text = BeautifulSoup(raw_review, "lxml").get_text()
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    words = letters_only.lower().split()
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if not w in stops]
    return " ".join(meaningful_words)


# 训练数据预处理
train = pd.read_csv("../tutorialData/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
num_reviews = train["review"].size
clean_train_reviews = []
print("正在清理和解析训练集的电影评论...")
for i in range(0, num_reviews):
    if ((i+1) % 5000 == 0):
        print("正在处理第 %d 条评论，共 %d 条\n" % (i+1, num_reviews))
    clean_train_reviews.append(review_to_words(train["review"][i]))

# 创建词袋模型
print("正在创建词袋模型...")
vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, max_features=5000)
train_data_features = vectorizer.fit_transform(clean_train_reviews)
train_data_features = train_data_features.toarray()

# 训练随机森林模型
print("正在训练随机森林模型...")
forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit(train_data_features, train["sentiment"])

# 测试集数据预处理
test = pd.read_csv("../tutorialData/testData.tsv", header=0, delimiter="\t", quoting=3)
num_reviews = len(test["review"])
clean_test_reviews = []
print("正在清理和解析测试集的电影评论...")
for i in range(0, num_reviews):
    if( (i+1) % 5000 == 0 ):
        print("已处理 %d 条评论，共 %d 条\n" % (i+1, num_reviews))
    clean_review = review_to_words(test["review"][i])
    clean_test_reviews.append(clean_review)

# 随机森林预测
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()
print("正在使用随机森林进行预测...")
result = forest.predict(test_data_features)
output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
output.to_csv("../results/Bag_of_Words_model.csv", index=False, quoting=3)
print("结果已保存！")