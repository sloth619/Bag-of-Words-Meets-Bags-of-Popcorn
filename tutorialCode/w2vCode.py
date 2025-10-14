import re
import pandas as pd
import nltk.data
import numpy as np
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from gensim.models import word2vec
from sklearn.ensemble import RandomForestClassifier
import time
from sklearn.cluster import KMeans

# 加载NLTK的分词器
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# 将影评文本转换为单词列表
def review_to_wordlist(review, remove_stopwords=False):
    review_text = BeautifulSoup(review, "lxml").get_text()
    review_text = re.sub("[^a-zA-Z]", " ", review_text)
    words = review_text.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return words

# 将影评分割成句子列表，每个句子又是一个单词列表
def review_to_sentences(review, tokenizer, remove_stopwords=False):
    raw_sentences = tokenizer.tokenize(review.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(review_to_wordlist(raw_sentence, remove_stopwords))
    return sentences

# 数据加载
train = pd.read_csv("../tutorialData/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv("../tutorialData/testData.tsv", header=0, delimiter="\t", quoting=3)
unlabeled_train = pd.read_csv("../tutorialData/unlabeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
print("读取了 %d 条有标签的训练评论, %d 条有标签的测试评论, 以及 %d 条无标签的评论" % (
    train["review"].size, test["review"].size, unlabeled_train["review"].size))

# 准备Word2Vec的输入数据
sentences = []  # 初始化一个空列表来存储所有句子

print("正在从训练集中解析句子...")
for review in train["review"]:
    sentences += review_to_sentences(review, tokenizer)

print("正在从无标签训练集中解析句子...")
for review in unlabeled_train["review"]:
    sentences += review_to_sentences(review, tokenizer)

# 训练Word2Vec模型
# 设置Word2Vec模型参数
num_features = 300    # 词向量维度
min_word_count = 40   # 最小词频
num_workers = 4       # 并行运行的线程数
context = 10          # 上下文窗口大小
downsampling = 1e-3   # 高频词的下采样设置

print("正在训练Word2Vec模型...")
model = word2vec.Word2Vec(sentences, workers=num_workers, vector_size=num_features,
                          min_count=min_word_count, window=context, sample=downsampling)

# 保存模型
model_name = "300features_40minwords_10context"
model.save(model_name)
print(f"Word2Vec模型已保存为: {model_name}")


# 将影评转换为特征向量
def makeFeatureVec(words, model, num_features):
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0.
    index2word_set = set(model.wv.index_to_key)
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec, model.wv[word])
    if nwords > 0:
        featureVec = np.divide(featureVec, nwords)
    return featureVec

def getAvgFeatureVecs(reviews, model, num_features):
    counter = 0
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")
    for review in reviews:
        if counter % 5000 == 0.:
            print("正在处理第 %d 条评论，共 %d 条" % (counter, len(reviews)))
        reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
        counter = counter + 1
    return reviewFeatureVecs

# 计算训练集和测试集的平均特征向量
print("正在为训练评论创建平均特征向量...")
clean_train_reviews = []
for review in train["review"]:
    clean_train_reviews.append(review_to_wordlist(review, remove_stopwords=True))

trainDataVecs = getAvgFeatureVecs(clean_train_reviews, model, num_features)

print("正在为测试评论创建平均特征向量...")
clean_test_reviews = []
for review in test["review"]:
    clean_test_reviews.append(review_to_wordlist(review, remove_stopwords=True))

testDataVecs = getAvgFeatureVecs(clean_test_reviews, model, num_features)

# 训练随机森林分类器并进行预测
# 初始化一个有100棵树的随机森林分类器
forest = RandomForestClassifier(n_estimators=100)
print("正在用带标签的训练数据拟合随机森林模型...")
forest = forest.fit(trainDataVecs, train["sentiment"])
result = forest.predict(testDataVecs)
output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
output_path = "../results/Word2Vec_AverageVectors.csv"
output.to_csv(output_path, index=False, quoting=3)
print(f"预测结果已保存至: {output_path}")

# 用聚类对向量进行分组
start = time.time()
word_vectors = model.wv.vectors
num_clusters = word_vectors.shape[0] // 5
kmeans_clustering = KMeans( n_clusters = num_clusters, n_init='auto' )
idx = kmeans_clustering.fit_predict( word_vectors )
end = time.time()
elapsed = end - start
print ("KMeans 聚类耗时: ", elapsed, "秒。")

# 创建一个字典，将词汇表中的每个单词映射到其所属的聚类编号
word_centroid_map = dict(zip( model.wv.index_to_key, idx ))

# 遍历前10个聚类
for cluster in range(0,10):
    print ("\nCluster %d" % cluster)
    words = []
    all_keys = list(word_centroid_map.keys())
    all_values = list(word_centroid_map.values())
    for i in range(0,len(all_values)):
        if( all_values[i] == cluster ):
            words.append(all_keys[i])
    print (words)

# 创建质心袋
def create_bag_of_centroids( wordlist, word_centroid_map ):
    num_centroids = max( word_centroid_map.values() ) + 1
    bag_of_centroids = np.zeros( num_centroids, dtype="float32" )
    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1
    return bag_of_centroids

# 为每条训练评论创建质心袋
train_centroids = np.zeros( (train["review"].size, num_clusters), \
    dtype="float32" )
counter = 0
for review in clean_train_reviews:
    train_centroids[counter] = create_bag_of_centroids( review, \
        word_centroid_map )
    counter += 1
test_centroids = np.zeros(( test["review"].size, num_clusters), \
    dtype="float32" )
counter = 0
for review in clean_test_reviews:
    test_centroids[counter] = create_bag_of_centroids( review, \
        word_centroid_map )
    counter += 1

# 拟合随机森林模型并提取预测结果
forest = RandomForestClassifier(n_estimators = 100)
print ("正在将随机森林模型拟合到带标签的训练数据上...")
forest = forest.fit(train_centroids,train["sentiment"])
result = forest.predict(test_centroids)
output = pd.DataFrame(data={"id":test["id"], "sentiment":result})
output.to_csv( "../results/BagOfCentroids.csv", index=False, quoting=3 )
print ("写入完成")