import re
import time
import pandas as pd
import numpy as np
import nltk.data
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from gensim.models import word2vec
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



def review_to_wordlist(review, remove_stopwords=False):
    """将HTML评论转换为一个单词列表"""
    review_text = BeautifulSoup(review, "lxml").get_text()
    review_text = re.sub("[^a-zA-Z]", " ", review_text)
    words = review_text.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return words


def review_to_sentences(review, tokenizer, remove_stopwords=False):
    """将评论分割成句子列表，每个句子又是一个单词列表"""
    raw_sentences = tokenizer.tokenize(review.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(review_to_wordlist(raw_sentence, remove_stopwords))
    return sentences


def makeFeatureVec(words, model, num_features):
    """将单词列表转换为平均特征向量"""
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
    """为一组评论计算平均特征向量"""
    counter = 0
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")
    for review in reviews:
        reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
        counter = counter + 1
    return reviewFeatureVecs


def create_bag_of_centroids(wordlist, word_centroid_map):
    """为单词列表创建质心袋特征"""
    num_centroids = max(word_centroid_map.values()) + 1
    bag_of_centroids = np.zeros(num_centroids, dtype="float32")
    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1
    return bag_of_centroids


if __name__ == '__main__':
    total_start_time = time.time()
    # 加载NLTK的分词器
    tokenizer = nltk.tokenize.punkt.PunktSentenceTokenizer()

    # --- 1. 数据加载与划分 ---
    print("正在加载数据...")
    train_df = pd.read_csv("../tutorialData/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
    test_df = pd.read_csv("../tutorialData/testData.tsv", header=0, delimiter="\t", quoting=3)
    unlabeled_train_df = pd.read_csv("../tutorialData/unlabeledTrainData.tsv", header=0, delimiter="\t", quoting=3)

    # 划分训练集和验证集
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        train_df['review'], train_df['sentiment'], test_size=0.2, random_state=42
    )
    print(
        f"数据划分完成: {len(X_train_raw)}条训练, {len(X_val_raw)}条验证, {len(unlabeled_train_df)}条无标签, {len(test_df)}条测试。")

    # --- 2. 训练 Word2Vec 模型 ---
    # 准备Word2Vec的输入数据 (使用新的训练集 + 无标签数据)
    w2v_parsing_start_time = time.time()
    sentences = []
    print("\n正在从训练集和无标签数据中解析句子...")
    for review in X_train_raw:
        sentences += review_to_sentences(review, tokenizer)
    for review in unlabeled_train_df["review"]:
        sentences += review_to_sentences(review, tokenizer)
    w2v_parsing_duration = time.time() - w2v_parsing_start_time
    print(f"句子解析完成，耗时: {w2v_parsing_duration:.2f} 秒")

    # 训练Word2Vec模型
    num_features = 300
    min_word_count = 40
    num_workers = 4
    context = 10
    downsampling = 1e-3

    print("\n正在训练Word2Vec模型...")
    w2v_training_start_time = time.time()
    model = word2vec.Word2Vec(sentences, workers=num_workers, vector_size=num_features,
                              min_count=min_word_count, window=context, sample=downsampling)
    w2v_training_duration = time.time() - w2v_training_start_time
    model_name = "300features_40minwords_10context"
    model.save(model_name)
    print(f"Word2Vec模型训练完成并已保存，耗时: {w2v_training_duration:.2f} 秒")

    # --- 3. 模型一: 平均词向量 + 随机森林 ---
    print("\n--- 开始处理模型一：平均词向量法 ---")

    # 清洗训练和验证集评论
    print("正在清洗训练集和验证集评论...")
    clean_train_reviews = [review_to_wordlist(review, remove_stopwords=True) for review in X_train_raw]
    clean_val_reviews = [review_to_wordlist(review, remove_stopwords=True) for review in X_val_raw]

    # 为训练集创建平均特征向量
    print("正在为训练集创建平均特征向量...")
    trainDataVecs = getAvgFeatureVecs(clean_train_reviews, model, num_features)

    # 训练随机森林
    print("正在训练随机森林分类器...")
    rf1_training_start_time = time.time()
    forest1 = RandomForestClassifier(n_estimators=100, random_state=42)
    forest1 = forest1.fit(trainDataVecs, y_train)
    rf1_training_duration = time.time() - rf1_training_start_time
    print(f"模型一训练完成，耗时: {rf1_training_duration:.2f} 秒")

    # 在验证集上评估准确率
    print("正在验证集上评估模型一...")
    valDataVecs = getAvgFeatureVecs(clean_val_reviews, model, num_features)
    val_predictions1 = forest1.predict(valDataVecs)
    accuracy1 = accuracy_score(y_val, val_predictions1)
    print(f"模型一 (平均向量法) 在验证集上的准确率是: {accuracy1 * 100:.2f}%")

    # 对官方测试集进行预测
    print("正在为官方测试集生成预测结果...")
    clean_test_reviews = [review_to_wordlist(review, remove_stopwords=True) for review in test_df["review"]]
    testDataVecs = getAvgFeatureVecs(clean_test_reviews, model, num_features)
    result1 = forest1.predict(testDataVecs)
    output1 = pd.DataFrame(data={"id": test_df["id"], "sentiment": result1})
    output1.to_csv("../results/Word2Vec_AverageVectors.csv", index=False, quoting=3)
    print("模型一预测结果已保存！")

    # --- 4. 模型二: 质心袋 + 随机森林 ---
    print("\n--- 开始处理模型二：质心袋法 ---")

    # 使用KMeans进行聚类
    print("正在使用KMeans进行聚类...")
    kmeans_start_time = time.time()
    word_vectors = model.wv.vectors
    num_clusters = word_vectors.shape[0] // 5
    kmeans_clustering = KMeans(n_clusters=num_clusters, n_init='auto', random_state=42)
    idx = kmeans_clustering.fit_predict(word_vectors)
    kmeans_duration = time.time() - kmeans_start_time
    print(f"KMeans聚类完成，耗时: {kmeans_duration:.2f} 秒")
    word_centroid_map = dict(zip(model.wv.index_to_key, idx))

    # 为训练集和验证集创建质心袋
    print("正在为训练集和验证集创建质心袋特征...")
    train_centroids = np.array([create_bag_of_centroids(review, word_centroid_map) for review in clean_train_reviews])
    val_centroids = np.array([create_bag_of_centroids(review, word_centroid_map) for review in clean_val_reviews])

    # 训练随机森林
    print("正在训练随机森林分类器...")
    rf2_training_start_time = time.time()
    forest2 = RandomForestClassifier(n_estimators=100, random_state=42)
    forest2 = forest2.fit(train_centroids, y_train)
    rf2_training_duration = time.time() - rf2_training_start_time
    print(f"模型二训练完成，耗时: {rf2_training_duration:.2f} 秒")

    # 在验证集上评估准确率
    print("正在验证集上评估模型二...")
    val_predictions2 = forest2.predict(val_centroids)
    accuracy2 = accuracy_score(y_val, val_predictions2)
    print(f"模型二 (质心袋法) 在验证集上的准确率是: {accuracy2 * 100:.2f}%")

    # 对官方测试集进行预测
    print("正在为官方测试集生成预测结果...")
    test_centroids = np.array([create_bag_of_centroids(review, word_centroid_map) for review in clean_test_reviews])
    result2 = forest2.predict(test_centroids)
    output2 = pd.DataFrame(data={"id": test_df["id"], "sentiment": result2})
    output2.to_csv("../results/BagOfCentroids.csv", index=False, quoting=3)
    print("模型二预测结果已保存！")

    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    print(f"\n脚本总运行时间: {total_duration:.2f} 秒")