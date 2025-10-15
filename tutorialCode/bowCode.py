import re
import time
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def review_to_words(raw_review):
    """将原始HTML评论转换为一个由有意义的单词组成的字符串"""
    review_text = BeautifulSoup(raw_review, "lxml").get_text()
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    words = letters_only.lower().split()
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if not w in stops]
    return " ".join(meaningful_words)


if __name__ == '__main__':
    total_start_time = time.time()

    # --- 1. 数据加载与划分 ---
    print("正在加载和划分数据...")
    train_df = pd.read_csv("../tutorialData/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)

    # 将原始训练数据划分为新的训练集(80%)和验证集(20%)
    # X 是评论 (features), y 是情感标签 (labels)
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        train_df['review'], train_df['sentiment'], test_size=0.2, random_state=42
    )
    print(f"数据划分完成：{len(X_train_raw)}条用于训练, {len(X_val_raw)}条用于验证。")

    # --- 2. 训练数据预处理 ---
    print("\n正在清理和解析训练集的电影评论...")
    cleaning_start_time = time.time()

    clean_train_reviews = []
    for review in X_train_raw:
        clean_train_reviews.append(review_to_words(review))

    cleaning_end_time = time.time()
    cleaning_duration = cleaning_end_time - cleaning_start_time
    print(f"训练集清洗完成，耗时: {cleaning_duration:.2f} 秒")

    # --- 3. 创建词袋模型 (特征提取) ---
    print("\n正在创建词袋模型...")
    vectorizer_start_time = time.time()

    vectorizer = CountVectorizer(analyzer="word", max_features=5000)
    # Fit a-n-d transform the training data
    train_data_features = vectorizer.fit_transform(clean_train_reviews)
    train_data_features = train_data_features.toarray()

    vectorizer_end_time = time.time()
    vectorizer_duration = vectorizer_end_time - vectorizer_start_time
    print(f"词袋特征提取完成，耗时: {vectorizer_duration:.2f} 秒")

    # --- 4. 训练随机森林模型 ---
    print("\n正在训练随机森林模型...")
    training_start_time = time.time()

    forest = RandomForestClassifier(n_estimators=100, random_state=42)
    forest = forest.fit(train_data_features, y_train)

    training_end_time = time.time()
    training_duration = training_end_time - training_start_time
    print(f"随机森林训练完成，耗时: {training_duration:.2f} 秒")

    # --- 5. 在验证集上评估模型准确率 ---
    print("\n正在验证集上评估模型...")
    # 清洗验证集数据
    clean_val_reviews = [review_to_words(review) for review in X_val_raw]
    # 使用已经fit过的vectorizer来transform验证集数据
    val_data_features = vectorizer.transform(clean_val_reviews).toarray()

    # 进行预测
    val_predictions = forest.predict(val_data_features)

    # 计算准确率
    accuracy = accuracy_score(y_val, val_predictions)
    print(f"模型在验证集上的准确率是: {accuracy * 100:.2f}%")

    # --- 6. 对官方测试集进行预测并保存结果 ---
    print("\n正在处理官方测试集并进行最终预测...")
    prediction_start_time = time.time()

    test_df = pd.read_csv("../tutorialData/testData.tsv", header=0, delimiter="\t", quoting=3)
    clean_test_reviews = [review_to_words(review) for review in test_df["review"]]
    test_data_features = vectorizer.transform(clean_test_reviews).toarray()

    result = forest.predict(test_data_features)

    output = pd.DataFrame(data={"id": test_df["id"], "sentiment": result})
    output.to_csv("../results/Bag_of_Words_model.csv", index=False, quoting=3)

    prediction_end_time = time.time()
    prediction_duration = prediction_end_time - prediction_start_time
    print(f"官方测试集预测完成并已保存结果，耗时: {prediction_duration:.2f} 秒")

    # --- 7. 打印总运行时间 ---
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    print(f"\n脚本总运行时间: {total_duration:.2f} 秒")