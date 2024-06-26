# pip3 install openai
from openai import OpenAI
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

# OpenAI APIキーの設定
os.environ["OPENAI_API_KEY"] = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# OpenAIクライアントの初期化
client = OpenAI()

def vectorize_text(text):
    response = client.embeddings.create(
        input = text,
        # モデルの指定 (様々なモデルが利用可能)
        model = "text-embedding-3-small"
    )
    return response.data[0].embedding

# 回答データベース
documents = [
    "システム運用事業部",
    "販売管理システム",
    "第一システム部",
    "システム開発事業部"
]

def rag_sample(question, documents):

    # 質問の埋め込みベクトルを取得
    question_vector = vectorize_text(question)
    # 回答の埋め込みベクトルを取得
    vectors = [vectorize_text(doc) for doc in documents]

    # コサイン類似度が最も高い回答を取得
    max_similarity = 0
    most_similar_index = 0
    for index, vector in enumerate(vectors):
        similarity = cosine_similarity([question_vector], [vector])[0][0]
        print(f"コサイン類似度: {similarity.round(4)}:{documents[index]}")
        # 取り出したコサイン類似度が最大のものを保存
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_index = index
    print(f"\n質問: {question}\n回答: {documents[most_similar_index]}\n")

if __name__ == '__main__':
    # 質問文
    question1 = "株式会社ABCの開発を行う事業部は？"
    question2 = "株式会社ABCの運用を行う事業部は？"
    question3 = "売上を管理するシステムは？"
    rag_sample(question1, documents)
    rag_sample(question2, documents)
    rag_sample(question3, documents)