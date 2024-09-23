# pip3 install openai
from openai import OpenAI
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
import requests
from bs4 import BeautifulSoup

# OpenAI APIキーの設定
os.environ["OPENAI_API_KEY"] = "sk-proj-JCOqpomOERRia7xRaYMDT3BlbkFJQ3ZkgGpKrwI7kgAjk8iE"

# OpenAIクライアントの初期化
client = OpenAI()

# URLから記事情報を抽出
def _scrape_article(url):
    # ページのHTMLを取得
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    # divタグデータを抽出
    text_nodes = soup.find_all("div")
    # テキスト要素のみ抽出
    t_all = []
    for t in text_nodes:
        t_all.append(t.text.replace("\t", "").replace("\n", ""))
    # 一つの文字列に結合
    joined_text = "".join(t_all)
    return joined_text


# 学習データとするチャンクの作成
def chunk_text(text, chunk_size, overlap):
    chunks = []
    start = 0
    while start + chunk_size <= len(text):
        chunks.append(text[start:start + chunk_size])
        start += (chunk_size - overlap)
    if start < len(text):
        chunks.append(text[-chunk_size:])
    return chunks


def vectorize_text(text):
    response = client.embeddings.create(
        input = text,
        # モデルの指定 (様々なモデルが利用可能)
        model = "text-embedding-3-small"
    )
    return response.data[0].embedding


# 最も類似した文書を見つける
def find_most_similar(question_vector, vectors, documents):
    max_similarity = 0
    most_similar_index = 0

    for index, vector in enumerate(vectors):
        similarity = cosine_similarity([question_vector], [vector])[0][0]
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_index = index
    
    return documents[most_similar_index]


# GPT-3に質問を投げる
def ask_question(question, context):
    prompt = f'''以下の質問に対する回答を教えてください。
    質問: {question}
    文脈: {context}
    回答:'''

    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=100
    )
    return response.choices[0].text

if __name__ == "__main__":
    url = "https://news.yahoo.co.jp/"
    chunk_size = 400
    overlap = 50

    article_text = _scrape_article(url)
    text_chunks = chunk_text(article_text, chunk_size, overlap)

    vectors = [vectorize_text(doc) for doc in text_chunks]

    question = "天気のニュースはありますか"

    question_vector = vectorize_text(question)

    most_similar_chunk = find_most_similar(question_vector, vectors, text_chunks)

    answer = ask_question(question, most_similar_chunk)
    print(answer)
