# pip3 install openai
from openai import OpenAI
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
import requests
from bs4 import BeautifulSoup

CHUNK_SIZE = 400
OVERLAP = 50

# OpenAI APIキーの設定
os.environ["OPENAI_API_KEY"] = "xxx"

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
def chunk_text(text):
    chunks = []
    start = 0
    while start + CHUNK_SIZE <= len(text):
        chunks.append(text[start:start + CHUNK_SIZE])
        start += (CHUNK_SIZE - OVERLAP)
    if start < len(text):
        chunks.append(text[-CHUNK_SIZE:])
    return chunks


def vectorize_text(text):
    response = client.embeddings.create(
        input = text,
        # モデルの指定 (様々なモデルが利用可能)
        model = "text-embedding-3-small"
    )
    return response.data[0].embedding


# 最も類似した文書を見つける
def find_most_similar(question_vector, document_vectors, text_chunks):
    max_similarity = 0
    most_similar_index = 0

    for index, vector in enumerate(document_vectors):
        similarity = cosine_similarity([question_vector], [vector])[0][0]
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_index = index
    
    return text_chunks[most_similar_index]


# GPT-3に質問を投げる
def ask_question(question, context):
    prompt = f'''以下の質問に対する回答を教えてください。
    質問: {question}
    文脈: {context}
    '''

    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=100
    )
    return response.choices[0].text

if __name__ == "__main__":
    # 抽出対象URL
    url = "https://docs.oracle.com/javase/jp/11/docs/api/index.html"
    print("指定したURL: " + url)

    # 抽出したいテキスト群に加工し, ベクトル化
    article_text = _scrape_article(url)
    text_chunks = chunk_text(article_text)
    document_vectors = [vectorize_text(doc) for doc in text_chunks]

    # 質問文
    question = "画像の読込に使える関数はどれですか？"
    print("質問: " + question)
    question_vector = vectorize_text(question)

    # 最も質問に対する回答の意図に近い文書を取得
    selected_chunk = find_most_similar(question_vector, document_vectors, text_chunks)
    print("もっともらしい文面: " + selected_chunk)

    # 取得したchunkをもとに質問をなげ, GPTから回答を取得する
    answer = ask_question(question, selected_chunk)
    print("回答: " + answer)
