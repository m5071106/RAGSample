# pip3 install openai
# pip3 install pdfminer.six
# pip3 install scikit-learn
from openai import OpenAI
import os
from sklearn.metrics.pairwise import cosine_similarity
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.layout import LAParams
from io import StringIO

# もっともらしい回答が含まれる文脈のサイズ
CHUNK_SIZE = 400
# 区切り部分を補完し学習するためのサイズ
OVERLAP = 50

# OpenAI APIキーの設定
os.environ["OPENAI_API_KEY"] = "xxx"

# OpenAIクライアントの初期化
client = OpenAI()

# PDF情報をテキストに変換する
def pdf2text(pdf_path):
    with open(pdf_path, "rb") as f:
        resource_manager = PDFResourceManager()
        output = StringIO()
        converter = TextConverter(resource_manager, output, laparams=LAParams())
        interpreter = PDFPageInterpreter(resource_manager, converter)
        for page in PDFPage.get_pages(f):
            interpreter.process_page(page)
        text = output.getvalue()
    return text


# PDFフォルダの内容をすべて取得し、記事情報として返す
def get_pdf_info():
    # pdfフォルダ内のファイルを取得
    pdf_files = os.listdir("pdf")
    pdf_text = ""
    for pdf_file in pdf_files:
        pdf_text += pdf2text("pdf/" + pdf_file)

    joined_text = "".join(pdf_text)
    # 空白を除去
    joined_text = joined_text.replace("\t", "").replace("\n", "").replace("\r", "").replace(" ", "").replace("　", "")
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


# テキストをベクトル化する
def vectorize_text(text):
    response = client.embeddings.create(
        input = text,
        # モデルの指定 (様々なモデルが利用可能)
        model = "text-embedding-3-small"
    )
    return response.data[0].embedding


# 質問内容に最も類似した文書を見つける
def find_most_similar(question_vector, document_vectors, text_chunks):
    max_similarity = 0
    most_similar_index = 0

    for index, vector in enumerate(document_vectors):
        similarity = cosine_similarity([question_vector], [vector])[0][0]
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_index = index
    
    return text_chunks[most_similar_index]


# GPT-3.5に質問を投げて回答を取得する
def ask_question(question, context):
    prompt = f'''以下の質問に対する回答を教えてください。文脈が途切れるときは文章を補完してください。
    質問: {question}
    文脈: {context}
    ちなみにこの内容は学習しないでください。
    '''
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=1024
    )
    returnWord = ""
    for choice in response.choices:
        returnWord += choice.text
    return returnWord

# CUIで実行する場合
if __name__ == "__main__":
    # 検索対象のテキスト
    pdf_info = get_pdf_info()

    # 抽出したテキストを学習データとしてベクトル化
    text_chunks = chunk_text(pdf_info)
    document_vectors = [vectorize_text(doc) for doc in text_chunks]

    # 質問文
    question = "東京へ日帰り出張したときに必要な申請は何ですか？"
    print("質問: \n" + question)
    question_vector = vectorize_text(question)

    # 最も質問に対する回答の意図に近い文書を取得
    selected_chunk = find_most_similar(question_vector, document_vectors, text_chunks)
    print("文脈: \n" + selected_chunk)

    # 取得したchunkをもとに質問をなげ, GPTから回答を取得する
    answer = ask_question(question, selected_chunk)
    print("回答: \n" + answer)
