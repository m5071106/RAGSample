from flask import Flask, request, jsonify, render_template
import pdf_rag_sample

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html',message="")

@app.route('/ask', methods=['POST'])
def ask():
    question = request.form['question']
    if not question:
        return jsonify({'error': '質問を入力してください'}), 400

    # pdfフォルダにあるpdfファイルを読み込み、テキストを取得
    pdf_info = pdf_rag_sample.get_pdf_info()
    # テキストを学習データとしてベクトル化
    text_chunks = pdf_rag_sample.chunk_text(pdf_info)
    # 回答ベクトル
    document_vectors = [pdf_rag_sample.vectorize_text(doc) for doc in text_chunks]
    # 質問ベクトル
    question_vector = pdf_rag_sample.vectorize_text(question)
    # 最も質問に対する回答の意図に近い文書を取得
    selected_chunk = pdf_rag_sample.find_most_similar(question_vector, document_vectors, text_chunks)
    # ChatGPTから回答を取得
    answer = pdf_rag_sample.ask_question(question, selected_chunk)
    
    return render_template('answer.html', question=question, answer=answer)

@app.route('/upload', methods=['POST'])
def upload():
    pdf_file = request.files['file']
    if not pdf_file:
        return jsonify({'error': 'ファイルがアップロードされていません'}), 400

    # アップロードされたPDFファイルを保存
    pdf_file.save('pdf/sample.pdf')
    return render_template('index.html',message="PDFファイルをアップロードしました")

if __name__ == '__main__':
    app.run(debug=True,port=5001)
