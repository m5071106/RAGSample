import tkinter as tk
from tkinter import messagebox
import pdf_rag_sample

class SampleApp(tk.Tk):
    # コンストラクタ
    def __init__(self):
        super().__init__()

        self.title("Sample GUI")
        self.geometry("800x800")

        # ラベルを追加
        self.label = tk.Label(self, text="質問を入力してください", font=("メイリオ", 16))
        self.label.pack()

        self.blank_label = tk.Label(self, text="", font=("メイリオ", 16))
        self.blank_label.pack()

        # 質問を入力するエントリを追加
        self.entry = tk.Entry(self, font=("メイリオ", 16))
        self.entry.config(width=400)
        self.entry.pack()

        self.blank_label = tk.Label(self, text="", font=("メイリオ", 16))
        self.blank_label.pack()

        # 質問ボタンを追加
        self.button = tk.Button(self, text="チャットに質問", command=self.on_button_click, font=("メイリオ", 16))
        self.button.pack()

        # プログラムを終了するボタンを追加
        self.quit_button = tk.Button(self, text="終了", command=self.quit, font=("メイリオ", 16))
        self.quit_button.pack()

        # 回答を表示するラベルを追加
        self.answer_label = tk.Label(self, text="", font=("メイリオ", 12), bg="lightgray")
        self.answer_label.config(wraplength=750)
        self.answer_label.config(anchor="w", justify="left")
        self.answer_label.config(height=400)
        self.answer_label.pack()

        # 起動位置を左上に調整
        self.update_idletasks()
        self.withdraw()
        self.deiconify()
        self.attributes("-topmost", True)
        self.update()
        x = 0
        y = 0
        self.geometry("+%d+%d" % (x, y))


    # 質問ボタンがクリックされたときの処理
    def on_button_click(self):
        question = self.entry.get()

        if question == "":
            messagebox.showerror("未入力", "質問を入力してください")
            return
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
        # GUIに回答を表示
        answer = "質問: " + question + "\n" + "回答: \n" + answer
        self.answer_label.config(text=answer)

# プログラムを実行
if __name__ == "__main__":
    app = SampleApp()
    app.mainloop()
