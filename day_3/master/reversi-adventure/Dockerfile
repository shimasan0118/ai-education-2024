# Python 3.10を含むベースイメージを使用
FROM python:3.10

# アプリケーションディレクトリを作成
WORKDIR /app

# 依存関係をコピー
COPY requirements.txt ./

# 依存関係をインストール
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションのソースコードをコピー
COPY . .

# サービスを開始するコマンド
CMD ["gunicorn", "-b", ":8080", "server:app"]