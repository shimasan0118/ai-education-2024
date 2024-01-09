import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import matplotlib.pyplot as plt
import japanize_matplotlib
import pandas as pd
import numpy as np
from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical, plot_model
import PIL.Image
from io import BytesIO
import base64
import time
from IPython.display import display, HTML, clear_output


layers_choices = ['Dense', 'Flatten']

# 各レイヤーに関する質問
questions = [
    "画像を縦横から横一列に変換するレイヤーは？",
    "特定の数のニューロン同士を全部繋げたレイヤーは？"
]

correct_answers = ['Flatten', 'Dense']


# 画像をBase64エンコードされた文字列に変換する関数
def image_to_base64(image_array):
    pil_image = PIL.Image.fromarray(image_array)
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def display_html(text, color='black'):
    display(HTML(f"<p style='color: {color};'>{text}</p>"))


def ask_question(question, choices, correct_answer):
    while True:
        clear_output(wait=True)
        display_html(question, color='blue')
        for i, choice in enumerate(choices):
            print(f"{i + 1}. {choice}")

        answer = input("選択してください（数字）: ")
        if choices[int(answer) - 1] == correct_answer:
            display_html("正解です！", color='green')
            time.sleep(1)
            break
        else:
            display_html("不正解です、もう一度考えてみましょう。", color='red')
            time.sleep(1) 

class JapaneseLogCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        train_loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        train_accuracy = logs.get('accuracy')
        val_accuracy = logs.get('val_accuracy')

        print("\n-------------------------------------------------------------------------------------------------")
        print(f"エポック {epoch + 1}: "
              f"学習時の損失 = {train_loss:.4f}, 学習時の正解率 = {train_accuracy * 100:.2f}%, "
              f"テスト時の損失 = {val_loss:.4f}, テスト時の正解率 = {val_accuracy * 100:.2f}%")
        print("-------------------------------------------------------------------------------------------------")
        
    def on_train_end(self, logs=None):
        bold = "\033[1m"
        red = "\033[31m"
        print("\n" + bold + red + "学習が完了しました！" + "\n")
        
class MnistModel:
    def __init__(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.mnist.load_data()

    def show_dataset_num(self):
        print("学習用データ　　　: {}個".format(self.x_train.shape[0]))
        print("学習用ラベル　　　: {}個".format(self.y_train.shape[0]))
        print("テスト用データ　　: {}個".format(self.x_test.shape[0]))
        print("テスト用ラベル　　: {}個".format(self.y_test.shape[0]))
        print("ラベルのクラス数　: {}個".format(np.max(self.y_train)+1))
    
    def show_dataset_graph(self):
        # 学習用データセットの各クラスのデータ数をカウント
        train_label_counts = [len(self.y_train[self.y_train == i]) for i in range(10)]
        # テスト用データセットの各クラスのデータ数をカウント
        test_label_counts = [len(self.y_test[self.y_test == i]) for i in range(10)]

        # 棒グラフで表示
        fig, ax = plt.subplots(1, 2, figsize=(15, 6))

        # 学習用データセットの棒グラフ
        ax[0].bar(range(10), train_label_counts, color='skyblue')
        ax[0].set_title('MNIST学習用データセット内の各クラスのデータ数')
        ax[0].set_xlabel('ラベル')
        ax[0].set_ylabel('データ数')
        ax[0].set_xticks(range(10))

        # 各棒の上部に値を表示（学習用データ）
        for i, count in enumerate(train_label_counts):
            ax[0].text(i, count, str(count), ha='center', va='bottom')

        # テスト用データセットの棒グラフ
        ax[1].bar(range(10), test_label_counts, color='lightgreen')
        ax[1].set_title('MNISTテスト用データセット内の各クラスのデータ数')
        ax[1].set_xlabel('ラベル')
        ax[1].set_ylabel('データ数')
        ax[1].set_xticks(range(10))

        # 各棒の上部に値を表示（テスト用データ）
        for i, count in enumerate(test_label_counts):
            ax[1].text(i, count, str(count), ha='center', va='bottom')

        plt.show()
        
    def display_mnist_samples(self, choose_label_class=-1, n=10):
        # 特定のラベルクラスのみを選択
        if choose_label_class in range(10):
            indices = [i for i, label in enumerate(self.y_train) if label == choose_label_class]
            sample_images = self.x_train[indices[:n]]
            sample_labels = self.y_train[indices[:n]]
        else:
            # ラベルクラスが指定されていない場合は最初のnサンプルを選択
            sample_images = self.x_train[:n]
            sample_labels = self.y_train[:n]

        # ラベルをワンホットエンコーディングに変換
        sample_labels_one_hot = to_categorical(sample_labels, num_classes=10)

        # データフレームを作成
        df = pd.DataFrame({
            '画像': [image_to_base64(img) for img in sample_images]
        })

        # ワンホットエンコーディングされたラベルを列に追加
        for i in range(10):
            df[f'ラベル_{i}'] = sample_labels_one_hot[:, i].astype(int)

        # DataFrameをHTMLとして表示する関数
        def display_df_with_images(df):
            format_dict = {'画像': lambda x: '<img src="data:image/png;base64,{}" style="width: 28px; height: 28px;">'.format(x)}
            return HTML(df.to_html(escape=False, formatters=format_dict))

        # DataFrameを表示
        display(display_df_with_images(df))
    

    def preprocess(self):
        # 0~255を0~1に正規化
        x_train_norm = self.x_train / 255.0
        x_test_norm = self.x_test / 255.0

        # 画像を(batch_size, rows, cols, channels)に拡張する。
        x_train_norm = tf.keras.backend.expand_dims(x_train_norm, axis=-1)
        x_test_norm = tf.keras.backend.expand_dims(x_test_norm, axis=-1)

        # ラベルを10クラスのワンホットベクトルに変更する
        y_train_one_hot  = tf.keras.utils.to_categorical(self.y_train, 10)
        y_test_one_hot   = tf.keras.utils.to_categorical(self.y_test, 10)

        return x_train_norm, y_train_one_hot, x_test_norm, y_test_one_hot
    
    def build_model(self):
        # モデルの構築
        self.model = tf.keras.models.Sequential()
        for i, question in enumerate(questions):
            ask_question(question, layers_choices, correct_answers[i])
            if correct_answers[i] == 'Flatten':
                self.model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
                self.model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
                self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
                # Flatten層の前にDropout層を追加
                self.model.add(tf.keras.layers.Dropout(0.25))
                self.model.add(tf.keras.layers.Flatten())
            elif correct_answers[i] == 'Dense':
                self.model.add(tf.keras.layers.Dense(128, activation='relu'))
                # Dropout層を追加
                self.model.add(tf.keras.layers.Dropout(0.5))
                # 2つ目のDense層も追加
                self.model.add(tf.keras.layers.Dense(10, activation='softmax'))

        clear_output(wait=True)
        display_html("モデルの構築が完了しました！", color='green')

        # モデルのプロット
        plot_model(self.model, to_file='model.png', show_shapes=True, show_layer_names=True)
        display(HTML("<img src='model.png'>"))

    def train(self, x_train, y_train, x_test, y_test):
        # 学習
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
        japanese_log_callback = JapaneseLogCallback()
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        history = self.model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, callbacks=[early_stopping_callback, japanese_log_callback])
        return history, self.model

    def show_train_graph(self, history):
        # 描画領域を設定
        plt.figure(1, figsize=(13,4))
        plt.subplots_adjust(wspace=0.5)

        # 学習曲線
        plt.subplot(1, 2, 1)
        plt.plot(history.history["loss"], label="学習")
        plt.plot(history.history["val_loss"], label="テスト")
        plt.title("学習とテストの損失")
        plt.xlabel("エポック")
        plt.ylabel("損失")
        plt.legend()
        plt.grid()

        # 精度表示
        plt.subplot(1, 2, 2)
        plt.plot([acc * 100 for acc in history.history["accuracy"]], label="学習")
        plt.plot([acc * 100 for acc in history.history["val_accuracy"]], label="テスト")
        plt.title("学習とテストの正解率")
        plt.xlabel("エポック")
        plt.ylabel("正解率 (%)")
        plt.legend()
        plt.grid()

        plt.show()
        
    def predict(self, n=10):
        random_indices = np.random.choice(len(self.x_test), n, replace=False)
        sample_images = self.x_test[random_indices]
        sample_labels = self.y_test[random_indices]

        # モデルによる予測
        predicted_labels = self.model.predict(sample_images)

        # 結果の表示
        fig, axes = plt.subplots(2, n, figsize=(15, 4))
        for i in range(n):
            # 画像を表示
            axes[0, i].imshow(sample_images[i].reshape(28, 28), cmap='gray')
            axes[0, i].axis('off')

            # 予測ラベルと実際のラベルを表示
            predicted_label = np.argmax(predicted_labels[i])
            actual_label = sample_labels[i]
            color = 'green' if predicted_label == actual_label else 'red'
            axes[1, i].text(0.5, 0.98, f"予測: {predicted_label}",
                            fontsize=14, ha='center', color=color)
            axes[1, i].text(0.5, 0.78, f"正解: {actual_label}",
                            fontsize=14, ha='center')
            axes[1, i].axis('off')