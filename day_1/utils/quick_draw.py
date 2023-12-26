import japanize_matplotlib
import matplotlib.pyplot as plt
import random
import numpy as np
import torch
import onnxruntime
from datasets import load_dataset
from utils.draw_game_labels import get_label_list, get_en_label_list

MODEL_IMG_PATH = 'speed_sketch_dash/models/model.png'


class QuickDrawModel:
    def __init__(self):
        self.dataset = load_dataset('Xenova/quickdraw-small')
        self.labels = get_label_list()
        self.en_labels = get_en_label_list()
    
    def show_dataset_num(self):
        print("学習用データ　　　: {}個".format(self.dataset['train'].num_rows))
        print("テスト用データ　　: {}個".format(self.dataset['test'].num_rows))
        print("ラベルのクラス数　: {}個".format(len(self.labels)))
        
    def load_pretrained_model(self, path):
        return onnxruntime.InferenceSession(path)
    
    # データセットから画像とラベルを表示するメソッド
    def show_random_images_and_labels(self, n):
        plt.figure(figsize=(12, 4))
        for i in range(n):
            # ランダムにデータを選択
            idx = random.randint(0, len(self.dataset['train']) - 1)
            example = self.dataset['train'][idx]
            image = example['image'].resize((128, 128))
            label = self.dataset['train'].features['label'].names[example['label']]
            label = self.labels[self.en_labels.index(label)]

            # サブプロットに画像とラベルを表示
            plt.subplot(1, n, i+1)
            plt.imshow(image, cmap='gray')
            plt.title(label)
            plt.axis('off')
        plt.show()
        
    def plot_model(self):
        # HTMLコンテンツを作成し、スクロール可能な領域に画像を表示します
        html_content = f"""
        <div style="width:1400px; height:600px; overflow:auto;">
            <img src="{MODEL_IMG_PATH}" style="width:100%; height:auto;">
        </div>
        </div>
        """
        return html_content
        
    
    def predict_and_show(self, model, n=10):
        random_indices = np.random.choice(len(self.dataset['test']), n, replace=False)
        fig, axes = plt.subplots(4, 5, figsize=(15, 8))  # 4行5列のグリッド

        for i, idx in enumerate(random_indices):
            idx = int(idx)  # idxを整数型に変換

            # テストデータの取得
            test_image = np.array(self.dataset['test'][idx]['image'])
            test_image = np.expand_dims(test_image, 0)
            test_image_norm = test_image / 255.0
            test_image_norm = torch.from_numpy(test_image_norm.astype(np.float32)).clone()
            test_image_norm = np.expand_dims(test_image_norm, 0)

            # モデルによる予測
            results = model.run(["logits"], {"pixel_values": test_image_norm})
            predicted_label = self.dataset['test'].features['label'].names[np.argmax(results[0])]
            predicted_label = self.labels[self.en_labels.index(predicted_label)]

            # 実際のラベル
            actual_label = self.dataset['test'].features['label'].names[self.dataset['test'][idx]['label']]
            actual_label = self.labels[self.en_labels.index(actual_label)]

            # 画像と予測結果の表示位置を計算
            row = i // 5
            col = i % 5

            # 画像を表示
            axes[row * 2, col].imshow(test_image.reshape(28, 28), cmap='gray')
            axes[row * 2, col].axis('off')

            # 予測ラベルと実際のラベルを表示
            color = 'green' if predicted_label == actual_label else 'red'
            axes[row * 2 + 1, col].text(0.5, 0.5, f"予測: {predicted_label}\n正解: {actual_label}",
                                        fontsize=14, ha='center', color=color)
            axes[row * 2 + 1, col].axis('off')

        plt.tight_layout()
        plt.show()