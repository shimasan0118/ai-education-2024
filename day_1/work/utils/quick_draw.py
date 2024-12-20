import japanize_matplotlib
import matplotlib.pyplot as plt
import random
import numpy as np
import torch
import onnxruntime
from datasets import load_dataset
from collections import Counter
from utils.draw_game_labels import LABELS, EN_TO_JA_LABELS

MODEL_IMG_PATH = 'speed_sketch_dash/models/model.png'


class QuickDrawModel:
    def __init__(self):
        self.dataset = load_dataset('Xenova/quickdraw-small')
        self.labels = LABELS
        self.en_labels = EN_TO_JA_LABELS
        
    def show_dataset_num(self):
        print("学習用データ　　　: {}個".format(self.dataset['train'].num_rows))
        print("テスト用データ　　: {}個".format(self.dataset['test'].num_rows))
        print("ラベルのクラス数　: {}個".format(len(self.labels)))
        
    def load_pretrained_model(self, path):
        return onnxruntime.InferenceSession(path)
    
    def show_dataset_graph(self, dataset_type='train'):
        # データセットからラベルを取得
        labels = self.dataset[dataset_type]['label']
        # 各クラスの出現回数をカウント
        label_counts = Counter(labels)
        
        # 最少および最多のデータ数を持つクラスを特定
        min_label = min(label_counts, key=label_counts.get)
        max_label = max(label_counts, key=label_counts.get)

        min_count = label_counts[min_label]
        max_count = label_counts[max_label]

        # クラスのインデックスを作成
        class_indexes = range(len(label_counts))

        # グラフのサイズを設定
        plt.figure(figsize=(20, 5))

        # ヒストグラムの作成
        plt.bar(class_indexes, [label_counts[i] for i in class_indexes])
        plt.xlabel('クラス')
        plt.ylabel('データ数')
        plt.title('クラス毎のデータ数')
        plt.show()

        # 最少および最多のデータ数を持つクラスの情報を表示
        print(f"最少データ数のクラス: 【ラベル】 {self.labels[str(min_label)]}, 【総数】 {min_count}")
        print(f"最大データ数のクラス: 【ラベル】 {self.labels[str(max_label)]}, 【総数】 {max_count}")  

    # データセットから画像とラベルを表示するメソッド
    def show_random_images_and_labels(self, n, dataset_type='train', class_label=None):
        plt.figure(figsize=(12, 4))
        selected_indices = []

        if class_label is not None:
            if class_label not in self.labels.values():
                print(f"指定されたクラスラベル '{class_label}' は存在しません。")
                return
            for i, example in enumerate(self.dataset[dataset_type]):
                if str(example['label']) in self.labels.keys():
                    if self.labels[str(example['label'])] == class_label:
                        found = True
                        selected_indices.append(i)
                        if len(selected_indices) == n:  # n枚の画像が見つかったら停止
                            break

            for i, idx in enumerate(selected_indices):
                example = self.dataset[dataset_type][idx]
                image = example['image'].resize((128, 128))
                label = class_label

                # サブプロットに画像とラベルを表示
                plt.subplot(1, n, i+1)
                plt.imshow(image, cmap='gray')
                plt.title(label)
                plt.axis('off')  
        else:
            # ランダムにデータを選択
            selected_label_keys = random.sample(LABELS.keys(), n)

            for i, label_key in enumerate(selected_label_keys):
                selected_idx = self.dataset[dataset_type]['label'].index(int(label_key))
                image = self.dataset[dataset_type][selected_idx]['image']
                image = image.resize((128, 128))
                label = self.labels[label_key]

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
        random_indices = np.random.choice(len(self.dataset['test']), n + 30, replace=False)

        random_indices = [
            idx
            for idx in random_indices if str(self.dataset['test'][int(idx)]['label']) in self.labels.keys()
        ][:n]

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
            results = model.run(["logits"], {"pixel_values": test_image_norm})[0][0]

            # ラベルにないものは0で置き換える
            results = [
                res if i in [int(l) for l in LABELS.keys()] else 0
            for i, res in enumerate(results)
            ]

            predicted_label = self.dataset['test'].features['label'].names[np.argmax(results)]
            if predicted_label not in self.en_labels.keys():
                print('pred: ', predicted_label)
                continue
            predicted_label = self.en_labels[predicted_label]

            # 実際のラベル
            actual_label = self.dataset['test'].features['label'].names[self.dataset['test'][idx]['label']]
            if actual_label not in self.en_labels.keys():
                print('acc: ',actual_label)
                continue
            actual_label = self.en_labels[actual_label]

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