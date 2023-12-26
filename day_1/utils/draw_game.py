import gradio as gr
import random
import cv2
import numpy as np
import time
import torch
import onnxruntime
from utils.draw_game_labels import get_label_list


def softmax(x):
    x = x - np.max(x, axis=0)
    return np.exp(x) / np.sum(np.exp(x), axis=0)


class DrawGame:
    def __init__(self, model):
        self.labels = get_label_list()
        self.name = ''
        self.score_info_list = []
        self.score_history = ''
        self.end_time = time.time() + 3600
        self.correct_count = 0
        self.score = 0
        self.selected_label = -1
        self.question_num = 0
        self.model = model

    def get_score(self, user_info):
        return int(user_info.split('スコア: ')[-1])

    def start_game(self, js_msg, input_name='', reset=False):
        if input_name != '':
            self.name = input_name
        if self.name == '':
            ranking_info = [f'{i + 1}位  {info}' for i, info in enumerate(self.score_info_list)]
            return "ユーザー名を入力してください。", '\n'.join(ranking_info)
        if time.time() > self.end_time or reset:
            return self.end_game(reset)
        return self.next_question()

    def end_game(self, reset):
        user_info = f' ユーザー名: {self.name}  正解数: {self.correct_count}/{self.question_num}  スコア: {self.score}'
        self.reset_game()
        if reset:
            return 'リセットしました。', None
        self.score_info_list.append(user_info)
        self.score_info_list.sort(key=self.get_score, reverse=True)
        ranking_info = [f'{rank + 1}位  {info}' for rank, info in enumerate(self.score_info_list)]
        return f"ゲーム終了\n" + user_info, '\n'.join(ranking_info)

    def reset_game(self):
        self.question_num = 0
        self.correct_count = 0
        self.score = 0
        self.score_history = ''
        self.selected_label = -1
        self.end_time = time.time() + 3600

    def next_question(self):
        while True:
            label_choice = random.choice([i for i in range(len(self.labels))])
            if self.selected_label != label_choice:
                break
        self.question_num += 1
        self.selected_label = label_choice
        message = f"{self.question_num}問目: {self.labels[self.selected_label]}を描いてください。"
        if self.question_num == 1:
            self.end_time = time.time() + 180  # 3分間のゲーム時間
        ranking_info = [f'{i + 1}位  {info}' for i, info in enumerate(self.score_info_list)]
        return message, '\n'.join(ranking_info)

    def recognize_drawing(self, desp, result, draw_image):
        if self.selected_label == -1:
            return "ユーザー名を入力してゲーム開始ボタンを押してください", result, None, None, None
        draw_image = cv2.resize(draw_image, (28, 28))        
        draw_image = draw_image / 255.0
        draw_image = draw_image.reshape(1, 28, 28)
        draw_image = np.array(draw_image)
        draw_image = torch.from_numpy(draw_image.astype(np.float32)).clone()
        draw_image = np.expand_dims(np.array(draw_image), 0)
        predicted_prob = self.model.run(["logits"], {"pixel_values": draw_image})
        predicted_prob = softmax(predicted_prob[0][0])
        preds = np.argsort(predicted_prob)[::-1].tolist()
        predicted_label = preds[0]
        prob_dict = {self.labels[pred]: predicted_prob.tolist()[pred] for pred in preds}
        return self.evaluate_prediction(desp, predicted_label, prob_dict)

    def evaluate_prediction(self, desp, predicted_label, prob_dict):
        current_score = int(prob_dict[self.labels[self.selected_label]] * 100)
        self.score += current_score
        if int(predicted_label) == int(self.selected_label):
            self.correct_count += 1
            self.score_history += f"{self.question_num}問目: ○  スコア:{current_score}\n"
            return desp, f"{self.question_num:2d}問目: 正解！  スコア:{current_score}\n", prob_dict, None, self.score_history
        else:
            self.score_history += f"{self.question_num}問目: ×  スコア:{current_score}\n"
            return desp, f"{self.question_num:2d}問目: 不正解 正解は「{self.labels[self.selected_label]}」で認識したものは「{self.labels[predicted_label]}」でした。", prob_dict, None, self.score_history
