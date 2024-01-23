import sys
import os

from reversi import BitBoard, Player, Game
from reversi import C as c

import creversi
import numpy as np
from copy import deepcopy
from tensorflow import keras
import tensorflow as tf

# ボードの大きさ
hw = 8

# ボードのマス数
hw2 = 64

# インデックスの個数 縦横各8x2、斜め11x2
n_board_idx = 38

# ボードの1つのインデックスが取りうる値の種類。3^8
n_line = 6561

global_place = [
    [0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15], [16, 17, 18, 19, 20, 21, 22, 23],
    [24, 25, 26, 27, 28, 29, 30, 31], [32, 33, 34, 35, 36, 37, 38, 39], [40, 41, 42, 43, 44, 45, 46, 47],
    [48, 49, 50, 51, 52, 53, 54, 55], [56, 57, 58, 59, 60, 61, 62, 63],    
    [0, 8, 16, 24, 32, 40, 48, 56], [1, 9, 17, 25, 33, 41, 49, 57], [2, 10, 18, 26, 34, 42, 50, 58],
    [3, 11, 19, 27, 35, 43, 51, 59], [4, 12, 20, 28, 36, 44, 52, 60], [5, 13, 21, 29, 37, 45, 53, 61],
    [6, 14, 22, 30, 38, 46, 54, 62], [7, 15, 23, 31, 39, 47, 55, 63],
    [5, 14, 23, -1, -1, -1, -1, -1], [4, 13, 22, 31, -1, -1, -1, -1], [3, 12, 21, 30, 39, -1, -1, -1],
    [2, 11, 20, 29, 38, 47, -1, -1], [1, 10, 19, 28, 37, 46, 55, -1], [0, 9, 18, 27, 36, 45, 54, 63],
    [8, 17, 26, 35, 44, 53, 62, -1], [16, 25, 34, 43, 52, 61, -1, -1], [24, 33, 42, 51, 60, -1, -1, -1],
    [32, 41, 50, 59, -1, -1, -1, -1], [40, 49, 58, -1, -1, -1, -1, -1],
    [2, 9, 16, -1, -1, -1, -1, -1], [3, 10, 17, 24, -1, -1, -1, -1], [4, 11, 18, 25, 32, -1, -1, -1],
    [5, 12, 19, 26, 33, 40, -1, -1], [6, 13, 20, 27, 34, 41, 48, -1], [7, 14, 21, 28, 35, 42, 49, 56],
    [15, 22, 29, 36, 43, 50, 57, -1], [23, 30, 37, 44, 51, 58, -1, -1], [31, 38, 45, 52, 59, -1, -1, -1],
    [39, 46, 53, 60, -1, -1, -1, -1], [47, 54, 61, -1, -1, -1, -1, -1]
]

pop_digit = np.zeros((n_line, hw), dtype=int)

local_place = np.zeros((n_board_idx, hw2), dtype=int)

place_included = np.zeros((hw2, 4), dtype=int)

pow3 = [0] * 11

# 3の累乗
p31 = 3
p32 = 9
p33 = 27
p34 = 81
p35 = 243
p36 = 729
p37 = 2187
p38 = 6561
p39 = 19683
p310 = 59049

black = 0
white = 1

surround_arr = [[0 for _ in range(n_line)] for _ in range(2)]

diagonal8_idx = [[0, 9, 18, 27, 36, 45, 54, 63], [7, 14, 21, 28, 35, 42, 49, 56]]

for pattern in deepcopy(diagonal8_idx):
    diagonal8_idx.append(list(reversed(pattern)))

edge_2x_idx = [[9, 0, 1, 2, 3, 4, 5, 6, 7, 14], [9, 0, 8, 16, 24, 32, 40, 48, 56, 49], [49, 56, 57, 58, 59, 60, 61, 62, 63, 54], [54, 63, 55, 47, 39, 31, 23, 15, 7, 14]]
for pattern in deepcopy(edge_2x_idx):
    edge_2x_idx.append(list(reversed(pattern)))

triangle_idx = [
    [0, 1, 2, 3, 8, 9, 10, 16, 17, 24], [0, 8, 16, 24, 1, 9, 17, 2, 10, 3], 
    [7, 6, 5, 4, 15, 14, 13, 23, 22, 31], [7, 15, 23, 31, 6, 14, 22, 5, 13, 4], 
    [63, 62, 61, 60, 55, 54, 53, 47, 46, 39], [63, 55, 47, 39, 62, 54, 46, 61, 53, 60],
    [56, 57, 58, 59, 48, 49, 50, 40, 41, 32], [56, 48, 40, 32, 57, 49, 41, 58, 50, 59]
]

pattern_idx = [diagonal8_idx, edge_2x_idx, triangle_idx]
# ln_in = 21 => 4 + 8 + 8 +1

class Board:
    def __init__(self):
        self.board_idx = [0] * n_board_idx
        self.player = 0
        self.policy = 0
        self.value = 0
        self.n_stones = hw2
        
    # ボードのコンソールへの表示
    def print(self):
        hw = 8  # ボードの高さと幅を表す変数
        for i in range(hw):
            tmp = self.board_idx[i]
            #print(tmp)
            res = ""
            for j in range(hw):
                if tmp % 3 == 0:
                    res = "X " + res
                elif tmp % 3 == 1:
                    res = "O " + res
                else:
                    res = ". " + res
                tmp //= 3
            print(res)
        print()        

    def translate_to_arr(self):
        res = np.zeros((hw, hw), dtype=int)
        for i in range(hw):
            for j in range(hw):
                res[i][j] = pop_digit[self.board_idx[i]][j]
        return res

    def translate_from_arr(self, arr, player):
        for i in range(n_board_idx):
            self.board_idx[i] = n_line - 1
        self.n_stones = hw2
        for i in range(hw2):
            for j in range(4):
                if place_included[i][j] == -1:
                    continue
                if arr[i] == black:
                    self.board_idx[place_included[i][j]] -= 2 * pow3[hw - 1 - local_place[place_included[i][j]][i]]
                elif arr[i] == white:
                    self.board_idx[place_included[i][j]] -= pow3[hw - 1 - local_place[place_included[i][j]][i]]
            if arr[i] != black and arr[i] != white:
                self.n_stones -= 1
        self.player = player
        
    # ボードの初期化
    def board_init(self):
        pow3[0] = 1;
        for idx in range(1, 11):
            pow3[idx] = pow3[idx - 1] * 3

        for i in range(n_line):
            for j in range(hw):
                pop_digit[i][j] = (i // pow3[hw - 1 - j]) % 3

        for place in range(hw2):
            inc_idx = 0
            for idx in range(n_board_idx):
                for l_place in range(hw):
                    if global_place[idx][l_place] == place:
                        place_included[place][inc_idx] = idx
                        inc_idx += 1
            if inc_idx == 3:
                place_included[place][inc_idx] = -1

        for idx in range(n_board_idx):
            for place in range(hw2):
                for l_place in range(hw):
                    if global_place[idx][l_place] == place:
                        local_place[idx][place] = l_place

    def create_one_color(self, idx, k):
        res = 0
        for i in range(hw):
            if idx % 3 == k:
                res |= 1 << i
            idx //= 3
        return res

    def get_correct_1d_board_list(self, board_list):
        bd_list = []
        for i in range(len(board_list)):
            for j in range(len(board_list[i])):
                # 空きマス = 2
                if board_list[i][j] == 0:
                    bd_list.append(2)
                # 白石 = 1
                elif board_list[i][j] == -1:
                    bd_list.append(1)
                # 黒石 = 0
                elif board_list[i][j] == 1:
                    bd_list.append(0)
        return bd_list


class LearningAi:
    def mobility_line(self, player, opposite):
        """
        This function finds the legal moves for black stones in a bitboard representation.

        :param black: An 8-bit value representing black stones (1 for black stone, 0 for empty).
        :param white: An 8-bit value representing white stones (1 for white stone, 0 for empty).
        :return: An 8-bit value representing legal moves for black (1 for legal move, 0 for illegal).
        """
        legal_moves = 0
        empty = ~(player | opposite) & 0xFF  # Find empty positions

        # Check each empty position if it's a legal move
        for pos in range(8):
            if not (empty & (1 << pos)):
                continue  # Skip if the position is not empty

            # Check adjacent positions to the left and right
            for direction in [-1, 1]:
                adjacent = pos + direction

                # Continue if adjacent is out of bounds or not occupied by the opposite color
                if adjacent < 0 or adjacent >= 8 or not (opposite & (1 << adjacent)):
                    continue

                # Move in the direction and check if it leads to a player's stone
                while 0 <= adjacent < 8 and (opposite & (1 << adjacent)):
                    adjacent += direction

                # If the line ends at a black stone, mark the original position as a legal move
                if 0 <= adjacent < 8 and (player & (1 << adjacent)):
                    legal_moves |= (1 << pos)
                    break  # No need to check other directions for this position

        return legal_moves

    def count_bits(self, n):
        count = 0
        while n:
            count += n & 1  # nの最下位ビットが1ならカウントを増やす
            n >>= 1         # nを右に1ビットシフトして次のビットを調べる
        return count

    def evaluate_init(self, bd):
        global mobility_arr, surround_arr
        for idx in range(n_line):
            b = bd.create_one_color(idx, 0)
            w = bd.create_one_color(idx, 1)
            surround_arr[black][idx] = 0
            surround_arr[white][idx] = 0   

            for place in range(hw):
                if place > 0:
                    if (b >> (place - 1)) & 1 == 0 and (w >> (place - 1)) & 1 == 0:
                        if (b >> place) & 1:
                            surround_arr[black][idx] += 1
                        elif (w >> place) & 1:
                            surround_arr[white][idx] += 1
                if place < hw - 1:
                    if (b >> (place + 1)) & 1 == 0 and (w >> (place + 1)) & 1 == 0:
                        if (b >> place) & 1:
                            surround_arr[black][idx] += 1
                        elif (w >> place) & 1:
                            surround_arr[white][idx] += 1

    def get_creversi_board_str_predict(self, board_str):
        bd_str = ''
        for v in board_str:
            if v == '*':
                bd_str = bd_str + 'X'
            elif v == 'O':
                bd_str = bd_str + 'O'
            else:
                bd_str = bd_str + '-'
        return bd_str

    def calc_canput_v2(self, player, board_str):
        bd_str = self.get_creversi_board_str_predict(board_str)
        player = creversi.WHITE_TURN if player == white else creversi.BLACK_TURN
        cr_board = creversi.Board(bd_str, player)
        puttable_num = -cr_board.puttable_num() if player == white else cr_board.puttable_num()
        return puttable_num

    # 囲み具合
    def sfill5(self, b):
        return b - p35 + 1 if pop_digit[b][2] != 2 else b

    def sfill4(self, b):
        return b - p34 + 1 if pop_digit[b][3] != 2 else b

    def sfill3(self, b):
        return b - p33 + 1 if pop_digit[b][4] != 2 else b

    def sfill2(self, b):
        return b - p32 + 1 if pop_digit[b][5] != 2 else b

    def sfill1(self, b):
        return b - p31 + 1 if pop_digit[b][6] != 2 else b

    # 囲み度合い
    def calc_surround(self, b, p):
        return (
            sum(surround_arr[p][b.board_idx[i]] for i in range(16)) +
            sum(surround_arr[p][self.sfill5(b.board_idx[i])] for i in [16, 26, 27, 37]) +
            sum(surround_arr[p][self.sfill4(b.board_idx[i])] for i in [17, 25, 28, 36]) +
            sum(surround_arr[p][self.sfill3(b.board_idx[i])] for i in [18, 24, 29, 35]) +
            sum(surround_arr[p][self.sfill2(b.board_idx[i])] for i in [19, 23, 30, 34]) +
            sum(surround_arr[p][self.sfill1(b.board_idx[i])] for i in [20, 22, 31, 33]) +
            surround_arr[p][b.board_idx[21]] + 
            surround_arr[p][b.board_idx[32]]
        )


    def make_lines(self, board, patterns, player) -> list[list[float]]:
        res = []
        for pattern in patterns:
            tmp = []
            # patternの要素分
            for elem in pattern:
                tmp.append(1.0 if board[elem] == str(player) else 0.0)
            for elem in pattern:
                tmp.append(1.0 if board[elem] == str(1 - player) else 0.0)
            res.append(tmp)
        return res


    def create_model_features(self, board, player, v1, v2, v3):
        features = []
        v1 = float(v1)
        v2 = float(v2)
        v3 = float(v3)
        player = int(player)
        # len(pattern_idx) = 3
        for i in range(len(pattern_idx)):
            lines = self.make_lines(board, pattern_idx[i], 0)
            # そのパターンの要素数 8 + 8 + 4
            for line in lines:
                features.append(np.array([line]))
        features.append(np.array([[v1 / 30, (v2 - 15) / 15, (v3 - 15) / 15]]))
        return features

    def predict_score(self, board_list, board_str, player, lmodel):
        bd = Board()
        board_data = []
        bd.translate_from_arr(bd.get_correct_1d_board_list(board_list), player)
        a = self.calc_canput_v2(bd, board_str)
        b = self.calc_surround(bd, black)
        c = self.calc_surround(bd, white)
        board_data.append([
            board_str, 
            player,
            a,
            b,
            c
        ])
        features = self.create_model_features(*board_data[0])
        return lmodel.predict_single(features)[0][0]


class LiteModel:
    
    @classmethod
    def from_file(cls, model_path):
        return LiteModel(tf.lite.Interpreter(model_path=model_path))
    
    @classmethod
    def from_keras_model(cls, kmodel):
        converter = tf.lite.TFLiteConverter.from_keras_model(kmodel)
        tflite_model = converter.convert()
        return LiteModel(tf.lite.Interpreter(model_content=tflite_model), kmodel)
    
    def __init__(self, interpreter, kmodel):
        self.interpreter = interpreter
        self.interpreter.allocate_tensors()
        self.input_detail_list = []
        self.output_detail_list = []
        for detail in self.interpreter.get_input_details():
            self.input_detail_list.append(detail)
        for detail in self.interpreter.get_output_details():
            self.output_detail_list.append(detail)      
        self.input_index = [input_det["index"] for input_det in self.input_detail_list]
        self.output_index = [output_det["index"] for output_det in self.output_detail_list]
        self.input_shape = [input_det["shape"] for input_det in self.input_detail_list]
        self.output_shape = [output_det["shape"] for output_det in self.output_detail_list]
        self.input_dtype = [input_det["dtype"] for input_det in self.input_detail_list]
        self.output_dtype = [output_det["dtype"] for output_det in self.output_detail_list]
        self.input_names = [
            input_det["name"].replace("serving_default_", "").replace(":0", "") for input_det in self.input_detail_list
        ]
        self.input_keras_names = [det['name'] for det in kmodel.get_config()['layers'] if det['class_name'] == 'InputLayer']
        # tfliteにした時にmulti inputだと順番が狂うので、idxを修正するためのlistを作成しておく
        self.input_keras_idx = [self.input_keras_names.index(element) for element in self.input_names]
        
    def predict(self, inp):
        inp = inp.astype(self.input_dtype)
        count = inp.shape[0]
        out = np.zeros((count, self.output_shape[1]), dtype=self.output_dtype)
        for i in range(count):
            self.interpreter.set_tensor(self.input_index, inp[i:i+1])
            self.interpreter.invoke()
            out[i] = self.interpreter.get_tensor(self.output_index)[0]
        return out
    
    def predict_single(self, inp):
        """ Like predict(), but only for a single record. The input data can be a Python list. """
        #inp = np.array([inp], dtype=self.input_dtype)
        for i in range(len(self.input_names)):
            input_value = inp[self.input_keras_idx[i]].astype(self.input_dtype[i])
            self.interpreter.set_tensor(self.input_index[i], input_value)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_index[0])
        return out

'''
model = keras.models.load_model('model/model.h5')
lmodel = LiteModel.from_keras_model(model)

bd = Board()
ai = LearningAi()
bd.board_init()
ai.evaluate_init(bd)

from reversi.strategies.common import AbstractEvaluator
from reversi.strategies import _NegaScout

class MyEvaluator(AbstractEvaluator):
    def evaluate(self, color, board, possibility_b, possibility_w):
        player = 0 if color == 'black' else 1
        board_list = board.get_board_info()
        board_str = board.get_board_line_info(player='black', black='0', white='1', empty='.')
        score = ai.predict_score(board_list, board_str, player, lmodel)
        return score


from reversi import Reversic, strategies

Reversic(
    {
        'X': strategies.Random(),
        'M-10': strategies.MonteCarlo(count=10),
        'M-100': strategies.MonteCarlo(count=100),
        'M-1000': strategies.MonteCarlo(count=1000),
        'M-5000': strategies.MonteCarlo(count=5000),
        'TheEnd': strategies.MonteCarlo_EndGame(count=10000, end=14),        
        'AI6': _NegaScout(depth=6, evaluator=MyEvaluator()),
    },
).start()
'''
