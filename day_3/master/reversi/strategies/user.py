#!/usr/bin/env python
"""
ユーザ入力
"""

import re
import time

import matplotlib.pyplot as plt
import numpy as np

from reversi.strategies.common import AbstractStrategy


plt.ioff()


class ConsoleUserInput(AbstractStrategy):
    """
    コンソールからのユーザ入力
    """
    def __init__(self):
        self.digit = re.compile(r'^[0-9]+$')
        self.alpha = re.compile('[a-zA-Z]')

    def next_move(self, color, board):
        """
        次の一手
        """
        legal_moves = board.get_legal_moves(color)
        select = None

        while True:
            user_in = input('>> ')

            if self._is_alpha(user_in):
                user_in_int = ord(user_in.lower()) - ord('a')
                select = user_in_int
                if 0 <= select < len(legal_moves):
                    break

        return legal_moves[select]

    def _is_digit(self, string):
        """
        半角数字の判定
        """
        return self.digit.match(string) is not None

    def _is_alpha(self, string):
        """
        アルファベットの判定
        """
        return bool(self.alpha.fullmatch(string))


class WindowUserInput(AbstractStrategy):
    """
    ウィンドウからのユーザ入力
    """
    def __init__(self, window):
        self.window = window

    def next_move(self, color, board):
        """
        次の一手
        """
        moves = board.get_legal_moves(color)
        self.window.board.selectable_moves(moves)

        while True:
            if self.window.menu.event.is_set():
                # キャンセル時は反則負け
                return (board.size//2-1, board.size//2-1)

            if self.window.board.event.is_set():
                move = self.window.board.move
                self.window.board.event.clear()

                if move in moves:
                    self.window.board.unselectable_moves(moves)
                    break

            time.sleep(0.01)

        return move


class MatplotlibUserInput(AbstractStrategy):
    """
    matplotlibのクリックを用いたユーザ入力
    """
    def __init__(self):
        self.fig = None
        self.x = -1
        self.y = -1
        self.cid = None

    def set_fig(self, fig):
        self.fig = fig

    def next_move(self, color, board):
        """
        次の一手
        """
        self.x = -1
        self.y = -1
        #print("aaa")
        legal_moves = board.get_legal_moves(color)
        self.fig.show()
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        if self.x >= 0 and self.y >= 0  and (self.x, self.y) in legal_moves:
            self.fig.canvas.mpl_disconnect(self.cid)
            return (self.x, self.y)
        else:
            self.next_move(color, board)
            #self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
    
    def onclick(self, event):
        self.x = int(event.xdata)
        self.y = int(event.ydata)
