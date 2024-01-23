"""Display
"""

import time
import abc
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output

plt.ioff()

PLAYER_COLORS = ('black', 'white')


class AbstractDisplay(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def progress(self, board, black_player, white_player, **kwargs):
        pass

    @abc.abstractmethod
    def turn(self, player, legal_moves):
        pass

    @abc.abstractmethod
    def move(self, player, legal_moves):
        pass

    @abc.abstractmethod
    def foul(self, player):
        pass

    @abc.abstractmethod
    def win(self, player, **kwargs):
        pass

    @abc.abstractmethod
    def draw(self):
        pass


class ConsoleDisplay(AbstractDisplay):
    """Console Display"""
    def __init__(self, sleep_time_turn=1, sleep_time_move=1):
        self.sleep_time_turn = sleep_time_turn  # sec
        self.sleep_time_move = sleep_time_move  # sec
        self.bg = '\033[48;2;46;139;87m'
        self.default = '\x1b[0m'
        self.score = ''
        self.disp = ''

    def progress(self, board, black_player, white_player, player=None):
        """display progress"""
        self._setup_screen()
        
        score_b = str(black_player) + ':' + str(board._black_score)
        score_w = str(white_player) + ':' + str(board._white_score)

        score = ' ' + self.bg + ' ' + score_b + '  ' + score_w + ' ' + self.default
        self.score = score
        print(score)
        if player is not None:
            color = player.color
        else:
            color = ''
        self._show_board(board, color)

        self._teardown_screen()

    def turn(self, player, legal_moves):
        """display turn"""
        time.sleep(self.sleep_time_turn)
        print(self.bg + ' ' + str(player) + "のターン " + self.default)

        for index, value in enumerate(legal_moves):
            coordinate = (chr(value[0] + 97), str(value[1] + 1))
            index = chr(ord('a') + index).upper()
            print(f'{index:2}:', coordinate)

    def move(self, player, legal_moves):
        """display move"""
        x = chr(player.move[0] + 97)
        y = str(player.move[1] + 1)

        print('置いた場所:', (x, y))
        print()
        time.sleep(self.sleep_time_move)

    def foul(self, player):
        """display foul player"""
        print(player, 'foul')

    def win(self, player, black_num=0, white_num=0):
        """display win player"""
        if black_num != 0 and white_num !=0:
            if black_num >= white_num:
                print('{}石差で{}の勝利です！'.format(black_num - white_num, player))
            else:
                print('{}石差で{}の勝利です！'.format(white_num - black_num, player))
        else:
            print(player, 'の勝利です！')

    def draw(self):
        """display draw"""
        print('引き分けです')

    def recommend(self, board, player, rec_player_name, move, legal_moves):
        idx = -1
        x = chr(move[0] + 97)
        y = str(move[1] + 1)
        legal_move_idx_dict = {k: v for v, k in enumerate(legal_moves)}
        if move in legal_move_idx_dict:
            idx = legal_move_idx_dict[move]
            idx = chr(ord('a') + idx).upper()
        self._show_rec_board(board, player, legal_moves, idx)
        print("----------------------------")
        print('{} >> {}: '.format(rec_player_name, idx), (x,y))
        print("----------------------------")

    def _setup_screen(self):
        clear_output(wait=True)
        time.sleep(0.1)
        # cursor-hyde, cursor-move-12row, erase-upto-end, cursor-move-top
        print("\033[?25l\033[12H\033[J\033[;H", end='')

    def _teardown_screen(self):
        # cursor-show
        print("\033[?25h", end='')

    def _show_board(self, board, color):
        board.color = color
        disp = str(board)
        default = '\x1b[0m'
        fg_w = '\033[38;2;255;255;255m'
        fg_b = '\033[38;2;0;0;0m'
        fg_r = '\033[38;2;255;0;0m'
        disp = disp.replace(fg_w + ' ●', self.bg + fg_w + ' ●' + default)
        disp = disp.replace(fg_b + ' ●', self.bg + fg_b + ' ●' + default)
        disp = disp.replace(' □', self.bg + ' □' + default)
        self.disp = disp
        print(disp)

    def _show_rec_board(self, board, player, legal_moves, rec_char):
        disp = self.disp
        default = '\x1b[0m'
        fg_r = '\033[38;2;255;0;0m'
        disp = disp.replace(' ' + rec_char.upper(), self.bg + fg_r + ' ' + rec_char + default)
        time.sleep(0.2)
        clear_output(wait=True)
        time.sleep(0.2)
        print(self.score)
        print(disp)
        self.turn(player, legal_moves)

class NoneDisplay(AbstractDisplay):
    """None Display"""
    def progress(self, board, black_player, white_player, player):
        pass

    def turn(self, player, legal_moves):
        pass

    def move(self, player, legal_moves):
        pass

    def foul(self, player):
        pass

    def win(self, player, **kwargs):
        pass

    def draw(self):
        pass


class WindowDisplay(AbstractDisplay):
    """GUI Window Display"""
    def __init__(self, window, sleep_time_turn=0.3, sleep_time_move=0.3):
        self.info = window.info
        self.board = window.board
        self.sleep_time_turn = sleep_time_turn  # sec
        self.sleep_time_move = sleep_time_move  # sec
        self.pre_move = None

    def progress(self, board, black_player, white_player, player=None):
        """display progress"""
        self.info.set_text('black', 'score', str(board._black_score))
        self.info.set_text('white', 'score', str(board._white_score))

    def turn(self, player, legal_moves):
        """display turn"""
        self.info.set_turn_text_on(player.color)  # 手番の表示
        self.board.enable_moves(legal_moves)      # 打てる候補を表示
        time.sleep(self.sleep_time_turn)

    def move(self, player, legal_moves):
        """display move"""
        x = chr(player.move[0] + 97)
        y = str(player.move[1] + 1)

        for color in PLAYER_COLORS:
            self.info.set_turn_text_off(color)  # 手番の表示を消す
            self.info.set_move_text_off(color)  # 打った手の表示を消す

        self.board.disable_moves(legal_moves)                # 打てる候補のハイライトをなくす
        if self.pre_move:
            self.board.disable_move(*self.pre_move)          # 前回打ったてのハイライトを消す
        self.board.enable_move(*player.move)                 # 打った手をハイライト
        self.board.put_disc(player.color, *player.move)      # 石を置く
        time.sleep(self.sleep_time_move)
        self.info.set_move_text_on(player.color, x, y)       # 打った手を表示
        self.board.turn_disc(player.color, player.captures)  # 石をひっくり返すアニメーション
        self.pre_move = player.move

    def foul(self, player):
        """display foul player"""
        self.info.set_foul_text_on(player.color)

    def win(self, player):
        """display win player"""
        winner, loser = ('black', 'white') if player.color == 'black' else ('white', 'black')
        self.info.set_win_text_on(winner)
        self.info.set_lose_text_on(loser)

    def draw(self):
        """display draw"""
        for color in PLAYER_COLORS:
            self.info.set_draw_text_on(color)


class MatplotlibDisplay(AbstractDisplay):
    """Matplotlib Display"""
    def __init__(self, sleep_time_turn=1, sleep_time_move=1, board_size=8):
        self.sleep_time_turn = sleep_time_turn  # sec
        self.sleep_time_move = sleep_time_move  # sec
        self.init_board(board_size)

    def init_board(self, board_size):
        print('fuga')
        self.fig = plt.figure(figsize=(4, 6), dpi=100)
        self.fig.subplots_adjust(bottom=0.4)
        self.ax = self.fig.add_subplot(1,1,1)

        self.ax.set_xlim([0, board_size])
        self.ax.set_ylim([board_size, 0])
        self.ax.axes.set_xticks(np.arange(0, board_size+1))
        self.ax.axes.set_yticks(np.arange(0, board_size+1))
        self.ax.tick_params(length=0)
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.grid()
        self.ax.set_facecolor('#2e8b57')
        
        self.sc1 = self.ax.scatter([], [], c='black', marker='o', s=400)
        self.sc2 = self.ax.scatter([], [], c='white', marker='o', s=400)

    def progress(self, board, black_player, white_player, player=None):

        score_b = str(black_player) + ':' + str(board._black_score)
        score_w = str(white_player) + ':' + str(board._white_score)
        score = score_b + score_w

        self._show_board(board, score)

    def turn(self, player, legal_moves):
        """display turn"""
        time.sleep(self.sleep_time_turn)
        if hasattr(player.strategy, 'set_fig'):
            player.strategy.set_fig(self.fig)
        self.ax.text(4, 9, str(player) + "'s turn", ha='center')
    def move(self, player, legal_moves):
        """display move"""
        x = chr(player.move[0] + 97)
        y = str(player.move[1] + 1)

        print('putted on', (x, y))
        print()
        time.sleep(self.sleep_time_move)

    def foul(self, player):
        """display foul player"""
        print(player, 'foul')

    def win(self, player):
        """display win player"""
        print(player, 'win')

    def draw(self):
        """display draw"""
        print('draw')


    def _show_board(self, board, score):
        board_arr = np.array(board.get_board_info())
        x, y = np.where(board_arr == 1)
        self.sc1.set_offsets(np.array([x, y]).T + 0.5)
        x, y = np.where(board_arr == -1)
        self.sc2.set_offsets(np.array([x, y]).T + 0.5)
        self.ax.set_title(score)
