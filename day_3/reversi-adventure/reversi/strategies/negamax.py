"""NegaMax
"""

import random

from reversi.strategies.common import Timer, Measure, AbstractStrategy


class _NegaMax_(AbstractStrategy):
    """
    NegaMax法で次の手を決める
    """
    def __init__(self, depth=3, evaluator=None):
        self._MIN = -10000000

        self.depth = depth
        self.evaluator = evaluator

    def next_move(self, color, board):
        """
        次の一手
        """
        pid = Timer.get_pid(self)  # タイムアウト監視用のプロセスID
        next_color = 'white' if color == 'black' else 'black'
        moves, max_score = {}, self._MIN

        # 打てる手の中から評価値の最も高い手を選ぶ
        legal_moves = board.get_legal_moves(color)
        for move in legal_moves:
            board.put_disc(color, *move)                                       # 一手打つ
            score = -self.get_score(next_color, board, self.depth-1, pid=pid)  # 評価値を取得
            board.undo()                                                       # 打った手を戻す

            if Timer.is_timeout(pid):       # タイムアウト発生時
                if max_score not in moves:  # 候補がない場合は現在の手を返す
                    return move
                break
            else:
                max_score = max(max_score, score)  # 最大値を選択
                if score not in moves:             # 次の候補を記憶
                    moves[score] = []
                moves[score].append(move)

        return random.choice(moves[max_score])  # 複数候補がある場合はランダムに選ぶ

    def get_score(self, color, board, depth, pid=None):
        """
        評価値の取得
        """
        # ゲーム終了 or 最大深さに到達
        legal_moves_b_bits = board.get_legal_moves_bits('black')
        legal_moves_w_bits = board.get_legal_moves_bits('white')
        is_game_end = True if not legal_moves_b_bits and not legal_moves_w_bits else False
        if is_game_end or depth <= 0:
            sign = 1 if color == 'black' else -1
            return self.evaluator.evaluate(color=color, board=board, possibility_b=board.get_bit_count(legal_moves_b_bits), possibility_w=board.get_bit_count(legal_moves_w_bits)) * sign  # noqa: E501

        # パスの場合
        legal_moves_bits = legal_moves_b_bits if color == 'black' else legal_moves_w_bits
        next_color = 'white' if color == 'black' else 'black'

        if not legal_moves_bits:
            return -self.get_score(next_color, board, depth, pid=pid)

        # 評価値を算出
        max_score = self._MIN
        size = board.size
        mask = 1 << ((size**2)-1)
        for y in range(size):
            skip = False
            for x in range(size):
                if legal_moves_bits & mask:
                    board.put_disc(color, x, y)
                    score = -self.get_score(next_color, board, depth-1, pid=pid)
                    board.undo()

                    if Timer.is_timeout(pid):
                        skip = True
                        break
                    else:
                        max_score = max(max_score, score)  # 最大値を選択

                mask >>= 1

            if skip:
                break

        return max_score


class _NegaMax(_NegaMax_):
    """NegaMax + Measure
    """
    @Measure.time
    def next_move(self, color, board):
        """next_move
        """
        return super().next_move(color, board)

    @Measure.countup
    def get_score(self, color, board, depth, pid=None):
        """get_score
        """
        return super().get_score(color, board, depth, pid=pid)


class NegaMax_(_NegaMax_):
    """NegaMax + Timer
    """
    @Timer.start(-10000000)
    def next_move(self, color, board):
        """next_move
        """
        return super().next_move(color, board)

    @Timer.timeout
    def get_score(self, color, board, depth, pid=None):
        """get_score
        """
        return super().get_score(color, board, depth, pid=pid)


class NegaMax(_NegaMax_):
    """NegaMax + Measure + Timer
    """
    @Timer.start(-10000000)
    @Measure.time
    def next_move(self, color, board):
        """next_move
        """
        return super().next_move(color, board)

    @Timer.timeout
    @Measure.countup
    def get_score(self, color, board, depth, pid=None):
        """get_score
        """
        return super().get_score(color, board, depth, pid=pid)
