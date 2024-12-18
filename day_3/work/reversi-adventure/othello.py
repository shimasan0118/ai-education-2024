# reversiをインポートします
import reversi
from utils import utils
import copy


# オセロゲームの状態を管理するクラス（サンプル）
class OthelloGame:
    def __init__(
        self, 
        black_cpu_dict={}, 
        white_cpu_dict={},
        lmodel=None,
        evaluator=None
    ):
        self.board = self.initialize_board()
        self.current_player = 'black'
        self.board_str = self.get_board_str()
        self.lmodel = lmodel
        self.ai = utils.LearningAi() # AIを定義します
        self.bd = utils.Board() # AI側が認識するボードを定義します
        self.bd.board_init() # AI側が認識するボード側の準備をします
        self.ai.evaluate_init(self.bd) # AI側の準備をします        
        self.game_over = False
        self.cpu_black_player = None
        self.cpu_white_player = None
        if black_cpu_dict:
            self.cpu_black_player = reversi.Player('black', black_cpu_dict["name"], black_cpu_dict["strategy"])
        if white_cpu_dict:
            self.cpu_white_player = reversi.Player('white', white_cpu_dict["name"], white_cpu_dict["strategy"])       
        ai_strategy = reversi.strategies.Usagi(base=reversi.strategies._NegaScout(
            depth = 6, # 6手先まで読む
            evaluator = evaluator()
        ))
        self.recommend_strategy = reversi.strategies.Switch(
            turns=[
                47,
                60             # (残り16手)からは完全読み
            ],
            strategies=[
                ai_strategy,
                reversi.strategies._EndGame()
            ],
        )
            
    def initialize_board(self):
        # オセロの初期盤面を設定するロジック
        return reversi.BitBoard()
    
    def get_board_str(self):
        return self.board.get_board_line_info(
            player=self.current_player, black='B', white='W', empty='.'
        )[0:64]
        

    def make_move(self, x=-1, y=-1, cpu_player=None):    
        # 手を実行し、盤面を更新するロジック
        self.board.put_disc(self.current_player, x, y)
        self.current_player = 'white' if self.current_player == 'black' else 'black'
        return None
     

    def get_valid_moves(self, color):        
        # 有効な手のリストを返すロジック
        legal_moves = self.board.get_legal_moves(color)
        return legal_moves
    
    def cpu_move(self, color):       
        if color == 'black':
            cpu_player = self.cpu_black_player
        else:
            cpu_player = self.cpu_white_player
            
        cpu_player.put_disc(self.board)
        self.current_player = 'white' if self.current_player == 'black' else 'black'
        return cpu_player.move
    
    def rec_move(self):
        rec_board = copy.deepcopy(self.board)
        move = self.recommend_strategy.next_move(
            self.current_player,
            rec_board
        )
        print(move)
        return move, rec_board.get_board_line_info(
            player=self.current_player, black='B', white='W', empty='.'
        )[0:64]

    def pass_turn(self):
        self.current_player = 'white' if self.current_player == 'black' else 'black'    

    def check_game_over(self):
        # ゲームが終了したかどうかを確認するロジック
        if not self.get_valid_moves('black') and not self.get_valid_moves('white'):
            return True
        if self.board._black_score == 0 and self.board._white_score == 0:
            return True
        
        return False

    def get_game_result(self):
        # ゲームの結果を返すロジック
        if self.board._black_score > self.board._white_score:
            return "Black WIN!", self.board._black_score - self.board._white_score
        elif self.board._white_score > self.board._black_score:
            return "White WIN!", self.board._white_score - self.board._black_score
        else:
            return "Draw", 0