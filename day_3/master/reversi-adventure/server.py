from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from tensorflow import keras
from othello import OthelloGame
from utils import utils
import reversi

app = Flask(__name__)
CORS(app)

class MyEvaluator(reversi.strategies.common.AbstractEvaluator):
    def evaluate(self, color, board, possibility_b, possibility_w):
        player = 0 if color == 'black' else 1
        board_list = board.get_board_info()
        board_str = board.get_board_line_info(player='black', black='0', white='1', empty='.')
        score = game.ai.predict_score(
            board_list=board_list, 
            board_str=board_str, 
            player=player, 
            lmodel=game.lmodel # ここに作ったモデルを入れてください！
        )
        return score

model = keras.models.load_model("models/model_0220.h5")

# AIが予測を返す部分を高速化させます
lmodel = utils.LiteModel.from_keras_model(model)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start_game():
    global game, model, lmodel
    data = request.get_json()
    cpu_strategy = data.get('cpuStrategy')
    if cpu_strategy == "slime":
        cpu_dict = {'name':  'スライム', 'strategy': reversi.strategies.Random()}
    elif cpu_strategy == "dragon":
        cpu_dict = {'name':  'ドラゴン', 'strategy': reversi.strategies.MonteCarlo(count=100)}
    elif cpu_strategy == "maou":
        cpu_dict = {'name': '魔王', 'strategy': reversi.strategies.MonteCarlo_EndGame(count=10000, end=14)}
    else:
        cpu_dict = None
    game = OthelloGame(lmodel=lmodel, white_cpu_dict = cpu_dict, evaluator=MyEvaluator)    
    
    return jsonify({"board": game.get_board_str(),  "current_player": game.current_player})


@app.route('/move', methods=['POST'])
def make_move():
    if game.check_game_over():
        return jsonify({"error": "ゲームオーバー"}), 400

    x = request.json.get('x')
    y = request.json.get('y')
    game.make_move(x, y)
    
    if game.cpu_white_player:
        is_cpu_player = True
    else:
        is_cpu_player = False
    
    return jsonify({"board": game.get_board_str(), "valid_moves": game.get_valid_moves(game.current_player), "current_player": game.current_player, "is_cpu_player": is_cpu_player })


@app.route('/cpu_move', methods=['GET'])
def cpu_move():
    if game.check_game_over():
        return jsonify({"error": "Game is over"}), 400
    # 現在のプレイヤーの有効な手を取得
    valid_moves = game.get_valid_moves(game.current_player)    
    if valid_moves:
        move = game.cpu_move(game.current_player)
    else:
        move = []
    return jsonify({"board": game.get_board_str(), "valid_moves": valid_moves, "cpu_move": move, "current_player": game.current_player})


@app.route('/status', methods=['GET'])
def game_status():
    return jsonify({
        "current_player": game.current_player,
        "board": game.get_board_str(),
        "game_over": game.check_game_over(),
        "valid_moves": game.get_valid_moves(game.current_player)
    })


@app.route('/pass', methods=['POST'])
def pass_turn():
    if game.check_game_over():
        return jsonify({"error": "Game is over"}), 400

    game.pass_turn()  # 手番を交代させる関数
    if game.cpu_white_player:
        is_cpu_player = True
    else:
        is_cpu_player = False    
    return jsonify({"board": game.get_board_str(), "current_player": game.current_player, "is_cpu_player": is_cpu_player})


@app.route('/recommend', methods=['GET'])
def recommend_move():
    if game.check_game_over():
        return jsonify({"error": "Game is over"}), 400

    recommended_move, board = game.rec_move()
    return jsonify({"move": recommended_move, "board": board})


@app.route('/result', methods=['GET'])
def game_result():
    if not game.check_game_over():
        return jsonify({"error": "Game is not over"}), 400
    result, diff = game.get_game_result()
    return jsonify({"result": result, "board": game.get_board_str(), "diff": diff})

if __name__ == '__main__':
    app.run(debug=True)