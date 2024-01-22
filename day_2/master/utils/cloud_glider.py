import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import animation, rc
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import japanize_matplotlib
import sys

sys.path.append('..')

from utils import db_utils

DU = db_utils.DbUtils('aws')


class Obstacle:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def check_collision(self, agent_x, agent_y):
        return (self.x - self.width/2 < agent_x < self.x + self.width/2) and \
               (self.y - self.height/2 < agent_y < self.y + self.height/2)

    
class Brain:
    def __init__(self, n_state=20, w_y=0.2, w_vy=0.2, n_action=2, gamma=0.9, r=0.995, lr=0.01):
        self.n_state = n_state  # 状態の数
        self.w_y = w_y  # 位置の刻み幅
        self.w_vy = w_vy  # 速度の刻み幅
        self.n_action = n_action  # 行動の数

        self.eps = 1.0  # ε
        self.gamma = gamma  # 割引率
        self.r = r  # εの減衰率
        self.lr = lr  # 学習係数

        self.q_table = np.zeros((n_state*n_state, n_action)) # Q-Table

    def show_q_table(self):
        # 状態と行動の名称を生成
        states = [f's{i}' for i in range(1, self.n_state * self.n_state + 1)]  # s1からs400まで
        actions = [f'行動: a{j}' for j in range(1, self.n_action + 1)]  # a1とa2

        # pandas DataFrameを作成
        q_table_df = pd.DataFrame(self.q_table, index=states, columns=actions)
        q_table_df.index.name = '状態'
        return q_table_df

    def quantize(self, state, n_state, w):  # 状態の値を整数のインデックスに変換
        min = - n_state / 2 * w
        nw = (state - min) / w
        nw = int(nw)
        nw = 0 if nw < 0 else nw
        nw = n_state-1 if nw >= n_state-1 else nw
        return nw

    def train(self, states, next_states, action, reward, terminal):  # Q-Tableを訓練
        i = self.quantize(states[0], self.n_state, self.w_y)  # 位置のインデックス
        j = self.quantize(states[1], self.n_state, self.w_vy)  # 速度のインデックス
        q = self.q_table[i*self.n_state+j, action]  # 現在のQ値

        next_i = self.quantize(next_states[0], self.n_state, self.w_y)  # 次の位置のインデックス
        next_j = self.quantize(next_states[1], self.n_state, self.w_vy)  # 次の速度のインデックス
        q_next = np.max(self.q_table[next_i*self.n_state+next_j])  # 次の状態で最大のQ値

        if terminal:
            self.q_table[i*self.n_state+j, action] = q + self.lr*reward  # 終了時は報酬のみ使用
        else:
            self.q_table[i*self.n_state+j, action] = q + self.lr*(reward + self.gamma*q_next - q)  # Q値の更新式

    def get_action(self, states, frames):
        if np.random.rand() < self.eps:  # ランダムな行動
            action = np.random.randint(self.n_action)
        else:  # Q値の高い行動を選択
            i = self.quantize(states[0], self.n_state, self.w_y)
            j = self.quantize(states[1], self.n_state, self.w_vy)
            action = np.argmax(self.q_table[i*self.n_state+j])
        if self.eps > 0.1:  # εの下限
            self.eps *= self.r
            # 指数関数的に減衰させる場合
            #self.eps_decay = (0.1 / 1.0) ** (1 / 60)
            #self.eps *= self.eps_decay
        return action

    
class Agent:
    def __init__(self, brain, v_x=0.05, v_y_sigma=0.1, v_jump=0.2):
        self.v_x = v_x  # x速度
        self.v_y_sigma = v_y_sigma  # y速度、初期値の標準偏差
        self.v_jump = v_jump  # ジャンプ速度
        self.brain = brain
        self.reset()

    def reset(self):
        self.x = -2.4  # 初期x座標
        self.y = 0.5  # 初期y座標
        self.v_y = self.v_y_sigma * np.random.randn()  # 初期y速度

    def step(self, g, obstacles, frames):  # 時間を進めるメソッドに障害物のリストを追加
        states = np.array([self.y, self.v_y])
        self.x += self.v_x
        self.y += self.v_y

        reward = 0  # 報酬
        terminal = False  # 終了判定

        if self.x > -1.0:
            reward = 0.3
        if self.x > 0.0:
            reward = 0.5
        if self.x > 1.75:
            reward = 1
            terminal = True
        elif self.y < -1.75 or self.y > 2.0:
            reward = -1
            terminal = True

        # 障害物との衝突判定を追加                                                                                                                           
        for ob in obstacles:
            if ob.check_collision(self.x, self.y):
                reward = -1
                terminal = True
                break
            
        action = self.brain.get_action(states, frames)
        if action == 0:
            self.v_y -= g  # 自由落下
        else:
            self.v_y = self.v_jump  # ジャンプ

        next_states = np.array([self.y, self.v_y])
        self.brain.train(states, next_states, action, reward, terminal)

        if terminal:
            self.reset()
    
        return self.x, self.y, reward

    
class Environment:
    def __init__(self, agent, g=0.2, num_obstacles=0):
        self.agent = agent
        self.g = g  # 重力加速度
        self.obstacles = [Obstacle(np.random.uniform(-1.75, 1.75), 
                                   np.random.uniform(-1.4, 0.9), 
                                   0.2, 0.2) for _ in range(num_obstacles)]
        self.success_count = 0
        self.frames = 0
        self.frame_count = 0
        self.total_games = 0
        self.success = False
        self.success_rate = 0  # 初期の成功率
        self.update_text = ''

    def step(self):
        # Agentのstepメソッドに障害物のリストを渡す
        x, y, reward = self.agent.step(self.g, self.obstacles, self.frames)
        
        # ゲームが終了したら試行回数を更新し、成功率を計算
        if reward == 1 or reward == -1:  # 成功または失敗
            self.total_games += 1
            if self.update_text != '':
                self.update_text += "\n"            
            if reward == 1:
                self.update_text = self.update_text + f"{self.total_games}回目: ○"
                self.success = True
                self.success_count += 1
            else:
                self.update_text = self.update_text + f"{self.total_games}回目: ×"
        return x, y, reward


def show_env_animation(environment, frames, interval=100):
    fig, ax = plt.subplots(figsize=[10,8])
    plt.close()
    # x軸とy軸の数値表示を消去
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim((-2.5, 2.0))
    ax.set_ylim((-1.75, 2.0))
    
    # ゴールの位置に目立つ線を描画
    ax.axvline(x=1.75, color='gold', linestyle='--', linewidth=3)

    # 背景色の設定（オプション）
    ax.set_facecolor('skyblue')    

    # 障害物の画像を読み込む
    obstacle_image = plt.imread('img/obstacle.png')
    obstacle_images = []
    for ob in environment.obstacles:
        imagebox = OffsetImage(obstacle_image, zoom=0.2)
        obstacle_ab = AnnotationBbox(imagebox, (ob.x, ob.y), frameon=False)
        ax.add_artist(obstacle_ab)
        obstacle_images.append(obstacle_ab)    

    # エージェントの画像を読み込む
    agent_image = plt.imread('img/agent_image.png')
    imagebox = OffsetImage(agent_image, zoom=0.4)
    agent_ab = AnnotationBbox(imagebox, (0,0), frameon=False)
    ax.add_artist(agent_ab)
    frame_cnt = 0
    # ステータステキストオブジェクトを一度だけ作成
    status_text = ax.text(0.5, 1.05, '', transform=ax.transAxes, ha='center', size=20)    

    def plot(frame):
        nonlocal frame_cnt
        x, y, reward = environment.step()

        agent_ab.xybox = (x, y)
        # for obstacle_ab, obstacle in zip(obstacle_images, environment.obstacles):
        #    obstacle_ab.xybox = (obstacle.x, obstacle.y)

        frame_cnt += 1
        
        if environment.success and environment.frame_count == 0:
            environment.frame_count = frame_cnt
            
        # テキストとその色を更新
        if reward == 1:
            status_text.set_text('成功')
            status_text.set_color('green')
        elif reward == -1:
            status_text.set_text('失敗')
            status_text.set_color('red')
        else:
            status_text.set_text('')

        return (agent_ab, *obstacle_images, status_text)
    
    if frames == 0:
        plot(0)
        return fig    
    

    anim = animation.FuncAnimation(fig, plot, interval=interval, frames=frames, repeat=False, blit=True)
    return anim


def update_result(state, user_name, scoreboard):
    environment = state['env']
    frames = environment.frames
    # 100点満点で50フレーム(5秒)ごとに3点減点
    time_score = 100 - (environment.frame_count // 50) * 3
    obst_score = (len(environment.obstacles) + 1) * 100
    score = time_score + obst_score if environment.success else 0

    # dbにスコアを登録
    DU.regist_score('rl-score', user_name, score, len(environment.obstacles))

    result_text = "成功です！" if environment.success else "失敗です"

    result_text += f"  障害物: {len(environment.obstacles)}個,  経過時間: {environment.frame_count/10},  スコア: {score}点"

    ranking_data = DU.get_top_n_player('rl-score')
    scoreboard = ''
    for data in ranking_data:
        scoreboard += f"  ユーザー名: {data[1]},  障害物: {data[4]}個, スコア: {data[2]}点\n"
    return result_text, scoreboard, environment.update_text


def clear_user_name():
        return '', 'クリアしました。'


def create_animation_in_gr(max_time_input, state, num_obstacles, user_name, animation_output):
    if user_name == '':
        return animation_output, '「ユーザー名」に名前を入力してください。'
    anim = create_animation(max_time_input, state, num_obstacles)
    video_path = 'animation.mp4'
    anim.save(video_path, writer='ffmpeg')
    return video_path, user_name
    
    


def create_animation(max_time_input, state={"env": None}, num_obstacles=0, env=None, r=0.995, random=False):
    if env is None:
        n_state = 20
        w_y = 0.2
        w_vy = 0.2
        n_action = 2
        if random:
            r = 1.0
        brain = Brain(n_state, w_y, w_vy, n_action, r=r)

        v_x = 0.05
        v_y_sigma = 0.1
        v_jump = 0.2
        agent = Agent(brain, v_x, v_y_sigma, v_jump)

        g = 0.2
        environment = Environment(agent, g, int(num_obstacles))
    else:
        if random:
            env.agent.brain.r = 1.0
        else:
            env.agent.brain.r = r
        environment = env

    interval = 100  # ミリ秒
    frames = int((max_time_input * 1000) / interval)  # 総フレーム数を計算
    environment.frames = frames
    state['env'] = environment
    
    anim = show_env_animation(environment, frames, interval)
    return anim
