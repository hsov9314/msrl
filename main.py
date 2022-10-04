import numpy as np
import torch

from minesweeper import MinesweeperEnv
from agent import Agent

# main
import csv
import datetime

dtnow = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
result_csv_path = f"data_{dtnow}.csv"
with open(result_csv_path, "w") as f:
    writer = csv.writer(f)
    writer.writerow(["episode", "win", "lose", "win_rate"])

row = 6
col = 6

cell_image_w = 14
cell_image_h = 14
screen_w = col * cell_image_w
screen_h = row * cell_image_h

numMines = 6
env = MinesweeperEnv(row, col, numMines)
num_states = row * col
num_actions = row * col

# 環境内で行動するAgentを生成
agent = Agent(screen_w * screen_h, num_actions)

NUM_EPISODES = 100  # 最大試行回数
win = 0
lose = 0

step = 0  # ステップ数

for episode in range(1, NUM_EPISODES + 1):
    ######******   訓練条件   ******######
    obs = env.reset()

    state = env.stateImage()
    # state = torch.reshape(state, (-1, screen_w * screen_h))
    reward = 0
    done = False

    while not done:  # 1エピソードのループ
        action = agent.get_action(state, env)
        coordinates = divmod(action.item(), col)
        obs, reward, done, _, state_next = env.step(coordinates)
        reward = torch.FloatTensor([reward])  # 報酬0

        # state_next = torch.reshape(state_next, (-1, screen_w * screen_h))
        # 終了状態なら価値 0
        if done:
            state_next = None
        # メモリに経験を追加
        agent.memorize(state, action, state_next, reward)
        # Experience ReplayでQ関数を更新する
        agent.update_q_function()
        # 観測の更新
        state = state_next
        # ステップ +1
        step += 1
    # イプシロンを減衰
    agent.updateEpsilon()

    if env.won:
        win += 1
    else:
        lose += 1
    if episode % 100 == 0:
        print("==== Episode {} : win {}, lose {} ====".format(episode, win, lose))
        with open(result_csv_path, "a") as f:
            writer = csv.writer(f)
            writer.writerow([episode, win, lose, win / (win + lose)])
        win = 0
        lose = 0

    # if episode % 1000 == 0:
    #     # agent.brain.model.eval()
    #     winTest = 0
    #     for i in range(1000):
    #         ######******   テスト条件   ******######
    #         obs = env.reset()

    #         state = env.stateImage()
    #         state = torch.reshape(state, (-1, screen_w * screen_h))
    #         done = False
    #         step = 0
    #         while not done:
    #             step += 1
    #             if step > 10:
    #                 break
    #             with torch.no_grad():
    #                 value = agent.brain.model(state)
    #                 action = (value).max(1)[1].view(1, 1)
    #             coordinates = divmod(action.item(), col)
    #             obs, reward, done, _, state_next = env.step(coordinates)
    #             state_next = torch.reshape(state, (-1, screen_w * screen_h))
    #             state = state_next
    #         if env.won:
    #             winTest += 1
    #     with open(result_csv_path, "a") as f:
    #         writer = csv.writer(f)
    #         writer.writerow([episode, step, winTest])
    #     winTest = 0

torch.save(agent.brain.model.state_dict(), "model.pth")
