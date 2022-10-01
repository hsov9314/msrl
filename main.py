import numpy as np
import torch

from minesweeper import MinesweeperEnv
from agent import Agent


def obsTransform(obs, row, col):
    availableAction = np.zeros((row, col))
    availableAction.fill(-np.inf)
    state = np.zeros((row, col))
    for i in range(row):
        for j in range(col):
            if np.isnan(obs[i, j]):
                availableAction[i, j] = 0
                state[i, j] = -1
            else:
                state[i, j] = obs[i, j]
    state = torch.from_numpy(state.flatten()).type(torch.FloatTensor)
    state = torch.unsqueeze(state, 0)
    availableAction = torch.from_numpy(availableAction.flatten()).type(
        torch.FloatTensor
    )
    availableAction = torch.unsqueeze(availableAction, 0)

    return state, availableAction


# main
import csv

with open("data.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["episode", "step", "win rate"])

row = 6
col = 6
numMines = 6
env = MinesweeperEnv(row, col, numMines)
num_states = row * col
num_actions = row * col
agent = Agent(num_states, num_actions)  # 環境内で行動するAgentを生成

NUM_EPISODES = 1  # 最大試行回数
win = 0
lose = 0

step = 0  # ステップ数

for episode in range(1, NUM_EPISODES + 1):
    ######******   訓練条件   ******######
    obs = env.reset()

    state, availableAction = obsTransform(obs, row, col)
    reward = 0
    done = False

    while not done:  # 1エピソードのループ
        action = agent.get_action(state, availableAction, env)  # 行動を求める
        coordinates = divmod(action.item(), col)
        obs, reward, done, _ = env.step(coordinates)
        reward = torch.FloatTensor([reward])  # 報酬0
        state_next, availableAction = obsTransform(obs, row, col)
        # 終了状態なら価値 0
        if done:
            state_next = None
            availableAction = None
        # メモリに経験を追加
        agent.memorize(state, action, state_next, reward, availableAction)
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
        win = 0
        lose = 0

    if episode % 1000 == 0:
        agent.brain.model.eval()
        winTest = 0
        for i in range(1000):
            ######******   テスト条件   ******######
            obs = env.reset()

            state, availableAction = obsTransform(obs, row, col)
            done = False
            while not done:
                with torch.no_grad():
                    value = agent.brain.model(state)
                    action = (value + availableAction).max(1)[1].view(1, 1)
                coordinates = divmod(action.item(), col)
                obs, reward, done, _ = env.step(coordinates)
                state_next, availableAction = obsTransform(obs, row, col)
                state = state_next
            if env.won:
                winTest += 1
        print(winTest)
        with open("data.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow([episode, step, winTest])
        winTest = 0
torch.save(agent.brain.model.state_dict(), "model.pth")
