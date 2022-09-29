import numpy as np
from typing import NamedTuple


class MinesweeperReward(NamedTuple):
    win: float = 1.0
    lose: float = -1.0
    progress: float = 0.0


class MinesweeperEnv:
    def __init__(self, row=9, col=9, numMines=10):
        self.row = row  # 行（高さ）
        self.col = col  # 列（横幅）
        self.numMines = numMines  # 地雷の個数

        self.mines = np.zeros([self.row, self.col])  # 地雷の位置
        self.neighbors = np.zeros([self.row, self.col])  # 隣接する地雷の個数
        self.state = np.zeros([self.row, self.col])  # 観測状態
        self.state.fill(np.nan)

        self.noOpenCell = np.ones(self.row * self.col)  # 開いているマス0, 開いていないマス1

        self.reward = MinesweeperReward()  # 報酬

        self.initialized = False  # 初期化判定
        self.won = False  # 成功判定

    # ゲームのリセット
    def reset(self):
        self.mines.fill(0)
        self.neighbors.fill(0)
        self.state.fill(np.nan)
        self.noOpenCell.fill(1)

        self.initialized = False
        self.won = False
        return self.state

    # 1ステップ進める
    def step(self, coordinates):
        reward = self.reward.progress
        done = False
        # 開いているマスを開けた 例外処理
        if not np.isnan(self.state[coordinates[0], coordinates[1]]):
            print("noprogress")
            reward = -1
        # 地雷マスを開けた
        if self.mines[coordinates[0], coordinates[1]] > 0:
            self.state[coordinates[0], coordinates[1]] = -100  # 地雷
            reward = self.reward.lose
            done = True
        #
        else:
            if not self.initialized:  # 初期化
                self.initializeBoard(coordinates)
                reward = 0.0
            # マスを開く
            self.openCell(coordinates)
            # 終了判定
            if np.sum(np.isnan(self.state)) == self.numMines:
                reward = self.reward.win
                done = True
                self.won = True

        return self.state, reward, done, {}

    # 盤面の初期化
    def initializeBoard(self, coordinates):
        # 最初のマスは0
        numTotalCells = self.row * self.col
        select = coordinates[0] * self.col + coordinates[1]
        offLimits = np.array(
            [
                select - self.col - 1,
                select - self.col,
                select - self.col + 1,
                select - 1,
                select,
                select + 1,
                select + self.col - 1,
                select + self.col,
                select + self.col + 1,
            ]
        )
        availableCells = np.setdiff1d(np.arange(numTotalCells), offLimits)
        # 最初のマスとその周辺以外に爆弾を配置する
        minesFlattend = np.zeros([numTotalCells])
        minesFlattend[
            np.random.choice(availableCells, self.numMines, replace=False)
        ] = 1
        self.mines = minesFlattend.reshape([self.row, self.col])
        # 隣接する地雷の個数
        for row in range(self.row):
            for col in range(self.col):
                numNeighbors = 0
                for i in range(-1, 2):
                    if row + i >= 0 and row + i < self.row:
                        for j in range(-1, 2):
                            if col + j >= 0 and col + j < self.col:
                                if not (i == 0 and j == 0):
                                    numNeighbors += self.mines[row + i, col + j]
                self.neighbors[row, col] = numNeighbors
        # 初期化終了
        self.initialized = True

    # (coordinates[0], coordinates[1]) のマスを開ける
    def openCell(self, coordinates):
        row = coordinates[0]
        col = coordinates[1]
        self.state[row, col] = self.neighbors[row, col]
        self.noOpenCell[row * self.col + col] = 0  # 開けたマスは0にする
        # 0なら周囲の開いていないマスも開ける
        if self.state[row, col] == 0:
            for i in range(-1, 2):
                if row + i >= 0 and row + i < self.row:
                    for j in range(-1, 2):
                        if col + j >= 0 and col + j < self.col:
                            if np.isnan(self.state[row + i, col + j]):
                                self.openCell([row + i, col + j])

    # ランダムアクション（開いていないマスを開ける）
    def randomAction(self):
        nonOpenCell = np.array(np.where(self.noOpenCell)).flatten()
        action = np.random.choice(nonOpenCell)
        return action

    # 最初の行動の前に地雷設置
    def resetRandomInit(self):
        self.mines.fill(0)
        self.neighbors.fill(0)

        numTotalCells = self.row * self.col
        availableCells = np.arange(numTotalCells)
        minesFlattend = np.zeros([numTotalCells])
        minesFlattend[
            np.random.choice(availableCells, self.numMines, replace=False)
        ] = 1
        self.mines = minesFlattend.reshape([self.row, self.col])
        # 隣接する地雷の個数
        for row in range(self.row):
            for col in range(self.col):
                numNeighbors = 0
                for i in range(-1, 2):
                    if row + i >= 0 and row + i < self.row:
                        for j in range(-1, 2):
                            if col + j >= 0 and col + j < self.col:
                                if not (i == 0 and j == 0):
                                    numNeighbors += self.mines[row + i, col + j]
                self.neighbors[row, col] = numNeighbors

        self.state.fill(np.nan)
        self.noOpenCell.fill(1)

        self.initialized = True
        self.won = False
        return self.state

    # 固定の地雷配置パターン
    def ResetAndSetMines(self, mines):
        self.mines = np.copy(mines)
        # 隣接する地雷の個数
        for row in range(self.row):
            for col in range(self.col):
                numNeighbors = 0
                for i in range(-1, 2):
                    if row + i >= 0 and row + i < self.row:
                        for j in range(-1, 2):
                            if col + j >= 0 and col + j < self.col:
                                if not (i == 0 and j == 0):
                                    numNeighbors += self.mines[row + i, col + j]
                self.neighbors[row, col] = numNeighbors

        self.state.fill(np.nan)
        self.noOpenCell.fill(1)

        self.initialized = True
        self.won = False
        return self.state
