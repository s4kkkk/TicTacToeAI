import torch as t
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import random
from game import TicTacToe
import pdb
import code
"""
Два класса. Первый представляет собой неройсеть, которую будет использовать агент,
а второй - сам агент
"""
# Устройство, на котором будет выполняться вычисления.
# Если "cuda" не работает, то выбрать "cpu"

DEVICE = "cuda"


class Neuro(nn.Module):

    def __init__(self, LearningRate=0.001, Epsilon=0.3):
        super().__init__()

        # Структура сети - перцептрон с 1 скрытым слоём
        self.__inp = nn.Linear(9, 128).to(DEVICE)
        self.__hidden1 = nn.Linear(128, 128).to(DEVICE)
        self.__hidden2 = nn.Linear(128, 128).to(DEVICE)
        self.__hidden3 = nn.Linear(128, 128).to(DEVICE)
        self.__out = nn.Linear(128, 9).to(DEVICE)

        # Инициализация слоев
        init.uniform_(self.__inp.weight, -1, 1)
        init.uniform_(self.__hidden1.weight, -1, 1)
        init.uniform_(self.__hidden2.weight, -1, 1)
        init.uniform_(self.__hidden3.weight, -1, 1)
        init.uniform_(self.__out.weight, -1, 1)

        self.__Epsilon = Epsilon
        self.__optimizer = optim.Adam(self.parameters(), lr=LearningRate)
        # self.__optimizer = optim.SGD(self.parameters(), lr=0.1)
        self.__activation = nn.LeakyReLU()
        # self.__criterion = nn.CrossEntropyLoss()
        # self.__criterion = nn.MSELoss()

    # Прямой проход
    def forward(self, x):
        x = t.sigmoid(self.__inp(x))
        # breakpoint()
        x = t.sigmoid(self.__hidden1(x))
        # breakpoint()
        x = t.sigmoid(self.__hidden2(x))
        x = t.sigmoid(self.__hidden3(x))
        x = self.__activation(self.__out(x))
        # breakpoint()
        # x = x/sum(x)
        x = t.softmax(x, dim=-1)
        # print(x)
        return x

    # Предсказание следующего шага
    def predict_step(self, state: t.FloatTensor) -> (bool, t.IntTensor):
        NeuroAnswer = self.forward(state)

        # получаем тензор доступных ходов
        available_steps = t.where(state == 0.01)[0]
        step = NeuroAnswer.argmax()

        # проверка на то, что предсказание сети соответствует одному
        # из доступных ходов
        if (not t.any(available_steps == step)):
            return (True, step)

        # с вероятностью epsilon выбираем случайное действие
        if (random.random() < self.__Epsilon):
            step = random.choice(available_steps)

        return (False, step)

    # Метод для обучения сети
    # states - тензор состояний в ходе эпизода
    # actions - тензор индексов (позиций на поле), на которые походил агент
    # win - в ходе эпизода агент выиграл?
    def train_(self, states: t.FloatTensor, actions: t.IntTensor, win: bool):
        reward = None
        if (win):
            reward = 100
        else:
            reward = -100

        # прямой проход через сеть
        net_answer = self.forward(states)
        # выбираем элементы, соответствующие действиям
        predicts = net_answer[range(len(actions)), actions]

        # вычисление логарифмов
        log_predicts = t.log(predicts)

        # policy-gradient. Получаем сумму произведений логарифмов на награду

        loss = -t.sum(log_predicts * reward)
        self.__optimizer.zero_grad()
        loss.backward()
        self.__optimizer.step()
        # print(f"Текущая ошибка: {loss}")


# Класс, представляющий собой игрока, выполняющего случайные ходы
class RandomPlayer:

    def __init__(self, PlayerNumber: int):
        self.__playerNumber = PlayerNumber

    # Шаг игрока. Возвращаемое значение - результат
    def step(self, game: TicTacToe) -> int:
        current_state, EndStatus = game.get_current_state()
        if (EndStatus != 0):
            return EndStatus

        current_state_tensor = t.FloatTensor(current_state).to(DEVICE)

        if (t.all(current_state_tensor != 0.01)):
            # ничья. нет доступных ходов.
            return 3

        available_steps = t.where(current_state_tensor == 0.01)[0]
        step = random.choice(available_steps)
        game.step(self.__playerNumber, step.item())

        current_state, EndStatus = game.get_current_state()
        if (EndStatus != 0):
            return EndStatus
        return 0


# Класс, представляющий агента (игрока)
class Player:

    # playerNumber - номер игрока. 1 - крестик, 2 - нолик
    def __init__(self, playerNumber: int, LearningRate=0.001, Epsilon=0.3):
        self.__neuro = Neuro(LearningRate, Epsilon).to(DEVICE)
        self.__playerNumber = playerNumber

        # Память для сохранения игр
        self.__memory_states = []
        self.__memory_actions = []
        self.__memory_win = False
        self.__memory_lose = False
        self.memory_len = 0

    # Метод для запоминания ходов в ходе игры
    def __remember(self, state, action):
        self.__memory_states.append(state)
        self.__memory_actions.append(action)
        self.memory_len += 1

        # Метод для получения тензоров для обучения
    def __samplebatch(self) -> (t.FloatTensor, t.IntTensor):
        states = t.FloatTensor(self.__memory_states).to(DEVICE)
        self.__memory_states.clear()

        actions = t.IntTensor(self.__memory_actions).to(DEVICE)
        self.__memory_actions.clear()

        self.memory_len = 0
        return (states, actions)

    # Шаг игрока. Возвращаемое значение - результат
    def step(self, game: TicTacToe) -> int:
        current_state, EndStatus = game.get_current_state()
        if (EndStatus != 0):
            if (EndStatus == self.__playerNumber):
                self.__memory_win = True
            else:
                self.__memory_lose = True
            return EndStatus

        current_state_tensor = t.FloatTensor(current_state).to(DEVICE)
        if (t.all(current_state_tensor != 0.01)):
            # ничья. нет доступных ходов.
            self.__memory_lose = True
            return 3

        (isError, action) = self.__neuro.predict_step(current_state_tensor)

        if (isError):
            self.__memory_lose = True
            self.__remember(current_state, action.item())
            if (self.__playerNumber == 1):
                return 2
            else:
                return 1

        game.step(self.__playerNumber, action.item())
        self.__remember(current_state, action.item())

        current_state, EndStatus = game.get_current_state()
        if (EndStatus != 0):
            if (EndStatus == self.__playerNumber):
                self.__memory_win = True
            else:
                self.__memory_lose = True
            return EndStatus

        return 0

    # Метод для запуска обучения нейросети
    def train(self):
        if ((self.memory_len == 0)
                or (self.__memory_win == self.__memory_lose)):
            return

        states, actions = self.__samplebatch()
        self.__neuro.train_(states, actions, self.__memory_win)
        # сброс
        self.__memory_win = False
        self.__memory_lose = False

    # Метод для сравнения обученной нейросети с игроком,
    # выполняющим случайные шаги
    def validate(self, games=1000):
        randomPlayer = None
        if (self.__playerNumber == 1):
            randomPlayer = RandomPlayer(2)
        else:
            randomPlayer = RandomPlayer(1)
        game = TicTacToe()
        statistics = []

        for i in range(games):
            status = 0
            game.reset()
            while status == 0:
                player_ans = self.step(game)
                randomPlayer_ans = randomPlayer.step(game)
                if (player_ans == randomPlayer_ans != 0):
                    status = player_ans

            if (status == self.__playerNumber):
                statistics.append(1)
            else:
                statistics.append(0)
        player_wins_percent = (sum(statistics) / len(statistics)) * 100
        print(
            f"Валидация из {games} игр завершена. Выиграно: {player_wins_percent}%"
        )

    # метод для получения состояния нейросети (весов)
    def get_state(self):
        return self.__neuro.state_dict()

    # метод для загрузки состояния в нейросеть (загрузки весов)
    def load_state(self, state):
        self.__neuro.load_state_dict(state)


if __name__ == '__main__':
    player1 = Player(1, LearningRate=0.0001, Epsilon=0.)
    game = TicTacToe()

    statistics = []

    epochs = 10000
    games = 100

    player2 = None
    player2 = RandomPlayer(PlayerNumber=2)

    for i in range(epochs):
        isNeuro = False
        if (i % 2 == 0):
            player2 = RandomPlayer(PlayerNumber=2)
            print("РАНДОМ:", end="")
        else:
            player2 = Player(2, LearningRate=0.0001, Epsilon=0.)
            player2.load_state(player1.get_state())
            print("САМ С СОБОЙ:", end="")
            isNeuro = True

        for j in range(games):
            status = 0
            game.reset()
            while status == 0:
                player1_ans = player1.step(game)
                player2_ans = player2.step(game)
                if (player1_ans != 0 or player2_ans != 0):
                    status = player1_ans

            player1.train()

            if (isNeuro):
                player2.train()
            if (status == 1):
                statistics.append(1)
            else:
                statistics.append(0)
            # game.print_field()
        player_1_wins_percent = (sum(statistics) / len(statistics)) * 100
        statistics.clear()
        print(f" {player_1_wins_percent}%")
