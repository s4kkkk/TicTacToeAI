#!bin/python
""""
Класс, реализующий игру в крестики-нолики;
step() - метод для выполнения шага игры, возвращает кортеж из состояния поля
(позиции крестиков-ноликов) и флага окончания игры.
Состояние поля - массив из 9 элементов, соответствующих клеткам поля.
Значение в массиве: 0 - клетка пустая, 1 - в клетке крестик,
2 - в клетке нолик.
"""


class TicTacToe:

    def __init__(self):
        self.__field = [0.01] * 9
        self.__win_combinations = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6],
                                   [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]]
        self.__is_end = False

    # сброс игры
    def reset(self):
        self.__field = [0.01] * 9
        self.__is_end = False

    # метод для вывода поля в консоль
    def print_field(self):
        for i in range(3):
            for j in range(3):
                current_value = self.__field[3 * i + j]
                if (current_value == 0.01):
                    print("-%d-" % (3 * i + j), end="")
                    continue
                elif (current_value == 1):
                    print("-X-", end="")
                    continue
                else:
                    print("-O-", end="")
                    continue
            print("")

    # метод для получения состояния
    # возвращает кортеж из состояния игры (0 - игра не
    # закончена, 1 - выиграл крестик, 2 - выиграл нолик) и
    # состояния поля
    def get_current_state(self) -> (list[int], int):
        return (self.__field.copy(), self.__who_win())

    # шаг игры. player - игрок. 1 - крестик, 2 - нолик;
    # position - позиция для шага.
    # возвращает кортеж состояние игры (0 - игра не
    # закончена, 1 - выиграл крестик, 2 - выиграл нолик)
    def step(self, player: int, position: int):
        if (not ((player == 1) or (player == 2))):
            return
        if ((position < 0) or (position > 8)):
            return
        if (self.__field[position] != 0.01):
            return
        self.__field[position] = player
        return

    def __who_win(self) -> int:
        for indices in self.__win_combinations:
            if (self.__field[indices[0]] == self.__field[indices[1]] ==
                    self.__field[indices[2]] == 0.01):
                return 0
            elif (self.__field[indices[0]] == self.__field[indices[1]] ==
                  self.__field[indices[2]]):
                return self.__field[indices[0]]
        return 0
