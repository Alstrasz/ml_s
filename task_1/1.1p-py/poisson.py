import random

import numpy as np
import util
import matplotlib.pyplot as plt


def main(lr, train_path, eval_path, save_path):
    """Задача: Регрессия Пуассона с градиентным подъемом.

    Аргументы:
        lr: Скорость обучения (learning rate) для градиентного подъема.
        train_path: Путь к CSV файлу, содержащему обучающую выборку.
        eval_path: Путь к CSV файлу, содержащему тестовую выборку.
        save_path: Путь к файлу для сохранения результата прогнозирования.
    """
    # Загружаем обучающую выборку
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_val, y_val = util.load_dataset(eval_path, add_intercept=True)

    pr = PoissonRegression(theta_0=np.zeros(5))

    data = pr.fit(x_train, y_train)

    print('delta', np.sum((data[-1] - data[-2])*(data[-1] - data[-2])))

    # data = list(map(lambda x: np.sum(x * x), data))

    # plt.figure()
    # plt.plot(range(len(data)), data, 'bx', linewidth=2)
    # plt.show()

    res = pr.predict(x_val)

    print('y_val', y_val)
    print('theta', pr.theta)
    print('res', res)
    print('diffs', res - y_val)
    print('mean_abs_error', np.mean(np.abs(res - y_val)))

    plt.figure()
    plt.plot(y_val, res, 'bx', linewidth=2)
    plt.xlabel('True value')
    plt.ylabel('Predicted value')
    plt.savefig(save_path)
    plt.show()
    # util.plot(x_train, y_train, pr.theta, './')
    # util.plot(y_val, res, pr.theta, './')

    # *** НАЧАЛО ВАШЕГО КОДА ***
    # Обучите модель регрессии Пуассона
    # Прогоните обученную модель на тестовой выборке и сохраните результат в файле с именем save_path
    # *** КОНЕЦ ВАШЕГО КОДА ***


class PoissonRegression:
    """Регрессия Пуассона.

    Пример использования:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, step_size=1e-5, max_iter=500000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Аргументы:
            step_size: Скорость обучения (learning rate).
            max_iter: Максимальное количество итераций.
            eps: Порог для определения сходимости.
            theta_0: Начальное значение theta. Если None, используется нулевой вектор.
            verbose: Печатать значения функции потерь во время обучения.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Выполнить градиентный подъем для максимизации функции правдоподобия регрессии Пуассона.

        Аргументы:
            x: Обучающие примеры. Размерность (n_examples, dim).
            y: Классы. Размерность (n_examples,).
        """
        # *** НАЧАЛО ВАШЕГО КОДА ***

        data = []
        old = 0
        for i in range(self.max_iter):
            if np.floor(i * 100 / self.max_iter) > old:
                old = np.floor(i * 100 / self.max_iter)
                print('done: ', old, '%')
            rand = random.randint(0, len(x) - 1)
            x_iter = x[rand]
            y_iter = y[rand]
            data.append(self.theta)
            self.theta = self.theta + self.step_size * (y_iter - np.exp(x_iter @ self.theta)) * x_iter
        return data

        # for x_iter, y_iter in zip(x, y):
        #    # print(x_iter @ self.theta)
        #    self.theta = self.theta + self.step_size * (y_iter - np.exp(x_iter @ self.theta)) * x_iter
        # *** КОНЕЦ ВАШЕГО КОДА ***

    def predict(self, x):
        """Выдать прогноз для значений x.

        Аргументы:
            x: Входные данные размерности (n_examples, dim).

        Возвращаемое значение:
            Вещественный прогноз для каждого входного знчаения, размерность (n_examples,).
        """
        # *** НАЧАЛО ВАШЕГО КОДА ***

        return np.exp(x @ self.theta.T)

        # *** КОНЕЦ ВАШЕГО КОДА ***


if __name__ == '__main__':
    main(lr=1e-5,
         train_path='train.csv',
         eval_path='valid.csv',
         save_path='poisson_pred.jpg')
