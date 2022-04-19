import numpy as np
import util


def main(train_path, valid_path, save_path):
    """Задача: Гауссовский дискриминантный анализ (ГДА)

    Аргументы:
        train_path: Путь к CSV-файлу, содержащему обучающую выборку.
        valid_path: Путь к CSV-файлу, содержащему валидационную выборку.
        save_path: Путь к файлу, в котором надо сохранить прогнозы.
    """
    # Загружаем выборку
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=False)
    # *** НАЧАЛО ВАШЕГО КОДА ***
    # Обучите ГДА классификатор
    gda = GDA(theta_0=np.zeros(len(x_train[0])))
    gda.fit(x_train, y_train)
    util.plot(x_valid, y_valid, gda.theta, save_path)
    # Нарисуйте решающую границу поверх валидационной выборки
    # Используйте np.savetxt, чтобы сохранить прогнозы на валидационной выборке в файле save_path
    # *** КОНЕЦ ВАШЕГО КОДА ***


class GDA:
    """Гауссовский дискриминантный анализ.

    Пример использования:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, step_size=0.01, max_iter=10000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Аргументы:
            step_size: Шаг обучения для итеративных решателей.
            max_iter: Максимальное количество итераций для решателя.
            eps: Пороговое значение для определения сходимости.
            theta_0: Начальное значение theta. Если None, необходимо использовать вектор нулей.
            verbose: Печатать значения функции потерь во время обучения.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Подгоните ГДА модель к обучающей выборке, заданной в x и y, обновляя self.theta.

        Аргументы:
            x: Входные вектора обучающей выборки. Размерность (n_examples, dim).
            y: Метки обучающей выборки. Размерность (n_examples,).
        """
        # *** НАЧАЛО ВАШЕГО КОДА ***
        # Найти phi, mu_0, mu_1 и sigma
        n = len(x[0])
        sy_0, sy_1, sy_0x, sy_1x = 0, 0, 0, 0
        for i in range(len(y)):
            if y[i] == 0:
                sy_0 += 1
                sy_0x += x[i]
            if y[i] == 1:
                sy_1 += 1
                sy_1x += x[i]
        print(sy_0, sy_1, sy_0x, sy_1x)
        phi = sy_1 / len(y)
        mu_0 = sy_0x / sy_0
        mu_1 = sy_1x / sy_1
        acc = 0
        for i in range(len(y)):
            if y[i] == 0:
                acc += (x[i] - mu_0).reshape((n, 1)) @ (x[i] - mu_0).reshape((1, n))
            if y[i] == 1:
                acc += (x[i] - mu_1).reshape((n, 1)) @ (x[i] - mu_1).reshape((1, n))
        sigma = acc / len(y)
        inv_sigma = np.linalg.inv(sigma)
        print('params', phi, mu_0, mu_1, sigma)

        self.theta = np.zeros(n + 1)
        self.theta[0] = -0.5 * np.log(1 / phi - 1) * (mu_0 @ inv_sigma @ mu_0 - mu_1 @ inv_sigma @ mu_1)
        self.theta[1:] = -(mu_1 - mu_0) @ inv_sigma
        print('theta', self.theta)
        # Выразите theta через параметры
        # *** КОНЕЦ ВАШЕГО КОДА ***

    def predict(self, x):
        """Сделать прогноз на новых входных данных x.

        Аргументы:
            x: Входные данные. Размерность (n_examples, dim).

        Возвращаемое значение:
            Прогноз размерности (n_examples,).
        """
        # *** НАЧАЛО ВАШЕГО КОДА ***
    # *** КОНЕЦ ВАШЕГО КОДА ***


if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='gda_pred_1')

    main(train_path='ds2_train.csv',
        valid_path='ds2_valid.csv',
        save_path='gda_pred_2')
