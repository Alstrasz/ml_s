import numpy as np
import util


def main(train_path, valid_path, save_path):
    """Задача: Логистическая регрессия с обучением методом Ньютона.

    Аргументы:
        train_path: Путь к CSV-файлу, содержащему обучающую выборку.
        valid_path: Путь к CSV-файлу, содержащему валидационную выборку.
        save_path: Путь к файлу, в котором надо сохранить прогнозы.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)

    # *** НАЧАЛО ВАШЕГО КОДА ***
    # Обучите логистическую регрессию
    logreg = LogisticRegression(theta_0=np.zeros(len(x_train[0])))
    predicted = logreg.fit(x_train, y_train)
    util.plot(x_valid, y_valid, logreg.theta, save_path)
    # Нарисуйте решающую границу поверх валидационной выборки
    # Используйте np.savetxt, чтобы сохранить прогнозы на валидационной выборке в файле save_path
    # *** КОНЕЦ ВАШЕГО КОДА ***


class LogisticRegression:
    """Логистическая регрессия с методом Ньютона в качестве решателя.

    Пример использования:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, step_size=0.01, max_iter=10000, eps=1e-5,
                 theta_0=None, verbose=True, alpha=1):
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
        self.alpha = alpha

    def fit(self, x, y):
        """Выполнить метод Ньюьтона для минимизации J(theta) логистической регрессии.

        Аргументы:
            x: Входные вектора обучающей выборки. Размерность (n_examples, dim).
            y: Метки обучающей выборки. Размерность (n_examples,).
        """
        # *** НАЧАЛО ВАШЕГО КОДА ***
        for _ in range(self.max_iter):
            old = self.theta.copy()
            h = self.predict(x)
            d1 = -1.0/len(x) * x.T @ (y - h)
            d2 = 1.0/len(x) * (x.T @ h) * (x.T @ (1 - h))
            self.theta = self.theta - d1 / d2
            delta = np.sqrt(np.sum(((self.theta - old) * (self.theta - old ))))
            # print(delta)

            if delta < self.eps:
                break
        # *** КОНЕЦ ВАШЕГО КОДА ***

    def predict(self, x):
        """Сделать прогноз на новых входных данных x.

        Аргументы:
            x: Входные данные. Размерность (n_examples, dim).

        Возвращаемое значение:
            Прогноз размерности (n_examples,).
        """
        # *** НАЧАЛО ВАШЕГО КОДА ***
        return 1. / (1. + np.exp(-(self.theta @ x.T))) * self.alpha
        # *** КОНЕЦ ВАШЕГО КОДА ***


if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='logreg_pred_1')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='logreg_pred_2')
