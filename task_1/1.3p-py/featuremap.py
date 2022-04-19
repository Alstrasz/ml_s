import numpy.linalg

import util
import numpy as np
import matplotlib.pyplot as plt

np.seterr(all='raise')

factor = 2.0


class LinearModel(object):
    """Базовый класс линейной модели."""

    def __init__(self, theta=np.ones(2)):
        """
        Аргументы:
            theta: Вектор весов для модели.
        """
        self.theta = theta

    def fit(self, x, y):
        """Запустите решатель для подгонки модели. Вам нужно записать найденные
        с помощью нормальных уравнений значения в self.theta.

        Аргументы:
            X: Примеры обучающей выборки. Размерность (n_examples, dim).
            y: Метки обучающей выборки. Размерность (n_examples,).
        """
        # *** НАЧАЛО ВАШЕГО КОДА ***
        xx = x.T @ x
        xy = x.T @ y

        try:
            self.theta = np.linalg.solve(xx, xy)
        except:
            print('Error. Probably matrix is singular')
            self.theta = np.linalg.lstsq(xx, xy)[0]
        # *** КОНЕЦ ВАШЕГО КОДА ***

    def create_poly(self, k, X):
        """
        Генерирует полиномиальные признаки на основе значений x.
        Карта полиномов должна иметь все степени от 0 до k
        Выходом должен являться numpy массив размерности (n_examples, k+1)

        Аргументы:
            X: Примеры обучающей выборки. Размерность (n_examples, 2).
        """
        # *** НАЧАЛО ВАШЕГО КОДА ***
        ret = np.ones((X.shape[0], k + 1))
        ret[:, 0] = X[:, 0]
        ret[:, 1] = X[:, 1]
        for i in range(2, k + 1):
            ret[:, i] = ret[:, i - 1] * ret[:, 1]

        return ret
        # *** КОНЕЦ ВАШЕГО КОДА ***

    def create_sin(self, k, X):
        """
        Генерирует синусоидальную и полиномиальные признаки на основе значений x.
        Выходом должен являться numpy массив размерности (n_examples, k+2)

        Аргументы:
            X: Примеры обучающей выборки. Размерность (n_examples, 2).
        """
        # *** НАЧАЛО ВАШЕГО КОДА ***
        if k == 0:
            ret = np.ones((X.shape[0], 2))
            ret[:, 1] = np.sin(X[:, 1])
            return ret
        ret = np.ones((X.shape[0], k + 2))
        ret[:, 0: k+1] = self.create_poly(k, X)
        ret[:, k+1] = np.sin(X[:, 1])
        return ret

        # *** КОНЕЦ ВАШЕГО КОДА ***

    def predict(self, x):
        """
        Выдать прогноз на основе новых признаков x.
        Возвращает numpy массив с прогнозными значениями.

        Аргументы:
            X: Входные данные размерности (n_examples, dim).

        Выход:
            Выходные данные размерности (n_examples,).
        """
        # *** НАЧАЛО ВАШЕГО КОДА ***
        return x @ self.theta
        # *** КОНЕЦ ВАШЕГО КОДА ***


def run_exp(train_path, sine=False, ks=[1, 2, 3, 5, 10, 20], alphas=[1.e-2, 1.e-3, 1.e-4, 1.e-7, 1.e-15, 1.e-31], iter=10000, filename='plot.png'):
    train_x, train_y = util.load_dataset(train_path, add_intercept=True)
    plot_x = np.ones([1000, 2])
    plot_x[:, 1] = np.linspace(-factor * np.pi, factor * np.pi, 1000)
    plt.figure()
    plt.scatter(train_x[:, 1], train_y)

    # lm.fit(train_x, train_y)
    # plot_y = lm.predict(plot_x)

    for k, alpha in zip(ks, alphas):
        '''
        Наша цель - обучить модель и сделать прогноз для значений plot_x
        '''
        # *** НАЧАЛО ВАШЕГО КОДА ***

        if not sine:
            lm = LinearModel(np.zeros(k + 1))

            lm.fit(lm.create_poly(k, train_x), train_y)
            plot_y = lm.predict(lm.create_poly(k, plot_x))
        else:
            lm = LinearModel(np.zeros(k + 2))

            lm.fit(lm.create_sin(k, train_x), train_y)
            plot_y = lm.predict(lm.create_sin(k, plot_x))

        # *** КОНЕЦ ВАШЕГО КОДА ***
        '''
        Здесь plot_y - прогнозируемые значения линейной модели на значениях plot_x
        '''
        plt.ylim(-2, 2)
        plt.plot(plot_x[:, 1], plot_y, label='k=%d' % k)

    plt.legend()

    # plt.show()
    plt.savefig(filename)
    plt.clf()


def main(train_path, small_path, eval_path):
    '''
    Запустить все эксперименты
    '''
    # *** НАЧАЛО ВАШЕГО КОДА ***
    run_exp(train_path, sine=False, ks=[1, 2, 3, 5, 10, 20], alphas=[1.e-2, 1.e-3, 1.e-4, 1.e-7, 1.e-15, 1.e-31], iter=100000, filename='poly_plot.png')
    run_exp(train_path, sine=True, ks=[0, 1, 2, 3, 5, 10, 20], alphas=[1.e-1, 1.e-2, 1.e-3, 1.e-4, 1.e-7, 1.e-15, 1.e-31], iter=100000, filename='sin_plot.png')
    run_exp(small_path, sine=False, ks=[1, 2, 3, 5, 10, 20], alphas=[1.e-2, 1.e-3, 1.e-4, 1.e-7, 1.e-15, 1.e-31], iter=100000, filename='short_poly_plot.png')
    # *** КОНЕЦ ВАШЕГО КОДА ***


if __name__ == '__main__':
    main(train_path='train.csv',
         small_path='small.csv',
         eval_path='test.csv')
