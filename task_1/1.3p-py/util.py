import matplotlib.pyplot as plt
import numpy as np


def add_intercept(x):
    """Добавить столбец с коэффициентами для свободных членов.

    Аргументы:
        x: 2D NumPy array.

    Возвращаемое значение:
        Новая матрица, являющаяся конкатенацией столбца из единиц и матрицы x.
    """
    new_x = np.zeros((x.shape[0], x.shape[1] + 1), dtype=x.dtype)
    new_x[:, 0] = 1
    new_x[:, 1:] = x

    return new_x


def load_dataset(csv_path, label_col='y', add_intercept=False):
    """Загрузить датасет из CSV файла.

    Аргументы:
         csv_path: Пусть к CSV файлу с датасетом.
         label_col: Имя столбца, содержащего классы (должно быть 'y' или 'l').
         add_intercept: Добавить столбец из 1 к матрице x.

    Возвращаемое значение:
        xs: Numpy array со входными значениями x.
        ys: Numpy array с выходными значениями y.
    """

    def add_intercept_fn(x):
        global add_intercept
        return add_intercept(x)

    # Validate label_col argument
    allowed_label_cols = ('y', 't')
    if label_col not in allowed_label_cols:
        raise ValueError('Invalid label_col: {} (expected {})'
                         .format(label_col, allowed_label_cols))

    # Load headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    l_cols = [i for i in range(len(headers)) if headers[i] == label_col]
    inputs = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols)
    labels = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=l_cols)

    if inputs.ndim == 1:
        inputs = np.expand_dims(inputs, -1)

    if add_intercept:
        inputs = add_intercept_fn(inputs)

    return inputs, labels


def plot(x, y, theta, save_path, correction=1.0):
    """Построить график датасета и решающей границы регрессии Пуассона.

    Args:
        x: Матрица обучающих примеров, по одному на строке.
        y: Вектор классов из {0, 1}.
        theta: Вектор параметров регрессионной модели.
        save_path: Имя файла для сохранения графика.
        correction: Коррекционный фактор.
    """
    # Построить график датасета
    plt.figure()
    plt.plot(x[y == 1, -2], x[y == 1, -1], 'bx', linewidth=2)
    plt.plot(x[y == 0, -2], x[y == 0, -1], 'go', linewidth=2)

    # Построить график решающей границы (найденной в результате решения уравнения theta^T x = 0)
    x1 = np.arange(min(x[:, -2]), max(x[:, -2]), 0.01)
    x2 = -(theta[0] / theta[2] + theta[1] / theta[2] * x1
           + np.log((2 - correction) / correction) / theta[2])
    plt.plot(x1, x2, c='red', linewidth=2)
    plt.xlim(x[:, -2].min()-.1, x[:, -2].max()+.1)
    plt.ylim(x[:, -1].min()-.1, x[:, -1].max()+.1)

    # Добавить подписи и сохранить на диск.
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.savefig(save_path)
