import numpy as np
import util
import sys
from logreg import LogisticRegression

# Символ-маска для названий файлов, в которых будут сохраняться результаты решения подзадач.
WILDCARD = 'X'


def main(train_path, valid_path, test_path, save_path):
    """Логистическая регрессия, когда известны метки только для подмножества положительных примеров.

    Выполняется:
        1. для t-меток,
        2. для y-меток,
        3. для y-меток с alpha-коррекцией.

    Аргументы:
        train_path: Путь к CSV-файлу, содержащему обучающую выборку.
        valid_path: Путь к CSV-файлу, содержащему валидационную выборку.
        test_path: Путь к CSV-файлу, содержащему тестовую выборку.
        save_path: Шаблон пути к файлу, в котором надо сохранить прогнозы.
    """
    output_path_true = save_path.replace(WILDCARD, 'true')
    output_path_naive = save_path.replace(WILDCARD, 'naive')
    output_path_adjusted = save_path.replace(WILDCARD, 'adjusted')


    # *** НАЧАЛО ВАШЕГО КОДА ***
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    _, t_train = util.load_dataset(train_path, label_col='t', add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)
    _, t_test = util.load_dataset(test_path, label_col='t', add_intercept=True)
    print(t_train)
    # Подзадача 1: Обучите и протестируйте на истинных метках.
    logreg_t = LogisticRegression(theta_0=np.zeros(len(x_train[0])))
    predicted = logreg_t.fit(x_train, t_train)
    util.plot(x_test, t_test, logreg_t.theta, save_path+'logreg_t')
    print(logreg_t.theta)
    # Сохраните результат в файле output_path_true с помощью np.savetxt()
    #np.savetxt(save_path + '.txt', predicted)
    # Подзадача 2: Обучите на y-метках и протестируйте на истинных метках
    logreg_y = LogisticRegression(theta_0=np.zeros(len(x_train[0])))
    predicted = logreg_y.fit(x_train, y_train)
    print(logreg_y.theta)
    util.plot(x_test, t_test, logreg_y.theta, save_path + 'logreg_y')
    # Сохраните результат в файле output_path_naive с помощью np.savetxt()
    #np.savetxt(save_path + '.txt', predicted)
    # Подзадача 6: Примение корректирующий множитель, вычисленный на валидационной
	#			   выборке и протестируйте на истинных метках
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    acc = 0
    count = 0
    for i in range(len(y_valid)):
        if y_valid[i] == 1:
            acc += logreg_y.predict(x_valid[i])
            count += 1
    alpha = acc / count
    print('alpha', alpha, acc, count)

    #logreg_alpha = LogisticRegression(theta_0=np.zeros(len(x_train[0])), alpha=alpha)
    #predicted = logreg_y.fit(x_train, y_train)
    #print('theta alpha', logreg_alpha.theta)
    #util.plot(x_test, t_test, logreg_alpha.theta, save_path + 'logreg_alpha')
    util.plot(x_test, t_test, logreg_y.theta / alpha, save_path + 'logreg_alpha')
    # Сохраните результат в файле output_path_adjusted с помощью np.savetxt()
    # *** КОНЕЦ ВАШЕГО КОДА ***

if __name__ == '__main__':
    main(train_path='train.csv',
        valid_path='valid.csv',
        test_path='test.csv',
        save_path='posonly_X_pred')
