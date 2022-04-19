import collections

import numpy as np
import re
import matplotlib as plt
import util
import svm


def get_words(message):
    """Возвращает список нормализованных слов, полученных из строки сообщения.

    Эта функция должна разбивать сообщение на слова, нормализовать их и возвращать
	получившийся список. Слова необходимо разбивать по пробелам. Под нормализацией
	понимается перевод всех букв в нижний регистр.

    Аргументы:
        message: Строка, содержащая SMS сообщение.

    Возвращаемое значение:
       Список нормализованных слов из текстового сообщения.
    """

    # *** НАЧАЛО ВАШЕГО КОДА ***
    return re.sub(r"[,!.?();\[\]{}]", "", message.lower()).split(' ')
    # *** КОНЕЦ ВАШЕГО КОДА ***


def create_dictionary(messages):
    """Создает словарь, отображающий слова в целые числа.

    Данная функция должна создать словарь всех слов из сообщений messages,
	в котором каждому слову будет соответствовать порядковый номер.
	Для разделения сообщений на слова используйте функцию get_words.

    Редкие слова чаще всего бывают бесполезными при построении классификаторов. Пожалуйста,
	добавляйте слово в словарь, только если оно встречается минимум в пяти сообщениях.

    Аргументы:
        messages: Список строк с SMS сообщениями.

    Возвращаемое значение:
        Питоновская структура dict, отображающая слова в целые числа.
		Нумерацию слов нужно начать с нуля.
    """

    # *** НАЧАЛО ВАШЕГО КОДА ***
    dictionary = dict()
    for message in messages:
        words = set(get_words(message))
        for word in words:
            if word in dictionary:
                dictionary[word] += 1
            else:
                dictionary[word] = 1
    keys = list(filter(lambda x: x[1] > 4, dictionary.items()))
    keys.sort(key=lambda x: -x[1])
    dictionary.clear()
    for i, key in enumerate(keys):
        dictionary[key[0]] = i
    return dictionary
    # *** КОНЕЦ ВАШЕГО КОДА ***


def transform_text(messages, word_dictionary):
    """Трансформирует список текстовых сообщений в массив numpy, пригодный для дальнейшего использования.

    Эта функция должна создать массив numpy, содержащий количество раз, которое каждое слово
	словаря появляется в каждом сообщении.
	Строки в результирующем массиве должны соответствовать сообщениям из массива messages,
	а столбцы - словам из словаря word_dictionary.
	
    Используйте предоставленный словарь, чтобы сопоставлять слова с индексами столбцов.
	Игнорируйте слова, которых нет в словаре. Используйте get_words, чтобы разбивать сообщения на слова.

    Аргументы:
        messages: Список строк, в котором каждая строка является одним SMS сообщением.
        word_dictionary: Питоновский словарь dict, отображающий слова в целые числа.

    Возвращаемое значение:
        Массив numpy с информацией о том, сколько раз каждое слово встречается в каждом сообщении.
        Элемент (i,j) массива равен количеству вхождений j-го слова (по словарю) в i-м сообщении.
    """
    # *** НАЧАЛО ВАШЕГО КОДА ***
    data = np.zeros((len(messages), len(word_dictionary)))
    for i, words in enumerate(map(get_words, messages)):
        for word in words:
            if word in word_dictionary:
                data[i][word_dictionary[word]] += 1
    # plt.pyplot.imsave('./test_arr.png', data)
    return data
    # *** КОНЕЦ ВАШЕГО КОДА ***


def fit_naive_bayes_model(matrix: np.array, labels: np.array):
    """Обучает наивный байесовский классификатор.

    Эта функция должна обучить наивную байесовскую модель по переданной обучающей выборке.

    Функция должна возвращать построенную модель.

    Вы можете использовать любой тип данных, который пожелаете, для возвращения модели.

    Аргументы:
        matrix: Массив array, содержащий количества вхождений слов в сообщения из обучающей выборки.
        labels: Бинарные метки (0 или 1) для обучающей выборки.

    Возвращаемое значение: Обученный классификатор, использующий мультиномиальную модель событий и сглаживание Лапласа.
    ЗАМЕЧАНИЕ: обученная модель должна содержать два поля: vocab @ (2, V) и class @ (2,):
    vocab[i, k] хранит значения параметров P(x[j]=k | y=i), i принадлежит {0, 1},
    class[i] хранит значения параметров P(y=i), i принадлежит {0, 1}.
    """
    # *** НАЧАЛО ВАШЕГО КОДА ***
    lab_false = labels == 0
    lab_true = labels == 1
    number_of_words = matrix.sum(axis=1)
    vocab = np.zeros((2, matrix.shape[1]))
    vocab[0, :] = matrix[lab_false, :].sum(axis=0)
    vocab[1, :] = matrix[lab_true, :].sum(axis=0)
    vocab += 1
    vocab[0] /= number_of_words[lab_false].sum() + matrix.shape[1]
    vocab[1] /= number_of_words[lab_true].sum() + matrix.shape[1]
    classes = np.zeros(2)
    classes[1] = labels.sum()
    classes[0] = len(labels) - classes[1]
    classes /= matrix.shape[0]

    return vocab, classes
    # *** КОНЕЦ ВАШЕГО КОДА ***


def predict_from_naive_bayes_model(model: np.array, matrix: np.array):
    """Используя функцию гипотезы наивного байесовского классификатора, выдает прогнозы для матрицы с данными matrix.

    Данная функция должна выдавать прогнозы, используя передаваемую ей из fit_naive_bayes_model модель классификатора.

    Аргументы:
        model: Обученная функцией fit_naive_bayes_model модель.
        matrix: Массив numpy, содержащий количества слов.

    Возвращаемое значение: Массив numpy с прогнозами наивного байесовского классификатора.
    """
    # *** НАЧАЛО ВАШЕГО КОДА ***
    vocab, classes = model
    px0 = vocab[0] * matrix
    px0[px0 != 0] = np.log(px0[px0 != 0])
    px0 = np.sum(px0, axis=1)
    px0 = np.exp(px0)
    px1 = vocab[1] * matrix
    px1[px1 != 0] = np.log(px1[px1 != 0])
    px1 = np.sum(px1, axis=1)
    px1 = np.exp(px1)
    ret = (px1 * classes[1]) / ((px1 * classes[1]) + (px0 * classes[0]))
    return np.round(ret)
    # *** КОНЕЦ ВАШЕГО КОДА ***


def get_top_five_naive_bayes_words(model, dictionary):
    """Определяет пять слов, наиболее характерных для спам сообщений.

    Используйте метрику, приведенную в теоретическом материале, чтобы понять, насколько данное
	конкретное слово хакактерно для того или иного класса.
    Верните список слов, отсортированный в порядке убывания "характерности".

    Аргументы:
        model: Обученная функцией fit_naive_bayes_model модель.
        dictionary: Питоновский словарь dict, отображающий слова в целые числа.

    Возвращаемое значение: список слов, отсортированный в порядке убывания "характерности".
    """
    # *** НАЧАЛО ВАШЕГО КОДА ***
    keys = list(filter(lambda x: x[1] > 4, dictionary.items()))
    keys.sort(key=lambda x: np.log(model[0][1][x[1]]) / np.log(model[0][0][x[1]]))
    return list(map(lambda x: x[0], keys))[0: 5]
    # *** КОНЕЦ ВАШЕГО КОДА ***


def compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, radius_to_consider):
    """Вычисляет оптимальный SVM радиус, используя предоставленную обучающую и валидационную выборки.

    Вы должны исследовать только те значения радиусов, которые переданы в списке radius_to_consider.
	Вы должны использовать точность классификации в качестве метрики для сравнения различных значений радиусов.

    Аргументы:
        train_matrix: Матрица с частотами слов для обучающей выборки.
        train_labels: Метки "спам" и "не спам" для обучающей выборки.
        val_matrix: Матрица с частотами слов для валидационной выборки.
        val_labels: Метки "спам" и "не спам" для валидационной выборки.
        radius_to_consider: Значения радиусов, среди которых необходимо искать оптимальное.

    Возвращаемое значение:
        Значение радиуса, при котором SVM достигает максимальной точности.
    """
    # *** НАЧАЛО ВАШЕГО КОДА ***
    metrics = []
    for rad in radius_to_consider:
        predicted = svm.train_and_predict_svm(train_matrix, train_labels, val_matrix, rad)
        metrics.append((rad, np.mean(predicted == val_labels)))
    metrics.sort(key=lambda x: -x[1])
    print(metrics)
    return metrics[0][0]
    # *** КОНЕЦ ВАШЕГО КОДА ***


def main():
    train_messages, train_labels = util.load_spam_dataset('spam_train.tsv')
    val_messages, val_labels = util.load_spam_dataset('spam_val.tsv')
    test_messages, test_labels = util.load_spam_dataset('spam_test.tsv')

    dictionary = create_dictionary(train_messages)

    print('Размер словаря: ', len(dictionary))

    util.write_json('spam_dictionary', dictionary)

    train_matrix = transform_text(train_messages, dictionary)

    np.savetxt('spam_sample_train_matrix', train_matrix[:100, :])

    val_matrix = transform_text(val_messages, dictionary)
    test_matrix = transform_text(test_messages, dictionary)

    naive_bayes_model = fit_naive_bayes_model(train_matrix, train_labels)

    naive_bayes_predictions = predict_from_naive_bayes_model(naive_bayes_model, test_matrix)

    np.savetxt('spam_naive_bayes_predictions', naive_bayes_predictions)

    naive_bayes_accuracy = np.mean(naive_bayes_predictions == test_labels)

    print('Наивный Байес показал точность {} на тестовой выборке'.format(naive_bayes_accuracy))

    top_5_words = get_top_five_naive_bayes_words(naive_bayes_model, dictionary)

    print('Пять наиболее характерных для спама слов: ', top_5_words)

    util.write_json('spam_top_indicative_words', top_5_words)

    optimal_radius = compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, [0.01, 0.1, 1, 10])

    util.write_json('spam_optimal_radius', optimal_radius)

    print('Оптимальное значение SVM-радиуса {}'.format(optimal_radius))

    svm_predictions = svm.train_and_predict_svm(train_matrix, train_labels, test_matrix, optimal_radius)

    svm_accuracy = np.mean(svm_predictions == test_labels)

    print('SVM модель имеет точность {} на тестовой выборке'.format(svm_accuracy, optimal_radius))


if __name__ == "__main__":
    main()
