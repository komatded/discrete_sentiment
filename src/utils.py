import numpy as np
from matplotlib import pyplot as plt


def pad_text(texts, emb_size, text_size):
    pad = np.zeros(emb_size)
    cutted_batch = [sen[:text_size] for sen in texts]
    cutted_batch = [[pad] * (text_size - len(tokens)) + list(tokens) for tokens in cutted_batch]
    return np.asarray(cutted_batch)


def visualize_result(data):
    words = [i['word'] for i in data['impact']]
    score = np.array([i['score'] for i in data['impact']])
    proba = np.array([i['probability'] for i in data['impact']])
    str_label = 'Sentence: ' + ' '.join(words)
    str_label += '\npos: {:.2f}, neu: {:.2f}, neg: {:.2f}'.format(*data['sentiment'])
    fig, (ax0, ax1) = plt.subplots(2)
    fig.suptitle(str_label, fontsize=7)
    ax0.matshow([score])
    ax0.set_xticks(range(len(words)))
    ax0.set_xticklabels(words, rotation=45)
    ax1.plot(proba)
    ax1.set_xticks(range(len(words)))
    ax1.set_xticklabels(words, rotation=45)
    plt.show()


if __name__ == '__main__':
    data = {'sentiment': [0.26938736, 0.72310555, 0.00750709],
            'impact': [{'word': 'это', 'score': 34.15327, 'probability': 0.092271306},
                       {'word': 'хороший', 'score': 77.71629, 'probability': 0.20996477},
                       {'word': 'проверочный', 'score': 40.296574, 'probability': 0.10886856},
                       {'word': 'текст', 'score': 24.762808, 'probability': 0.06690125},
                       {'word': ',', 'score': 24.722557, 'probability': 0.06679251},
                       {'word': 'потому', 'score': 29.53238, 'probability': 0.07978712},
                       {'word': 'что', 'score': 26.376385, 'probability': 0.07126062},
                       {'word': 'он', 'score': 23.089577, 'probability': 0.062380712},
                       {'word': 'точно', 'score': 25.652695, 'probability': 0.06930544},
                       {'word': 'отражает', 'score': 29.415634, 'probability': 0.079471715},
                       {'word': 'суть', 'score': 22.657728, 'probability': 0.061213993},
                       {'word': 'поставленной', 'score': 11.763776, 'probability': 0.03178199},
                       {'word': 'задачи', 'score': 0.0, 'probability': 0.0}]}
    visualize_result(data)
