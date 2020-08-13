from deeppavlov import build_model, configs
from src.utils import pad_text
from src.model import *
from config import *
import numpy as np

TEXT_SIZE = 25

model = create_model(n_classes=3, text_size=TEXT_SIZE, embedding_size=2560, units_lstm=200, dense_size=100,
                     dropout_rate=0.2)
model.load_weights(MODEL_PATH)

attention_score = Model(inputs=model.inputs, outputs=model.get_layer('attention_score').output)
tokenizer, elmo = build_model(configs.embedder.elmo_ru_twitter, download=True)


def to_probability(x: np.array) -> np.array:
    x += abs(x.min())
    return x / x.sum()


def predict_sentiment(text: str):
    words = tokenizer([text])[0]
    x = pad_text(np.array(elmo([words])), emb_size=2560, text_size=TEXT_SIZE)
    label = model.predict(x)[0]
    prediction_score = attention_score.predict(x)[0][-len(words):]
    prediction_score_probability = to_probability(prediction_score)
    return {'sentiment': list(label.astype(float)), 'impact': list(map(lambda i: dict(zip(['word',
                                                                                           'score',
                                                                                           'probability'], i)),
                                                                   zip(words,
                                                                       list(prediction_score.astype(float)),
                                                                       list(prediction_score_probability.astype(float))
                                                                       ))
                                                                   )}
