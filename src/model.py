from keras.layers import Input, Dropout, LSTM, Bidirectional, Dense, Lambda, Activation
from keras.layers import dot, concatenate
from keras.models import Model
from keras.regularizers import l2
from keras import backend as K


def f1(y_true, y_pred):

    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def attention_3d_block(hidden_states):
    hidden_size = int(hidden_states.shape[2])
    score_first_part = Dense(hidden_size, use_bias=False, name='attention_score_vec')(hidden_states)
    h_t = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,), name='last_hidden_state')(hidden_states)
    score = dot([score_first_part, h_t], [2, 1], name='attention_score')
    attention_weights = Activation('softmax', name='attention_weight')(score)
    context_vector = dot([hidden_states, attention_weights], [1, 1], name='context_vector')
    pre_activation = concatenate([context_vector, h_t], name='attention_output')
    attention_vector = Dense(hidden_size, use_bias=False, activation='tanh', name='attention_vector')(pre_activation)
    return attention_vector


def create_model(n_classes, text_size, embedding_size, units_lstm, dense_size, dropout_rate=0., coef_reg_den=0.):
    inp = Input(shape=(text_size, embedding_size))
    output = Bidirectional(LSTM(units_lstm, activation='tanh', return_sequences=True, dropout=dropout_rate))(inp)
    output = attention_3d_block(output)
    output = Dropout(rate=dropout_rate)(output)
    output = Dense(dense_size, activation='relu', kernel_regularizer=l2(coef_reg_den))(output)
    output = Dropout(rate=dropout_rate)(output)
    output = Dense(n_classes, activation='softmax', kernel_regularizer=l2(coef_reg_den))(output)
    model = Model(inputs=inp, outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', f1])
    return model
