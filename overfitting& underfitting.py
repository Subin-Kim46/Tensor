import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

NUM_WORDS = 1000

(train_data, train_labels), (test_data, test_labels)  = keras.datasets.imdb.load_data(num_words=NUM_WORDS)

def multi_hot_sequences(sequences, dimension) :
    results = np.zeros((len(sequences), dimension))
    for i, word_indices in enumerate(sequences) :
        results[i, word_indices] = 1.0 #results[i]의 특정 인텍스만 1로 설정합니다.
    return results

train_data = multi_hot_sequences(train_data, dimension= NUM_WORDS)
test_data = multi_hot_sequences(test_data, dimension= NUM_WORDS)

plt.plot(train_data[0])

#과대적합 예제
#기준 모델 만들기
baseline_model = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(NUM_WORDS, )),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

baseline_model.compile(optimizer='adam',
                       loss='binary_crossentropy',
                       metrics=['accuracy', 'binary_crossentropy'])

baseline_model.summary()

baseline_history = baseline_model.fit(train_data,
                                      train_labels,
                                      epochs=20,
                                      batch_size=512,
                                      validation_data=(test_data, test_labels),
                                      verbose=2)

#작은 모델 만들기
smaller_model = keras.Sequential([
    keras.layers.Dense(4, activation='relu', input_shape=(NUM_WORDS, )),
    keras.layers.Dense(4, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

smaller_model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy', 'binary_crossentropy'])
smaller_model.summary()

smaller_history = smaller_model.fit(train_data,
                                          train_labels,
                                          epochs=20,
                                          batch_size=512,
                                          validation_data=(test_data, test_labels),
                                          verbose=2)

#큰 모델 만들기
bigger_model = keras.models.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(NUM_WORDS, )),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
bigger_model.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy', 'binary_crossentropy'])
bigger_model.summary()
bigger_history= bigger_model.fit(train_data,
                                 train_labels,
                                 epochs=20,
                                 batch_size=512,
                                 validation_data=(test_data, test_labels),
                                 verbose=2)

#훈련 손실과 검증 손실 그래프 그리기
def plot_history(histories, key='binary_crossentropy'):
    plt.figure(figsize=(16,10))

    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_'+key],#이부분 질문
                       '_', label=name.title()+' Val')
        plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
                 label=name.title()+' Train')
        plt.xlabel('Epochs')
        plt.ylabel(key.replace('_',' ').title()) #replace로 인해 _ 를 빈칸으로 바꿈
        plt.legend() #자동방식을 이용한 레전드 작성성
        plt.xlim([0, max(history.epoch)])

    plot_history([('baseline', baseline_history),
                  ('smaller', smaller_history),
                  ('bigger', bigger_history)])

#과대적합을 방지하기위한 전략
#1) 가중치를 규제(weight regularization)
## L1에대한 파라미터를 0으로 만들어준다
## L2 규제는 가중치 파라미터에 제한은 두지만 0으로 만들지는 않는다.

l2_model = keras.models.Sequential([
    keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001),
                       activation='relu', input_shape=(NUM_WORDS, )),
    keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001),
                       activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

l2_model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy', 'binary_crossentropy'])

l2_model_history = l2_model.fit(train_data, train_labels,
                                epochs=20,
                                batch_size=512,
                                validation_data=(test_data, test_labels),
                                verbose=2)

plot_history([('baseline', baseline_history),
              ('l2', l2_model_history)])


#드롭아웃 추가하기
## 출력특성을 랜덤하게 끕니다.(0으로 만들어주는 것) 이전 층을 dropout 시켜준다.
dpt_model = keras.models.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(NUM_WORDS,)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation = 'sigmoid')
])

#compile 과정은 Artificial Neural Network의 토대로 Layer 정의/ Layer를 ANN로 묶어서 처리하는 과정
dpt_model.compile(optimizer='adam', #최적화
                  loss='binary_crossentropy', #예측값과 실제값 과의 차이
                  metrics=['accuracy', 'binary_crossentropy']) #평가지표를 accuracy를 추가했다는 것( 우리가 보는 output )

dpt_model_history = dpt_model.fit(train_data, train_labels,
                                  epochs=20,
                                  batch_size=512,
                                  validation_data=(test_data, test_labels),
                                  verbose=2)

plot_history([('baseline', baseline_history),
              ('dropout', dpt_model_history)])

plt.clf()