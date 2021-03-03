import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

imdb = keras.datasets.imdb #영화 리뷰 텍스트 데이터

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=1000)
# 레이블 데이터는 0(부정), 1(긍정) 으로 표시됩니다.

print(" 훈련 데이터 : {}, 레이블 : {} ".format(len(train_data), len(train_labels)))

print(train_data[0]) # data 중에서 첫번째 리뷰 확인
print(len(train_data[0]), len(train_data[1]))

# 정수를 단어로 다시 변환하기

word_index = imdb.get_word_index()

word_index = {k: (v+3) for k,v in word_index.items()} # v+3을 하는 이유는 값이 0-3 까지 있다면 겹치면 안되서 +3 시행
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])  #문자열로 바꾸는거, get(x, '디폴트값') 왜 디폴트값에 '?'를 사용하는걸가? 아마 text에 존재하지 않는 값은 ?이거로 대체 하라는 말인듯!

print(decode_review(train_data[0]))

# 데이터 준비( pad_sequences : 길이가 같지 않고 적거나 많을 때 길이를 맞춰야하고 2차원,3차원 으로 만들때 사용)

train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],  # 디폴트 값은 PAD로 채우기
                                                        padding='post',
                                                        maxlen=256)
test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',   # 최대길이를 초과할 경우 뒷쪽을 자른다.
                                                       maxlen=256)    # 최대길이는 256
print(len(train_data[0]), len(train_data[1]))
print(train_data[0])

# 모델 구성

vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16, input_shape=(None, ))) #add() 메서드를 통해 sequential 모델을 점진적으로 작성 가능, Embedding : 양의 정수가 주어지면 그걸 vector로 바꾸는 과정

#input_shape = (batch_size,input_length)

model.add(keras.layers.GlobalAveragePooling1D()) #GlobalAveragePooling1D : sequence 차원에 대해 평균을 계산하여 각 샘플에 대해 고정된 길이의 출력 벡터를 반환.(길이가 다를때 사용)
model.add(keras.layers.Dense(16, activation='relu'))  #relu : rectifier함수로 은닉층에 주로 쓰임
model.add(keras.layers.Dense(1, activation='sigmoid')) #sigmoid : 이진분류문제에서 출력 층에 주로 사용 0,1로 출력

model.summary()

# 모델을 훈련할떄는 loss function 과 optimizer 가 필요. 손실함수에서 mean_squared_error를 사용할수도 있고 binary_crossentropy를 사용 할 수도 있다.
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#검증 세트 만들기
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

#모델 훈련
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data = (x_val, y_val),
                    verbose=1) #verbose : 학습의 진행 상황을 보여줄것인지 결정 ( 값 = 0이면 아무것도 출력하지 않는다.
# 값이 1이면 훈련의 진행도를 보여주는 진행 막대를 보여주고, 2 이면 미니배치마다 손실 정보를 출력합니다.)

#모델 평가
results = model.evaluate(test_data, test_labels, verbose=2)

print(results)

# 정확도와 손실 그래프 그리기

history_dict = history.history
history_dict.keys()

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1) #이거 식 왜이런거지..?

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label="Validation loss")
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend() #legend(키, 라벨) 괄호 안에 아무것도 안적었으면 자동으로 설정된다.

plt.show()

plt.clf() #기존 그래프 초기화

plt.plot(epochs,acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# 결론 : 훈련 손실은 에포크마다 감소하고 훈련 정확도는 ㄷ증가합니다.(경사하강법을 사용할때 나타나는 현상)
# 에포크 20부터 그래프가 평이한것은 최적점에 도달했다는 말이고 과대적합때문에 일어나는 현상( 과대적합을 막기위해 에포크 20부터는 훈련을 멈출수 있다. 나중에 callback 참고