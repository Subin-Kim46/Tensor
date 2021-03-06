import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

print("버젼 : ", tf.__version__)
print("즉시 실행 모드: ", tf.executing_eagerly())
print("허브 버전", hub.__version__)
print("GPU", "사용 가능" if tf.config.experimental.list_physical_devices("GPU")
      else "사용 불가능")

#데이터셋 분리
train_data, validation_data, test_data = tfds.load(
      name = "imdb_reviews",
      split=('train[:60%]','train[60%:]','test'),
      as_supervised=True)

#데이터 탐색(레이블 0(부정) 1(긍정))
train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
#iter에 의해 10의 값이 나올때까지 반복, next는 기본값으로 지정한만큼 끝나면 계속 우리가 설정한 기본값만 출력

print(train_examples_batch) # 영화 리뷰에 대한 글 10문장
print(train_labels_batch) # 긍정부정에 따라 표현

#모델 구성 ( 다시 찾아보기 )
embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
hub_layer = hub.KerasLayer(embedding, input_shape=[],   #input_shape = [batch_size]
                           dtype=tf.string, trainable=True) #tf.string : input tensor

print(hub_layer(train_examples_batch[:3]))

# 전체 모델 구성
model = Sequential()
model.add(hub_layer)
model.add(Dense(16, activation='rule')) #hiddne layer
model.add(Dense(1)) # output sigmoid 활성함수

model.summary()

# 손실함수와 옵티마이저
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits = True),
              metrics=['accuracy'])
# from_logits : 모델이 출력하는 output이 logit인지 아닌지 판단 , (통계학) Logistic Regression 의 의미
 #(딥러닝) normalize 되었냐의 여부(softmax 함수를 사용하면 False / sigmoid, linear 함수를 사용하면 True)

# 모델 훈련
history = model.fit(train_data.shuffle(10000).batch(512),
                    epochs=20,
                    validation_data=validation_data.batch(512),
                    verbose=1)

results = model.evaluate(test_data.batch(512), verbose=2)

for name, value in zip(model.metrics_names, results):
    print("%s: %.3f" % (name, value))
