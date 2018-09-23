# 精度を向上させるために
# 1. データ量を増やす（水増し）
# 2. ハイパーパラメータ（学習率の増減）・アルゴリズムの調整
# 3. モデルの見直し（層、セル数）

from keras.models import Sequential
#from keras.layers import Convolution2D, MaxPooling2D #old keras
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten,Dense
from keras.utils import np_utils
import keras
import numpy as np

classes = ["monkey", "boar", "crow"] # label:0,1,2に対応する
num_classes = len(classes)
image_size = 50


def main():
	X_train, X_test, y_train, y_test = np.load("./animal_aug.npy")
	X_train = X_train.astype("float") / 256 # 0-1の値のほうが計算がはやい
	X_test = X_test.astype('float') / 256
	y_train = np_utils.to_categorical(y_train, num_classes)
	y_test = np_utils.to_categorical(y_test, num_classes)

	# modelのトレーニング
	model = model_train(X_train, y_train)
	# modelの評価
	model_eval(model, X_test, y_test)

def model_train(X, y):
	# モデル作成
	model = Sequential()
	# NNの層追加
	# 32個の3*3のフィルター、畳込みの結果が同じサイズになるように
	# padding : ピクセルを左右に足す
	# input_shape : 入力データの形状
	# 				X_train.shape : (450,50,50,3) 450個の50*50*3のデータ
	# 								なので、[1:]として、形状のみ読み出す
	model.add(Conv2D(32,(3,3), padding='same', input_shape=X.shape[1:]))
	model.add(Activation('relu')) # 正の数だけ通す活性化関数
	model.add(Conv2D(32,(3,3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2))) # 1番大きい値を取り出す。特徴を際立たせる
	model.add(Dropout(0.25)) # データの偏りをだす

	model.add(Conv2D(64,(3,3), padding='same'))
	model.add(Activation('relu'))
	model.add(Conv2D(64,(3,3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2))) # 1番大きい値を取り出す。特徴を際立たせる
	model.add(Dropout(0.25)) # データの偏りをだす

	model.add(Flatten()) # データを一列に並べる
	model.add(Dense(512)) # 全結合
	model.add(Activation('relu'))
	model.add(Dropout(0.5)) # データを半分捨てる
	model.add(Dense(3)) # 最後の出力層のノードは3つ
	model.add(Activation('softmax'))

	# 最適化処理 (rmsprop)
	#	decay : 学習効率を下げる
	opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

	# 損失関数 (categroical_crossentropy:正解と推定値の誤差を小さくする)
	model.compile(loss='categorical_crossentropy',
						optimizer=opt, metrics=['accuracy'])

	# トレーニング
	# batch_size : 1回のトレーニングで読み込む枚数
	# nb_epoch(Number Of epoch) : 何回繰り返すか
	model.fit(X, y, batch_size=32, epochs=50)

	# モデルの保存
	model.save('./animal_cnn_aug.h5')

	return model


# X: テスト
# y: テスト
def model_eval(model, X, y):
	scores = model.evaluate(X, y, verbose=1)
	print('Test Loss', scores[0])
	print('Test Accuracy: ', scores[1])

# このプログラムが直接pythonから呼ばれた場合だけmainを実行する
if __name__ == "__main__":
	main()
