from PIL import Image
import os, glob
import numpy as np
from sklearn import model_selection

classes = ["monkey", "boar", "crow"] # label:0,1,2に対応する
num_classes = len(classes)
image_size = 50
# 読み込んだデータを学習用、テスト用に分ける
num_testdata = 100

# 画像の読み込み
X_train = []
X_test = []
Y_train = []
Y_test = []

# １件ずつ取り出し、番号を振る
for index, classlabel in enumerate(classes):
	photos_dir = "./images/" + classlabel
	# ファイルをまとめて取得。パターン一致しているデータをまとめて取り出す
	files = glob.glob(photos_dir + '/*.jpg')
	for i, file in enumerate(files):
		if i >=200: break
		image = Image.open(file)
		image = image.convert("RGB") # 数値に変換
		image = image.resize((image_size, image_size))
		data = np.asarray(image) # イメージデータを数字の配列として渡す

		# num_testdata以下はテスト用にとっておき、それ以上なら水増しして追加する
		if i < num_testdata:
			X_test.append(data)
			Y_test.append(index)
		else:
			X_train.append(data)
			Y_train.append(index)

			# -20 - 20 まで 5度刻みで回転させる
			for angle in range(-20,20,5):
				img_r = image.rotate(angle)
				data = np.asarray(img_r) # pil形式 -> numpy形式に変換
				X_train.append(data)
				Y_train.append(index)

				# 反転
				img_trans = image.transpose(Image.FLIP_LEFT_RIGHT)
				data = np.asarray(img_trans)
				X_train.append(data)
				Y_train.append(index)

# ここまででpythonのlist型のデータになっている
# tensorflowが処理しやすいnumpyのarrayに変換している
#X = np.array(X)
#Y = np.array(Y)
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(Y_train)
y_test = np.array(Y_test)

#print(len(X))
# 教師データ、テストデータを分割（3:1で分割される）
# X_train, X_test, y_train, y_test = model_selection.train_test_split(X,Y)
xy = (X_train, X_test, y_train, y_test)
# 分割したデータを一つのファイルにまとめる
np.save("./animal_aug.npy", xy)
