from PIL import Image
import os, glob
import numpy as np
from sklearn import model_selection

classes = ["monkey", "boar", "crow"] # label:0,1,2に対応する
num_classes = len(classes)
image_size = 50

# 画像の読み込み
X = []
Y = []

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
		X.append(data) # 末尾に追加
		Y.append(index) # label情報

# ここまででpythonのlist型のデータになっている
# tensorflowが処理しやすいnumpyのarrayに変換している
X = np.array(X)
Y = np.array(Y)

#print(len(X))

# 教師データ、テストデータを分割（3:1で分割される）
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,Y)
xy = (X_train, X_test, y_train, y_test)
# 分割したデータを一つのファイルにまとめる
np.save("./animal.npy", xy)

