from flickrapi import FlickrAPI
from urllib.request import urlretrieve
from pprint import pprint
import os, time, sys

# API情報
key = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
secret = "YYYYYYYY"
wait_time = 1

# 保存先ディレクトリ
animalname = sys.argv[1]
savedir = "./images/" + animalname

flickr = FlickrAPI(key, secret, format='parsed-json')
result = flickr.photos.search(
	text = animalname,
	per_page = 400,
	media = 'photos',
	sort = 'relevance',
	safe_search = 1,
	extras = 'url_q, licence'
)

photos = result['photos']

# 返り値を表示する
# pprint(photos)

# photosの中を一枚ずつ値を代入していく
for i, photo in enumerate(photos['photo']):
	url_q = photo['url_q']
	filepath = savedir + '/' + photo['id'] + '.jpg'
	if os.path.exists(filepath): continue
	urlretrieve(url_q, filepath)
	time.sleep(wait_time)
