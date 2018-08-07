import i2v
from PIL import Image
import json

		 
illust2vec = i2v.make_i2v_with_chainer(
    "illust2vec_tag_ver200.caffemodel", "tag_list.json")

def tovec(filename):
	img = Image.open(filename)
	return illust2vec.estimate_plausible_tags([img], threshold=0.5)[0]