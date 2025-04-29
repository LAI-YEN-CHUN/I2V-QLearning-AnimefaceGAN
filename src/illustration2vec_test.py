import os
import time

from PIL import Image

import illustration2vec.i2v as i2v

# Set the file paths
caffemodel_path = './src/illustration2vec/illust2vec_tag_ver200.caffemodel'
tag_list_path = './src/illustration2vec/tag_list.json'
test_image_path = './src/illustration2vec/images/miku.jpg'

# Check if the files exist
if not os.path.exists(caffemodel_path):
    raise FileNotFoundError(f'Model file not found: {caffemodel_path}')
if not os.path.exists(tag_list_path):
    raise FileNotFoundError(f'Tag list file not found: {tag_list_path}')
if not os.path.exists(test_image_path):
    raise FileNotFoundError(f'Test image not found: {test_image_path}')

# Create the Illust2Vec instance
print('Loading Illust2Vec model...')
start_time = time.time()
illust2vec = i2v.make_i2v_with_chainer(caffemodel_path, tag_list_path)
load_time = time.time() - start_time
print(f'Illust2Vec model loaded in {load_time:.4f} seconds')

# Load the test image
img = Image.open(test_image_path)
print(f'Loaded test image: {test_image_path}')
print(f'Image size: {img.size}')

# Measure time for plausible tags estimation
print('\nEstimating plausible tags...')
start_time = time.time()
plausible_tags = illust2vec.estimate_plausible_tags([img], threshold=0.5)
plausible_time = time.time() - start_time
print(f'Plausible tags (threshold=0.5): {plausible_tags}')
print(f'Time to estimate plausible tags: {plausible_time:.4f} seconds')

# Measure time for specific tags estimation
print('\n\nEstimating specific tags...')
start_time = time.time()
specific_tags = illust2vec.estimate_specific_tags([img], ['1girl', 'blue eyes', 'safe'])
specific_time = time.time() - start_time
print(f'Specific tags: {specific_tags}')
print(f'Time to estimate specific tags: {specific_time:.4f} seconds')
