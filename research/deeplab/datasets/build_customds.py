# Lint as: python2, python3
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Modification by pdelgado, 21 - 06 - 2021
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ==============================================================================

"""Converts custom dataset data to TFRecord file format with Example protos.
The custom dataset is expected to have the following directory structure:
  + glomImData
     - build_customds.py (current working directiory).
     + train
       + images
       + masks
     + val
       + images
       + masks
     + test
       + images
       + masks
     + tfrecord
This script converts data into sharded data files and save at tfrecord folder.
The Example proto contains the following fields:
  image/encoded: encoded image content.
  image/filename: image filename.
  image/format: image file format.
  image/height: image height.
  image/width: image width.
  image/channels: image channels.
  image/segmentation/class/encoded: encoded semantic segmentation content.
  image/segmentation/class/format: semantic segmentation file format.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import glob
import math
import os.path
import re
import sys
import build_data
from six.moves import range
import tensorflow as tf
import tifffile as tiff

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('customds_root',
                           './glomImData',
                           'customds root folder.')

tf.app.flags.DEFINE_string(
    'output_dir',
    './tfrecord',
    'Path to save converted SSTable of TensorFlow examples.')


_NUM_SHARDS = 1


# A map from data type to filename postfix.
_DATATYPE_MAP = {
    'image': 'images',
    'mask': 'masks',
}

# A map from data type to data format.
_DATA_FORMAT_MAP = {
    'image': 'png',
    'label': 'png',
}

# Image file pattern.
_IMAGE_FILENAME_RE = re.compile('(.+)' + _POSTFIX_MAP['image'])


def _get_files(data, dataset_split):
  """Gets files for the specified data type and dataset split.
  Args:
    data: String, desired data ('image' or 'label').
    dataset_split: String, dataset split ('train_fine', 'val_fine', 'test_fine')
  Returns:
    A list of sorted file names or None when getting label for
      test set.
  """
  if dataset_split == 'train':
    split_dir = 'train'
  elif dataset_split == 'val':
    split_dir = 'val'
  elif dataset_split == 'test':
    split_dir = 'test'
  else:
    raise RuntimeError("Split {} is not supported".format(dataset_split))
    
  search_files = os.path.join(
      FLAGS.customds_root, split_dir,_DATATYPE_MAP[data], '*')
  filenames = glob.glob(search_files)
  return sorted(filenames)


def _convert_dataset(dataset_split):
  """Converts the specified dataset split to TFRecord format.
  Args:
    dataset_split: The dataset split (e.g., train_fine, val_fine).
  Raises:
    RuntimeError: If loaded image and label have different shape, or if the
      image file with specified postfix could not be found.
  """
  
  image_files = _get_files('image', dataset_split)
  label_files = _get_files('label', dataset_split)

  num_images = len(image_files)
  num_labels = len(label_files)
  num_per_shard = int(math.ceil(num_images / _NUM_SHARDS))

  if num_images != num_labels:
    raise RuntimeError("The number of images and labels doesn't match: {} {}".format(num_images, num_labels))

  image_reader = build_data.ImageReader('png', channels=3)
  label_reader = build_data.ImageReader('png', channels=1)

  
  for shard_id in range(_NUM_SHARDS):
    shard_filename = '%s-%05d-of-%05d.tfrecord' % (
        dataset_split, shard_id, _NUM_SHARDS)
    output_filename = os.path.join(FLAGS.output_dir, shard_filename)
    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
      start_idx = shard_id * num_per_shard
      end_idx = min((shard_id + 1) * num_per_shard, num_images)
      for i in range(start_idx, end_idx):
        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
            i + 1, num_images, shard_id))
        sys.stdout.flush()
        # Read the image.
        image_data = tf.gfile.FastGFile(image_files[i], 'rb').read()
        height, width = image_reader.read_image_dims(image_data)
        # Read the semantic segmentation annotation.
        seg_data = tf.gfile.FastGFile(label_files[i], 'rb').read()
        seg_height, seg_width = label_reader.read_image_dims(seg_data)
        if height != seg_height or width != seg_width:
          raise RuntimeError('Shape mismatched between image and label.')
        # Convert to tf example.
        re_match = _IMAGE_FILENAME_RE.search(image_files[i])
        if re_match is None:
          raise RuntimeError('Invalid image filename: ' + image_files[i])
        filename = os.path.basename(re_match.group(1))
        example = build_data.image_seg_to_tfexample(
            image_data, filename, height, width, seg_data)
        tfrecord_writer.write(example.SerializeToString())
    sys.stdout.write('\n')
    sys.stdout.flush()


def main(unused_argv):
  # Only support converting 'train_fine', 'val_fine' and 'test_fine' sets for now.
  for dataset_split in ['train_fine', 'val_fine', 'test_fine']:
    _convert_dataset(dataset_split)


if __name__ == '__main__':
  tf.app.run()
