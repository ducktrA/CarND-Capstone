
import os
import numpy as np
import cv2
import glob
import io

import tensorflow as tf
from PIL import Image
from object_detection.utils import dataset_util

SAMPLES_PER_TAG = 200
metadata = []

scenery = './scenery/*'
annotated_images = './annotated_images/'

tags = {'red', 'green', 'yellow', 'red-green'}
labelmap = dict(zip(tags, range(1, len(tags)+1)))

def create_yaml_entry(tag, x_width, xmin, y_height, ymin, filename):
    anno = '''- annotations:
  - {class: %s, x_width: %d, xmin: %d, y_height: %d, ymin: %d}
  class: image
  filename: %s
''' % (tag, x_width, xmin, y_height, ymin, filename)
    return anno

def create_csv_entry(tag, labelmap, x_width, xmin, y_height, ymin, filename):
    anno = '''%s, %d, %d, %d, %d, %d, %s
''' % (tag, labelmap[tag], x_width, xmin, y_height, ymin, filename)
    return anno

def create_label_map(labelmap):
    with open("labelmap.pbtxt", "w") as lmpbtxt:
        for k,v in labelmap.items():
            t = '''item {
  id: %d
  name: \'%s\'
}
''' % (v, k)

            lmpbtxt.write(t)

def create_tf_record(tag, labelmap, x_width, xmin, y_height, ymin, fname):
    global annotated_images
    path = annotated_images

    with tf.gfile.GFile(os.path.join(path, '{}'.format(fname)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = fname.encode()
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    xmins.append(xmin / width)
    xmaxs.append((xmin+x_width) / width)
    ymins.append(ymin / height)
    ymaxs.append((ymin+y_height) / height)
    classes_text.append(tag.encode('utf8'))
    classes.append(labelmap[tag])

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def simple_augment(tags, labelmap, scenery, metadata, train_or_test, amount):
    directory = "./positives/*/{}/*".format("red")

    writer = tf.python_io.TFRecordWriter("{}.record".format(train_or_test))

    S = []
    scenery_files = glob.glob(scenery)
    print("found %d files in scenery directory %s" % (len(scenery_files), scenery))

    for s in scenery_files:
        img = cv2.imread(s)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (800,600), interpolation=cv2.INTER_CUBIC)
        S.append(img)
        
    S = np.array(S)

    # Read X Vector

    for tag in tags:
        directory = "./positives/*/{}/*".format(tag)

        rois_images = glob.glob(directory)
        
        R = []
        
        for r in rois_images:
            img = cv2.imread(r)
            R.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        R = np.array(R)
        
        for i in range(SAMPLES_PER_TAG):
            r_rand = np.random.randint(0, len(R))
            s_rand = np.random.randint(0, len(S))

            roi = R[r_rand]
            scenery = S[s_rand]
        
            scale_down = np.random.randint(10, 100) / 100.
            roi = cv2.resize(roi, (0,0), fx=scale_down, fy=scale_down)
        
            annotated_image = np.copy(scenery)
            
            x_width = roi.shape[1]
            xmin = np.random.randint(10, scenery.shape[1] - x_width - 10)
            y_height = roi.shape[0]
            ymin = np.random.randint(10, scenery.shape[0] - y_height - 10)
            
            filename = "%s_%d.jpg" % (tag, i)
            
            annotated_image[ymin:ymin+y_height,xmin:xmin+x_width ] = roi
            
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(annotated_images + filename, annotated_image)
            
            #yaml = create_yaml_entry(tag, x_width, xmin, y_height, ymin, filename)
            csv = create_csv_entry(tag, labelmap, x_width, xmin, y_height, ymin, filename)
            tfrec = create_tf_record(tag, labelmap, x_width, xmin, y_height, ymin, filename)

            writer.write(tfrec.SerializeToString())
            metadata.append(csv)


    writer.close()

simple_augment(tags, labelmap, scenery, metadata, "train", 200)

create_label_map(labelmap)

with open("./annotation.csv", "w") as metadata_file:
    for line in metadata:
        metadata_file.write(line)


