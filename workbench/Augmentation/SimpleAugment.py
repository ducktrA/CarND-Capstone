
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


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
 
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def create_tf_record(tag, labelmap, x_width, xmin, y_height, ymin, fname):
    
    path = ""

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
    
    print("building {0} dataset with {1} samples per traffic sign.".format(train_or_test, amount))

    #directory = "./positives/*/{}/*".format("red")

    writer = tf.python_io.TFRecordWriter("{}.record".format(train_or_test))

    S = []
    scenery_files = glob.glob(scenery)
    print("found %d files in scenery directory %s" % (len(scenery_files), scenery))

    for s in scenery_files:
        img = cv2.imread(s)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (800,600), interpolation=cv2.INTER_CUBIC)
        S.append(img)
        S.append(cv2.flip(img, 1))
        
    S = np.array(S)

    # Read X Vector
    rois_d = {}

    for tag in tags:
        directory = "./positives/*/{}/*".format(tag)
        rois_images = glob.glob(directory)

        #print(rois_images)

        R = []

        for r in rois_images:
            img = cv2.imread(r)
            R.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        rois_d[tag] = np.array(R)

    for k,v in rois_d.items():
        print("tag: {0} elements: {1}".format(k, len(v)))


    for i in range(amount):      
        for tag in tags:            

            r_rand = np.random.randint(0, len(rois_d[tag]))
            s_rand = np.random.randint(0, len(S))
            gamma = float(np.random.randint(50, 300)) / 100.

            roi = rois_d[tag][r_rand]
            scenery = adjust_gamma(S[s_rand], gamma)
        
            scale_down = np.random.randint(30, 100) / 100.
            roi = cv2.resize(roi, (0,0), fx=scale_down, fy=scale_down)
        
            annotated_image = np.copy(scenery)
            
            x_width = roi.shape[1]
            xmin = np.random.randint(10, scenery.shape[1] - x_width - 10)
            y_height = roi.shape[0]
            ymin = np.random.randint(10, scenery.shape[0] - y_height - 10)
            
            filename = annotated_images + "%s_%d.jpg" % (tag, i)
            
            annotated_image[ymin:ymin+y_height,xmin:xmin+x_width ] = roi
            
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filename, annotated_image)
            
            #yaml = create_yaml_entry(tag, x_width, xmin, y_height, ymin, filename)
            csv = create_csv_entry(tag, labelmap, x_width, xmin, y_height, ymin, filename)
            tfrec = create_tf_record(tag, labelmap, x_width, xmin, y_height, ymin, filename)

            writer.write(tfrec.SerializeToString())
            metadata.append(csv)


    writer.close()



create_label_map(labelmap)

simple_augment(tags, labelmap, scenery, metadata, "train", 500)
simple_augment(tags, labelmap, scenery, metadata, "test", 50)

with open("./annotation.csv", "w") as metadata_file:
    for line in metadata:
        metadata_file.write(line)


