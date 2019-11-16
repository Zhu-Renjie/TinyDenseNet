# -*- coding: utf-8 -*-

import tensorflow as tf 
from network.tiny_densenet import TinyDenseNet
import numpy as np 
import cv2 as cv 
from tfrecords import tf_records 
import os
from tqdm import tqdm 

class TinyDenseNetLoader():
    def __init__(self, ckpt_path, num_class = 4, net = 'network2'):
        print("Start init TinyDenseNet for TinyImageNet-200 classification.")
        self.graph = tf.Graph()
        sess_config = tf.ConfigProto(log_device_placement = False, 
                                     allow_soft_placement = True)
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph = self.graph, config = sess_config)
        with self.graph.as_default():
            self.images = tf.placeholder(dtype = tf.float32,
                          shape = [None, None, None, 3], name = "xin")
            tiny_densenet = TinyDenseNet(self.images, num_class, name = "tiny_densenet") # check name.
            if 'network1' == net:
                self.logits = tiny_densenet.network1(False)
            elif 'network2' == net:
                self.logits = tiny_densenet.network2(False)
            self.scores = tf.nn.softmax(self.logits)
            self.classes = tf.argmax(self.scores, 1)
            saver = tf.train.Saver()
            saver.restore(self.sess, ckpt_path)
        print("TinyDenseNet initialized.")
    
    def run(self, bgrs):
        """
        Args:
            bgrs: image readed by opencv.
        Return: return the class of input images.
        """
        bgrs = bgrs / 255.0 - 0.5
        scores, classes = self.sess.run([self.scores, self.classes], 
                                         feed_dict = {self.images: bgrs})
        return scores, classes

def get_file_list(root_dir, exts = 'tfr'):
    if os.path.isfile(root_dir) and root_dir.endswith('.tfr'):
        return [root_dir]
    raw_files = os.listdir(root_dir)
    tfrs = []
    for each in raw_files:
        if each.split(".")[-1].lower() in exts:
            tfrs.append(os.path.join(root_dir, each))
    return tfrs

def load_cls_label(wnids):
    with open(wnids, 'r') as fr:
        names = [each.strip() for each in fr.readlines()]
    return names

if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-n', '--num_class', type = int, default = 200,
                          help = "Total classes to be classified.")
    argparser.add_argument('-b', '--batch_size', type = int, default = 1,
                          help = "Batch size.")
    argparser.add_argument('-r', '--root_dir', type = str, default = '',
                          help = "The root directory of images which to be evaluated.")
    argparser.add_argument('-l', '--list', type = str, default = '',
                          help = "The file list to be evaluated.")
    argparser.add_argument('--height', type = int, default = 64,
                          help = "THe height of image to be evaluated.")
    argparser.add_argument('-w', '--width', type = int, default = 64,
                          help = "The width of image to be evaluated.")
    argparser.add_argument('-c', '--ckpt_path', type = str, default = "./models.weight-ckpt",
                          help = "The weights of trained model.")
    argparser.add_argument('--wnids', type = str, default = './wnids.txt')
    argparser.add_argument('-o', '--output', type = str, default = "./output.txt",
                          help = "evaluation results file.")
    argparser.add_argument('-i', '--image', default = "")
    args = argparser.parse_args()
    
    CLS = load_cls_label(args.wnids)
    # CLS = [0, 90, 180, 270]

    if not os.path.exists(args.ckpt_path):
        print("{} not exist.".format(args.ckpt_path))
    tiny_densenet = TinyDenseNetLoader(args.ckpt_path, args.num_class, net = 'network2')
    
    if args.image != "":
        lines = [args.image]
    elif args.list != "":
        with open(args.list, 'r') as fr:
            lines = fr.readlines()
            lines = [line.strip() for line in lines]
    elif args.root_dir != "":
        print("Pred images within {}\n".format(args.root_dir))
        files = os.listdir(args.root_dir)
        lines = []
        for each in files:
            if os.path.isfile(os.path.join(args.root_dir, each)) and each.endswith('JPEG'):
                lines.append(each)
    print("{} examples to be evaluated.".format(len(lines)))
    
    fwl = open(args.output, 'w')
    
    with tf.Session() as sess:
        pred_right = 0.0
        pred_total = 0.0
        pred_unknow = 0.0
        index = 0
        for line in tqdm(lines):
            items = line.split(' ', 1)
            path = os.path.join(args.root_dir, items[0])
            if 2 == len(items):
                label = int(items[1])
            else:
                label = -1 # unknow
            img = cv.imread(path)
            ps, pc = tiny_densenet.run(np.expand_dims(cv.resize(img, (args.width, args.height)), 0))
            if label != -1:
                pred_right += label == pc[0]
                pred_total += 1
            else:
                pred_unknow += 1
            if -1 == label:
                fwl.write("{} {}\n".format(items[0], CLS[pc[0]]))
            else:
                fwl.write("{}\t{}\t{}\t{}\t{}\n".format(index, path, pc[0], label, pc[0] == label))
            index += 1
        if pred_total > 0:
            val_accu = float(pred_right) / pred_total
            print("Accuracy: {}".format(val_accu))
            fwl.write("Accuracy: {}\n".format(val_accu))
        else:
            print("No images or model to be evaluated.")
            fwl.write("No images or model to be evaluated.\n")
        fwl.write("pred_unknow: {}\n".format(pred_unknow))
    fwl.close()
    
# FILE END.