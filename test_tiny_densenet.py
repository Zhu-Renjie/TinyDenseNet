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
    argparser.add_argument('-b', '--batch_size', type = int, default = 100,
                          help = "Batch size.")
    argparser.add_argument('-t', '--tfr_path', type = str, default = './data.tfr',
                          help = "The file path of TFRecord which to be evaluated.")
    argparser.add_argument('--height', type = int, default = 64,
                          help = "THe height of image to be evaluated.")
    argparser.add_argument('-w', '--width', type = int, default = 64,
                          help = "The width of image to be evaluated.")
    argparser.add_argument('-c', '--ckpt_path', type = str, default = "./models.weight-ckpt",
                          help = "The weights of trained model.")
    argparser.add_argument('--wnids', type = str, default = './wnids.txt')
    argparser.add_argument('-o', '--output', type = str, default = "./output.txt",
                          help = "evaluation results file.")
    args = argparser.parse_args()
    
    CLS = load_cls_label(args.wnids)
    
    if not os.path.exists(args.ckpt_path):
        print("{} not exist.".format(args.ckpt_path))
    tiny_densenet = TinyDenseNetLoader(args.ckpt_path, args.num_class, net = 'network2')

    eval_imgs, eval_lbls = tf_records.read_tfrecords_by_data(
        get_file_list(args.tfr_path),
        (args.height, args.width),
        3,
        batch_size = args.batch_size)
    eval_imgs = tf.cast(eval_imgs, tf.float32)
    num_examples = tf_records.get_number_examples(get_file_list(args.tfr_path))
    print("{} examples to be evaluated.".format(num_examples))
    
    fwl = open(args.output, 'w')
    with tf.Session() as sess:
        pred_right = 0.0
        pred_total = 0.0
        index = 0
        for it in tqdm(range(num_examples // args.batch_size)):
            eval_img_batch, eval_lbl_batch = sess.run([eval_imgs, eval_lbls])
            ps, pc = tiny_densenet.run(eval_img_batch)
            equal = (pc == eval_lbl_batch).astype(np.float32)
            pred_right += sum(equal)
            pred_total += len(equal)
            for each in pc:
                fwl.write("{}\t{}\n".format(index, CLS[each]))
                index += 1
        if pred_total > 0:
            val_accu = float(pred_right) / pred_total
            print("Accuracy: {}".format(val_accu))
        else:
            print("No images or model to be evaluated.") 
    fwl.close()
    
# FILE END.