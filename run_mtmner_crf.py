from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import json
import time
import sys
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from my_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from my_bert.mner_modeling import (CONFIG_NAME, WEIGHTS_NAME,
                                              BertConfig, MSCMT)
from my_bert.optimization import BertAdam, warmup_linear
from my_bert.tokenization import BertTokenizer
from seqeval.metrics import classification_report
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

import resnet.resnet as resnet
from resnet.resnet_utils import myResnet

from torchvision import transforms
from PIL import Image

from sklearn.metrics import precision_recall_fscore_support

from ner_evaluate import evaluate_each_class
from ner_evaluate import evaluate

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)
from config import Args as unif_conf
from token_augmentation import HardInternalAugmentation


def image_process(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class MMInputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b, img_id, label=None, auxlabel=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.img_id = img_id
        self.label = label
        self.auxlabel = auxlabel


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class MMInputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, added_input_mask, segment_ids, img_feat, label_id, auxlabel_id,
                 char_input_ids, sent_token_aug, frcnn_feature, frcnn_mask, mrcnn_obj_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.added_input_mask = added_input_mask
        self.segment_ids = segment_ids
        self.img_feat = img_feat
        self.label_id = label_id
        self.auxlabel_id = auxlabel_id
        self.char_input_ids = char_input_ids
        self.sent_token_aug = sent_token_aug
        self.frcnn_feature = frcnn_feature
        self.frcnn_mask = frcnn_mask
        self.mrcnn_obj_ids = mrcnn_obj_ids


def readfile(filename):
    '''
    read file
    return format :
    [ ['EU', 'B-ORG'], ['rejects', 'O'], ['German', 'B-MISC'], ['call', 'O'], ['to', 'O'], ['boycott', 'O'], ['British', 'B-MISC'], ['lamb', 'O'], ['.', 'O'] ]
    '''
    f = open(filename)
    data = []
    sentence = []
    label= []
    for line in f:
        if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n":
            if len(sentence) > 0:
                data.append((sentence,label))
                sentence = []
                label = []
            continue
        splits = line.split(' ')
        sentence.append(splits[0])
        label.append(splits[-1][:-1])

    if len(sentence) >0:
        data.append((sentence,label))
        sentence = []
        label = []

    print("The number of samples: "+ str(len(data)))
    return data


def mmreadfile(filename):
    '''
    read file
    return format :
    [ ['EU', 'B-ORG'], ['rejects', 'O'], ['German', 'B-MISC'], ['call', 'O'], ['to', 'O'], ['boycott', 'O'], ['British', 'B-MISC'], ['lamb', 'O'], ['.', 'O'] ]
    '''
    f = open(filename)
    data = []
    imgs = []
    auxlabels = []
    sentence = []
    label= []
    auxlabel = []
    imgid = ''
    for line in f:
        if line.startswith('IMGID:'):
            imgid = line.strip().split('IMGID:')[1]+'.jpg'
            continue
        if line[0]=="\n":
            if len(sentence) > 0:
                data.append((sentence,label))
                imgs.append(imgid)
                auxlabels.append(auxlabel)
                sentence = []
                label = []
                imgid = ''
                auxlabel = []
            continue
        splits = line.split('\t')
        sentence.append(splits[0])
        cur_label = splits[-1][:-1]
        if cur_label == 'B-OTHER':
            cur_label = 'B-MISC'
        elif cur_label == 'I-OTHER':
            cur_label = 'I-MISC'
        label.append(cur_label)
        auxlabel.append(cur_label[0])

    if len(sentence) >0:
        data.append((sentence,label))
        imgs.append(imgid)
        auxlabels.append(auxlabel)
        sentence = []
        label = []
        auxlabel = []

    print("The number of samples: "+ str(len(data)))
    print("The number of images: "+ str(len(imgs)))
    return data, imgs, auxlabels


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        return readfile(input_file)

    def _read_mmtsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        return mmreadfile(input_file)


class MSCMT_NERProcessor(DataProcessor):
    """Processor for the CoNLL-2003 data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        data, imgs, auxlabels = self._read_mmtsv(os.path.join(data_dir, "train.txt"))
        return self._create_examples(data, imgs, auxlabels, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        data, imgs, auxlabels = self._read_mmtsv(os.path.join(data_dir, "valid.txt"))
        return self._create_examples(data, imgs, auxlabels, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        data, imgs, auxlabels = self._read_mmtsv(os.path.join(data_dir, "test.txt"))
        return self._create_examples(data, imgs, auxlabels, "test")

    def get_labels(self):
        return ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X", "[CLS]", "[SEP]"]

    def get_auxlabels(self):
        return ["O", "B", "I", "X", "[CLS]", "[SEP]"]

    def get_start_label_id(self):
        label_list = self.get_labels()
        label_map = {label: i for i, label in enumerate(label_list, 1)}
        return label_map['[CLS]']

    def get_stop_label_id(self):
        label_list = self.get_labels()
        label_map = {label: i for i, label in enumerate(label_list, 1)}
        return label_map['[SEP]']

    def _create_examples(self, lines, imgs, auxlabels, set_type):
        examples = []
        for i, (sentence, label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            text_b = None
            img_id = imgs[i]
            label = label
            auxlabel = auxlabels[i]
            examples.append(MMInputExample(guid=guid, text_a=text_a, text_b=text_b, img_id=img_id, label=label, auxlabel=auxlabel))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list,1)}
    
    features = []
    for (ex_index,example) in enumerate(examples):
        textlist = example.text_a.split(' ')
        labellist = example.label
        tokens = []
        labels = []
        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = labellist[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                else:
                    labels.append("X")
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        label_ids.append(label_map["[CLS]"])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            label_ids.append(label_map[labels[i]])
        ntokens.append("[SEP]")
        segment_ids.append(0)
        label_ids.append(label_map["[SEP]"])
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        
        if ex_index < 2:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s" % " ".join([str(x) for x in label_ids]))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_ids))
    return features

def rcnn_parser(feature_map, max_obj_num):
    rcnn_obj_num, rcnn_feature_dim = feature_map.shape
    if rcnn_obj_num < max_obj_num:
        pad_obj_num = max_obj_num - rcnn_obj_num
        real_rcnn_feature = torch.nn.functional.pad(feature_map, pad=[0, 0, 0, pad_obj_num], mode='constant', value=0)
        # we can also have
        # real_feature = torch.zeros([pad_obj_num, rcnn_feature_dim], dtype=torch.float32)
        # real_rcnn_feature = torch.cat([feature_map, real_feature], dim=0)
        rcnn_mask = [1] * rcnn_obj_num + [0] * pad_obj_num
    else:
        real_rcnn_feature = feature_map[:max_obj_num, :]
        rcnn_mask = [1] * max_obj_num

    assert real_rcnn_feature.shape[0] == max_obj_num
    return real_rcnn_feature, rcnn_mask

def convert_mm_examples_to_features(token_aug_emb_table, examples, label_list, auxlabel_list, max_seq_length, tokenizer, crop_size, path_img):
    """Loads a data file into a list of `InputBatch`s."""

    fast_rcnn_features = torch.load(unif_conf.img_feature_rcnn)
    mrcnn_object_ids_dict = torch.load(unif_conf.img_object_mrcnn)
    word2index = unif_conf.word_dict
    char2index = unif_conf.char_dict
    char_unk_idx = char2index['[UNK]']
    char_pad_idx = char2index['[PAD]']
    char_x_idx = char2index['[X]']
    char_xc_idx = char2index['[XC]']
    char_xs_idx = char2index['[XS]']
    max_wordlen = unif_conf.max_wordlen
    needed_object_num = unif_conf.obj_num

    label_map = {label: i for i, label in enumerate(label_list, 1)}
    auxlabel_map = {label: i for i, label in enumerate(auxlabel_list, 1)}

    features = []
    count = 0

    zero_aug = np.zeros([1,unif_conf.aug_dim], dtype=np.float32)[0]

    transform = transforms.Compose([
        transforms.RandomCrop(crop_size),  # args.crop_size, by default it is set to be 224
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    for (ex_index, example) in enumerate(examples):
        textlist = example.text_a.split(' ')
        labellist = example.label
        auxlabellist = example.auxlabel
        tokens = []
        char_sent = []
        labels = []
        auxlabels = []
        aug_sent = []
        for i, word in enumerate(textlist):
            # for char_cnn
            char_word = []
            for char in word.strip():
                char_word.append(char2index.get(char, char_unk_idx))
            char_padding_length = max_wordlen - len(char_word)
            char_word = char_word + [char_pad_idx]*char_padding_length
            char_word = char_word[:max_wordlen]

            aug_temp_scale = np.sqrt(3.0 / unif_conf.aug_dim)
            token_aug_emb = np.random.uniform(-aug_temp_scale, aug_temp_scale, [1, unif_conf.aug_dim])[0]
            try:
                token_aug_emb = token_aug_emb_table[word2index[word]]
            except Exception as e:
                print(e)
                print("There is the wrong key word: {}".format(word))
                time.sleep(2)

            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = labellist[i]
            auxlabel_1 = auxlabellist[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                    auxlabels.append(auxlabel_1)
                    char_sent.append(char_word)  # 对分割后的每个token都进行原词的char
                    aug_sent.append(token_aug_emb)
                else:
                    labels.append("X")
                    auxlabels.append("X")
                    char_sent.append([char_x_idx]*max_wordlen) #
                    aug_sent.append(zero_aug) # for token pad
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
            auxlabels = auxlabels[0:(max_seq_length - 2)]
            char_sent = char_sent[0:(max_seq_length - 2)]
            aug_sent = aug_sent[0:(max_seq_length - 2)]

        ntokens = []
        segment_ids = []
        label_ids = []
        auxlabel_ids = []
        ntokens.append("[CLS]")
        char_sent.insert(0, [char_xc_idx]*max_wordlen) # char for [CLS]
        aug_sent.insert(0, zero_aug) # aug for [CLS]
        segment_ids.append(0)
        label_ids.append(label_map["[CLS]"])
        auxlabel_ids.append(auxlabel_map["[CLS]"])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            label_ids.append(label_map[labels[i]])
            auxlabel_ids.append(auxlabel_map[auxlabels[i]])
        ntokens.append("[SEP]")
        char_sent.append([char_xs_idx]*max_wordlen) # char for [SEP]
        aug_sent.append(zero_aug) # aug for [SEP]
        segment_ids.append(0)
        label_ids.append(label_map["[SEP]"])
        auxlabel_ids.append(auxlabel_map["[SEP]"])
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        added_input_mask = [1] * (len(input_ids) + 49)  # 1 or 49 is for encoding regional image representations

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            added_input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            auxlabel_ids.append(0)
            char_sent.append([char_pad_idx]*max_wordlen)
            aug_sent.append(zero_aug) # for sent pad

        assert len(input_ids) == max_seq_length # 列表，元素是token在vocab中的id
        assert len(input_mask) == max_seq_length # 文本序列的有效输入mask
        assert len(segment_ids) == max_seq_length # 文本序列的第一个seg的标记
        assert len(label_ids) == max_seq_length # 有效输入序列的标签，包含cls和sep
        assert len(auxlabel_ids) == max_seq_length # 原始论文中的辅助标签
        assert len(char_sent) == max_seq_length # 列表的列表
        assert len(aug_sent) == max_seq_length # 每个元素的语义增强

        image_name = example.img_id
        image_path = os.path.join(path_img, image_name)
        
        if not os.path.exists(image_path):
            print(image_path)
        
        # this is for resnet
        if unif_conf.resize == 0:
            try:
                image = image_process(image_path, transform)
            except ValueError:
                image = Image.open(image_path).resize((224, 224)).convert('RGB')
                image = transform(image)
            except Exception as e:
                print(e)
                count += 1
                print('image has problem!: ', image_name, count)
                image_path_fail = os.path.join(path_img, '17_06_4705.jpg')
                image = image_process(image_path_fail, transform)
        elif unif_conf.resize == 1:
            try:
                image = image_process(image_path, transform)
            except Exception as e:
                print(e)
                count += 1
                print('image has problem!: ', image_name, count)
                image_path_fail = os.path.join(path_img, '17_06_4705.jpg')
                image = image_process(image_path_fail, transform)

        # this is for faster-rcnn
        rcnn_img_id = image_name.split('.jpg')[0]
        try:
            temp_rcnn_feature = fast_rcnn_features[int(rcnn_img_id)].contiguous()
        except:
            temp_rcnn_feature = torch.zeros(0, 2048)
        faster_rcnn_feature, faster_rcnn_mask = rcnn_parser(temp_rcnn_feature, needed_object_num)

        # this is for mrcnn object ids
        mrcnn_object_tensor = [0]*needed_object_num # 图片不存在或者检测不出来obj时，均适用
        try:
            # 说明存在检测结果
            mrcnn_object_ids = mrcnn_object_ids_dict[image_name]
            mrcnn_object_num = len(mrcnn_object_ids)
            if mrcnn_object_num < needed_object_num:
                mrcnn_object_tensor[:mrcnn_object_num] = mrcnn_object_ids
                # mrcnn_mask = [1]*mrcnn_object_num + [0]*(needed_object_num-mrcnn_object_num)
            else:
                mrcnn_object_tensor[:needed_object_num] = mrcnn_object_ids[:needed_object_num]
                # mrcnn_mask = [1]*needed_object_num
        except:
            print("There are some matter in image {}".format(image_name))
            time.sleep(3)

        if ex_index < 2:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s" % " ".join([str(x) for x in label_ids]))
            logger.info("auxlabel: %s" % " ".join([str(x) for x in auxlabel_ids]))

        features.append(
            MMInputFeatures(input_ids=input_ids, input_mask=input_mask, added_input_mask=added_input_mask,
                          segment_ids=segment_ids, img_feat=image, label_id=label_ids, auxlabel_id=auxlabel_ids, char_input_ids=char_sent,
                            sent_token_aug=aug_sent, frcnn_feature=faster_rcnn_feature, frcnn_mask=faster_rcnn_mask, mrcnn_obj_ids=mrcnn_object_tensor))

    print('the number of problematic samples: ' + str(count))

    return features


def macro_f1(y_true, y_pred):
    p_macro, r_macro, f_macro, support_macro \
      = precision_recall_fscore_support(y_true, y_pred, average='macro')
    f_macro = 2*p_macro*r_macro/(p_macro+r_macro)
    return p_macro, r_macro, f_macro


def main():
    # HA = HardInternalAugmentation(unif_conf, unif_conf.word_dict)
    # token_aug_emb_table = HA.hard_augmentation_embedding(unif_conf.aug_dim)
    token_aug_emb_table = np.load(unif_conf.token_aug_vec_addr)

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=12.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=32,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--mm_model', default='MSCMT', help='model name')  
    parser.add_argument('--layer_num1', type=int, default=1, help='number of txt2img layer')
    parser.add_argument('--layer_num2', type=int, default=1, help='number of img2txt layer')
    parser.add_argument('--layer_num3', type=int, default=1, help='number of txt2txt layer')
    parser.add_argument('--fine_tune_cnn', action='store_true', help='fine tune pre-trained CNN if True')
    parser.add_argument('--resnet_root', default='./resnet', help='path the pre-trained cnn models')
    parser.add_argument('--crop_size', type=int, default=224, help='crop size of image')
    parser.add_argument('--path_image', default='~/MultiModal_NER/data/twitter2015/ner_img', help='path to images')
    #parser.add_argument('--path_image', default='../pytorch-pretrained-BERT/twitter_subimages/', help='path to images')
    #parser.add_argument('--mm_model', default='TomBert', help='model name') #
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()

    if args.task_name == "twitter2017":
        args.path_image = "XXX/data/twitter2017/ner_img/"
    elif args.task_name == "twitter2015":
        args.path_image = "XXX/data/twitter2015/ner_img/"

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()
    
    processors = {
        "twitter2015": MSCMT_NERProcessor,
        "twitter2017": MSCMT_NERProcessor
        }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        print(device)
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))
    
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    print('batch_size: ', args.train_batch_size)

    '''
    if args.task_name == "twitter2015":
        # args.num_train_epochs = 24.0
        args.num_train_epochs = 75.0
    if args.task_name == "twitter2017":
        # args.num_train_epochs = 22.0
        args.num_train_epochs = 75.0
    '''
    args.num_train_epochs = unif_conf.epoch_num
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    # args.output_dir = args.output_dir + unif_conf.added_output_dir
    args.output_dir = args.output_dir + unif_conf.added_output_dir + '-LR-' + str(args.learning_rate) + '-EPOCH-' + "{0:g}".format(args.num_train_epochs)
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        shutil.rmtree(args.output_dir)
        # raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    label_list = processor.get_labels()
    auxlabel_list = processor.get_auxlabels()
    num_labels = len(label_list)+1 # label 0 corresponds to padding, label in label_list starts from 1
    auxnum_labels = len(auxlabel_list)+1 # label 0 corresponds to padding, label in label_list starts from 1

    start_label_id = processor.get_start_label_id()
    stop_label_id = processor.get_stop_label_id()

    #''' initialization of our conversion matrix, in our implementation, it is a 7*12 matrix initialized as follows:
    trans_matrix = np.zeros((auxnum_labels,num_labels), dtype=float)
    trans_matrix[0,0]=1 # pad to pad
    trans_matrix[1,1]=1 # O to O
    trans_matrix[2,2]=0.25 # B to B-MISC
    trans_matrix[2,4]=0.25 # B to B-PER
    trans_matrix[2,6]=0.25 # B to B-ORG
    trans_matrix[2,8]=0.25 # B to B-LOC
    trans_matrix[3,3]=0.25 # I to I-MISC
    trans_matrix[3,5]=0.25 # I to I-PER
    trans_matrix[3,7]=0.25 # I to I-ORG
    trans_matrix[3,9]=0.25 # I to I-LOC
    trans_matrix[4,10]=1   # X to X
    trans_matrix[5,11]=1   # [CLS] to [CLS]
    trans_matrix[6,12]=1   # [SEP] to [SEP]
    '''
    trans_matrix = np.zeros((num_labels, auxnum_labels), dtype=float)
    trans_matrix[0,0]=1 # pad to pad
    trans_matrix[1,1]=1
    trans_matrix[2,2]=1
    trans_matrix[4,2]=1
    trans_matrix[6,2]=1
    trans_matrix[8,2]=1
    trans_matrix[3,3]=1
    trans_matrix[5,3]=1
    trans_matrix[7,3]=1
    trans_matrix[9,3]=1
    trans_matrix[10,4]=1
    trans_matrix[11,5]=1
    trans_matrix[12,6]=1
    '''

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()
    
    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank))
    cache_dir = "XXX/.cache/torch/pytorch_pretrained_bert"
    print(cache_dir)
    if args.mm_model == 'MSCMT':
        model = MSCMT.from_pretrained(".cache/pytorch_pretrained_bert",
              cache_dir=cache_dir, layer_num1=args.layer_num1, layer_num2=args.layer_num2, layer_num3=args.layer_num3,
              num_labels = num_labels, auxnum_labels = auxnum_labels)
    else:
        print('please define your MNER Model')

    net = getattr(resnet, 'resnet152')()
    net.load_state_dict(torch.load(os.path.join(args.resnet_root, 'resnet152.pth')))
    encoder = myResnet(net, args.fine_tune_cnn, device)
    if args.fp16:
        model.half()
        encoder.half()
    print(device)
    model.to(device)
    encoder.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
        encoder = DDP(encoder)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)
        encoder = torch.nn.DataParallel(encoder)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0

    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
    output_encoder_file = os.path.join(args.output_dir, "pytorch_encoder.bin")

    if args.do_train:
        train_features = convert_mm_examples_to_features(token_aug_emb_table,
            train_examples, label_list, auxlabel_list, args.max_seq_length, tokenizer, args.crop_size, args.path_image)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_added_input_mask = torch.tensor([f.added_input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_img_feats = torch.stack([f.img_feat for f in train_features])
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        all_auxlabel_ids = torch.tensor([f.auxlabel_id for f in train_features], dtype=torch.long)
        all_char_input_ids = torch.tensor([f.char_input_ids for f in train_features], dtype=torch.long)
        all_sent_token_aug = torch.tensor([f.sent_token_aug for f in train_features], dtype=torch.float32)
        all_frcnn_feature = torch.stack([f.frcnn_feature for f in train_features])
        all_frcnn_mask = torch.tensor([f.frcnn_mask for f in train_features], dtype=torch.long)
        all_mrcnn_obj_ids = torch.tensor([f.mrcnn_obj_ids for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_added_input_mask, \
                     all_segment_ids, all_img_feats, all_label_ids, all_auxlabel_ids, all_char_input_ids,
                                   all_sent_token_aug, all_frcnn_feature, all_frcnn_mask, all_mrcnn_obj_ids)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        eval_examples = processor.get_dev_examples(args.data_dir)
        eval_features = convert_mm_examples_to_features(token_aug_emb_table,
            eval_examples, label_list, auxlabel_list, args.max_seq_length, tokenizer, args.crop_size, args.path_image)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_added_input_mask = torch.tensor([f.added_input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_img_feats = torch.stack([f.img_feat for f in eval_features])
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        all_auxlabel_ids = torch.tensor([f.auxlabel_id for f in eval_features], dtype=torch.long)
        all_char_input_ids = torch.tensor([f.char_input_ids for f in eval_features], dtype=torch.long)
        all_sent_token_aug = torch.tensor([f.sent_token_aug for f in eval_features], dtype=torch.float32)
        all_frcnn_feature = torch.stack([f.frcnn_feature for f in eval_features])
        all_frcnn_mask = torch.tensor([f.frcnn_mask for f in eval_features], dtype=torch.long)
        all_mrcnn_obj_ids = torch.tensor([f.mrcnn_obj_ids for f in eval_features], dtype=torch.long)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_added_input_mask, \
                                  all_segment_ids, all_img_feats, all_label_ids, all_auxlabel_ids, all_char_input_ids,
                                   all_sent_token_aug, all_frcnn_feature, all_frcnn_mask, all_mrcnn_obj_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        test_eval_examples = processor.get_test_examples(args.data_dir)
        test_eval_features = convert_mm_examples_to_features(token_aug_emb_table,
            test_eval_examples, label_list, auxlabel_list, args.max_seq_length, tokenizer, args.crop_size, args.path_image)
        all_input_ids = torch.tensor([f.input_ids for f in test_eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in test_eval_features], dtype=torch.long)
        all_added_input_mask = torch.tensor([f.added_input_mask for f in test_eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in test_eval_features], dtype=torch.long)
        all_img_feats = torch.stack([f.img_feat for f in test_eval_features])
        all_label_ids = torch.tensor([f.label_id for f in test_eval_features], dtype=torch.long)
        all_auxlabel_ids = torch.tensor([f.auxlabel_id for f in test_eval_features], dtype=torch.long)
        all_char_input_ids = torch.tensor([f.char_input_ids for f in test_eval_features], dtype=torch.long)
        all_sent_token_aug = torch.tensor([f.sent_token_aug for f in test_eval_features], dtype=torch.float32)
        all_frcnn_feature = torch.stack([f.frcnn_feature for f in test_eval_features])
        all_frcnn_mask = torch.tensor([f.frcnn_mask for f in test_eval_features], dtype=torch.long)
        all_mrcnn_obj_ids = torch.tensor([f.mrcnn_obj_ids for f in test_eval_features], dtype=torch.long)
        test_eval_data = TensorDataset(all_input_ids, all_input_mask, all_added_input_mask, all_segment_ids,
                                  all_img_feats, all_label_ids, all_auxlabel_ids, all_char_input_ids,
                                   all_sent_token_aug, all_frcnn_feature, all_frcnn_mask, all_mrcnn_obj_ids)
        # Run prediction for full data
        test_eval_sampler = SequentialSampler(test_eval_data)
        test_eval_dataloader = DataLoader(test_eval_data, sampler=test_eval_sampler, batch_size=args.eval_batch_size)

        max_dev_f1 = 0.0
        max_test_f1 = 0.0
        best_dev_epoch = 0
        best_test_epoch = 0
        logger.info("***** Running training *****")
        for train_idx in trange(int(args.num_train_epochs), desc="Epoch"):
            logger.info("********** Epoch: " + str(train_idx) + " **********")
            logger.info("  Num examples = %d", len(train_examples))
            logger.info("  Batch size = %d", args.train_batch_size)
            logger.info("  Num steps = %d", num_train_optimization_steps)
            model.train()
            encoder.train()
            encoder.zero_grad()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, added_input_mask, segment_ids, img_feats, label_ids, auxlabel_ids,\
                char_input_ids, sent_token_aug, frcnn_feature, frcnn_mask, mrcnn_obj_ids = batch
                with torch.no_grad():
                    imgs_f, img_mean, img_att = encoder(img_feats)
                trans_matrix = torch.tensor(trans_matrix).to(device)
                neg_log_likelihood = model(char_input_ids, sent_token_aug, frcnn_feature, frcnn_mask, mrcnn_obj_ids,
                    input_ids, segment_ids, input_mask, added_input_mask, img_att, trans_matrix, label_ids, auxlabel_ids)
                if n_gpu > 1:
                    neg_log_likelihood = neg_log_likelihood.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    neg_log_likelihood = neg_log_likelihood / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(neg_log_likelihood)
                else:
                    neg_log_likelihood.backward()

                tr_loss += neg_log_likelihood.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear(global_step/num_train_optimization_steps, args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            logger.info("***** Running evaluation on Dev Set*****")
            logger.info("  Num examples = %d", len(eval_examples))
            logger.info("  Batch size = %d", args.eval_batch_size)

            model.eval()
            encoder.eval()
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            y_true = []
            y_pred = []
            label_map = {i: label for i, label in enumerate(label_list, 1)}
            label_map[0] = "PAD"
            for input_ids, input_mask, added_input_mask, segment_ids, img_feats, label_ids, auxlabel_ids, \
                    char_input_ids, sent_token_aug, frcnn_feature, frcnn_mask, mrcnn_obj_ids in tqdm(eval_dataloader, desc="Evaluating"):
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                added_input_mask = added_input_mask.to(device)
                segment_ids = segment_ids.to(device)
                img_feats = img_feats.to(device)
                label_ids = label_ids.to(device)
                auxlabel_ids = auxlabel_ids.to(device)
                #trans_matrix = torch.tensor(trans_matrix).to(device)
                char_input_ids = char_input_ids.to(device)
                sent_token_aug = sent_token_aug.to(device)
                frcnn_feature = frcnn_feature.to(device)
                frcnn_mask = frcnn_mask.to(device)
                mrcnn_obj_ids = mrcnn_obj_ids.to(device)

                with torch.no_grad():
                    imgs_f, img_mean, img_att = encoder(img_feats)
                    predicted_label_seq_ids, _, _, _ = model(char_input_ids, sent_token_aug, frcnn_feature, frcnn_mask, mrcnn_obj_ids,
                                                    input_ids, segment_ids, input_mask, added_input_mask, img_att, trans_matrix)

                #logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
                #logits = logits.detach().cpu().numpy()
                # logits = predicted_label_seq_ids.detach().cpu().numpy()
                logits = predicted_label_seq_ids
                label_ids = label_ids.to('cpu').numpy()
                input_mask = input_mask.to('cpu').numpy()
                for i, mask in enumerate(input_mask):
                    temp_1 = []
                    temp_2 = []
                    for j, m in enumerate(mask):
                        if j == 0:
                            continue
                        if m:
                            if label_map[label_ids[i][j]] != "X" and label_map[label_ids[i][j]] != "[SEP]":
                                temp_1.append(label_map[label_ids[i][j]])
                                temp_2.append(label_map[logits[i][j]])
                        else:
                            #temp_1.pop()
                            #temp_2.pop()
                            break
                    y_true.append(temp_1)
                    y_pred.append(temp_2)
            report = classification_report(y_true, y_pred, digits=4)
            logger.info("***** Dev Eval results *****")
            logger.info("\n%s", report)
            #eval_true_label = np.concatenate(y_true)
            #eval_pred_label = np.concatenate(y_pred)
            #precision, recall, F_score = macro_f1(eval_true_label, eval_pred_label)
            F_score_dev = float(report.split('\n')[-3].split('      ')[-2].split('    ')[-1])
            print("F-score: ", F_score_dev)

            logger.info("***** Running Test evaluation *****")
            logger.info("  Num examples = %d", len(test_eval_examples))
            logger.info("  Batch size = %d", args.eval_batch_size)
            y_true = []
            y_pred = []
            label_map = {i: label for i, label in enumerate(label_list, 1)}
            label_map[0] = "PAD"
            for input_ids, input_mask, added_input_mask, segment_ids, img_feats, label_ids, auxlabel_ids,\
                    char_input_ids, sent_token_aug, frcnn_feature, frcnn_mask, mrcnn_obj_ids in tqdm(test_eval_dataloader,
                                                                                                   desc="Evaluating"):
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                added_input_mask = added_input_mask.to(device)
                segment_ids = segment_ids.to(device)
                img_feats = img_feats.to(device)
                label_ids = label_ids.to(device)
                auxlabel_ids = auxlabel_ids.to(device)
                #trans_matrix = torch.tensor(trans_matrix).to(device)
                char_input_ids = char_input_ids.to(device)
                sent_token_aug = sent_token_aug.to(device)
                frcnn_feature = frcnn_feature.to(device)
                frcnn_mask = frcnn_mask.to(device)
                mrcnn_obj_ids = mrcnn_obj_ids.to(device)

                with torch.no_grad():
                    imgs_f, img_mean, img_att = encoder(img_feats)
                    predicted_label_seq_ids, _, _, _ = model(char_input_ids, sent_token_aug, frcnn_feature, frcnn_mask, mrcnn_obj_ids,
                                                    input_ids, segment_ids, input_mask, added_input_mask, img_att,trans_matrix)

                # logits = torch.argmax(F.log_softmax(logits,dim=2),dim=2)
                # logits = logits.detach().cpu().numpy()
                # logits = predicted_label_seq_ids.detach().cpu().numpy()
                logits = predicted_label_seq_ids
                label_ids = label_ids.to('cpu').numpy()
                input_mask = input_mask.to('cpu').numpy()
                for i, mask in enumerate(input_mask):
                    temp_1 = []
                    temp_2 = []
                    for j, m in enumerate(mask):
                        if j == 0:
                            continue
                        if m:
                            if label_map[label_ids[i][j]] != "X" and label_map[label_ids[i][j]] != "[SEP]":
                                temp_1.append(label_map[label_ids[i][j]])
                                temp_2.append(label_map[logits[i][j]])
                        else:
                            # temp_1.pop()
                            # temp_2.pop()
                            break
                    y_true.append(temp_1)
                    y_pred.append(temp_2)
            report = classification_report(y_true, y_pred, digits=4)
            logger.info("***** Test Eval results *****")
            logger.info("\n%s", report)
            F_score_test = float(report.split('\n')[-4].split('      ')[-2].split('    ')[-1])

            if F_score_dev > max_dev_f1:
                # Save a trained model and the associated configuration
                """
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                encoder_to_save = encoder.module if hasattr(encoder,
                                                            'module') else encoder  # Only save the model it-self
                torch.save(model_to_save.state_dict(), output_model_file)
                torch.save(encoder_to_save.state_dict(), output_encoder_file)
                with open(output_config_file, 'w') as f:
                    f.write(model_to_save.config.to_json_string())
                label_map = {i: label for i, label in enumerate(label_list, 1)}
                model_config = {"bert_model": args.bert_model, "do_lower": args.do_lower_case,
                                "max_seq_length": args.max_seq_length, "num_labels": len(label_list) + 1,
                                "label_map": label_map}
                json.dump(model_config, open(os.path.join(args.output_dir, "model_config.json"), "w"))
                """
                max_dev_f1 = F_score_dev
                best_dev_epoch = train_idx
            if F_score_test > max_test_f1:
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                encoder_to_save = encoder.module if hasattr(encoder,
                                                            'module') else encoder  # Only save the model it-self
                torch.save(model_to_save.state_dict(), output_model_file)
                torch.save(encoder_to_save.state_dict(), output_encoder_file)
                with open(output_config_file, 'w') as f:
                    f.write(model_to_save.config.to_json_string())
                label_map = {i: label for i, label in enumerate(label_list, 1)}
                model_config = {"bert_model": args.bert_model, "do_lower": args.do_lower_case,
                                "max_seq_length": args.max_seq_length, "num_labels": len(label_list) + 1,
                                "label_map": label_map}
                json.dump(model_config, open(os.path.join(args.output_dir, "model_config.json"), "w"))

                max_test_f1 = F_score_test
                best_test_epoch = train_idx

    print("**************************************************")
    print("The best epoch on the dev set: ", best_dev_epoch)
    print("The best Micro-F1 score on the dev set: ", max_dev_f1)
    print("The best epoch on the test set: ", best_test_epoch)
    print("The best Micro-F1 score on the test set: ", max_test_f1)
    print('\n')

    config = BertConfig(output_config_file)
    if args.mm_model == 'MSCMT':
        model = MSCMT(config, layer_num1=args.layer_num1, layer_num2=args.layer_num2,
                                            layer_num3=args.layer_num3, num_labels=num_labels, auxnum_labels = auxnum_labels)
    else:
        print('please define your MNER Model')

    model.load_state_dict(torch.load(output_model_file))
    model.to(device)
    encoder_state_dict = torch.load(output_encoder_file)
    encoder.load_state_dict(encoder_state_dict)
    encoder.to(device)

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_examples = processor.get_test_examples(args.data_dir)
        eval_features = convert_mm_examples_to_features(token_aug_emb_table,
            eval_examples, label_list, auxlabel_list, args.max_seq_length, tokenizer, args.crop_size, args.path_image)
        logger.info("***** Running Test Evaluation with the Best Model on the Dev Set*****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_added_input_mask = torch.tensor([f.added_input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_img_feats = torch.stack([f.img_feat for f in eval_features])
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        all_auxlabel_ids = torch.tensor([f.auxlabel_id for f in eval_features], dtype=torch.long)
        all_char_input_ids = torch.tensor([f.char_input_ids for f in eval_features], dtype=torch.long)
        all_sent_token_aug = torch.tensor([f.sent_token_aug for f in eval_features], dtype=torch.float32)
        all_frcnn_feature = torch.stack([f.frcnn_feature for f in eval_features])
        all_frcnn_mask = torch.tensor([f.frcnn_mask for f in eval_features], dtype=torch.long)
        all_mrcnn_obj_ids = torch.tensor([f.mrcnn_obj_ids for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_added_input_mask, all_segment_ids, all_img_feats,
                                  all_label_ids, all_auxlabel_ids, all_char_input_ids,
                                   all_sent_token_aug, all_frcnn_feature, all_frcnn_mask, all_mrcnn_obj_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
        model.eval()
        encoder.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        y_true = []
        y_pred = []
        y_true_idx = []
        y_pred_idx = []

        text2img_att_dict = dict() # text for resnet
        text2img_att_list = []
        img2text_att_dict = dict() # resnet for text
        img2text_att_list = []
        obj2text_att_dict = dict() # rcnn for text
        obj2text_att_list = []

        label_map = {i : label for i, label in enumerate(label_list,1)}
        label_map[0] = "PAD"
        for input_ids, input_mask, added_input_mask, segment_ids, img_feats, label_ids, auxlabel_ids,\
                char_input_ids, sent_token_aug, frcnn_feature, frcnn_mask, mrcnn_obj_ids in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            added_input_mask = added_input_mask.to(device)
            segment_ids = segment_ids.to(device)
            img_feats = img_feats.to(device)
            label_ids = label_ids.to(device)
            auxlabel_ids = auxlabel_ids.to(device)
            trans_matrix = torch.tensor(trans_matrix).to(device)
            char_input_ids = char_input_ids.to(device)
            sent_token_aug = sent_token_aug.to(device)
            frcnn_feature = frcnn_feature.to(device)
            frcnn_mask = frcnn_mask.to(device)
            mrcnn_obj_ids = mrcnn_obj_ids.to(device)

            with torch.no_grad():
                imgs_f, img_mean, img_att = encoder(img_feats)
                predicted_label_seq_ids, text2img_att, img2text_att, obj2text_att = model(char_input_ids, sent_token_aug, frcnn_feature, frcnn_mask, mrcnn_obj_ids,
                                                 input_ids, segment_ids, input_mask, added_input_mask, img_att, trans_matrix)
            
            #logits = torch.argmax(F.log_softmax(logits,dim=2),dim=2)
            #logits = logits.detach().cpu().numpy()
            # logits = predicted_label_seq_ids.detach().cpu().numpy()
            logits = predicted_label_seq_ids
            label_ids = label_ids.to('cpu').numpy()
            input_mask = input_mask.to('cpu').numpy()
            for i,mask in enumerate(input_mask):
                temp_1 = []
                temp_2 = []
                tmp1_idx = []
                tmp2_idx = []
                #count = 0
                for j, m in enumerate(mask):
                    if j == 0:
                        continue
                    if m:
                        if label_map[label_ids[i][j]] != "X" and label_map[label_ids[i][j]] != "[SEP]":
                            temp_1.append(label_map[label_ids[i][j]])
                            tmp1_idx.append(label_ids[i][j])
                            temp_2.append(label_map[logits[i][j]])
                            tmp2_idx.append(logits[i][j])
                        #if label_map[label_ids[i][j]].startswith("B"):
                            #count += 1
                    else:
                        #temp_1.pop()
                        #temp_2.pop()
                        break
                y_true.append(temp_1)
                y_pred.append(temp_2)
                y_true_idx.append(tmp1_idx)
                y_pred_idx.append(tmp2_idx)
                #if count > 1:
                    #multi_ent_y_true.append(temp_1)
                    #multi_ent_y_pred.append(temp_2)
                text2img_att_list.append(text2img_att[i])
                img2text_att_list.append(img2text_att[i])
                obj2text_att_list.append(obj2text_att[i])

        report = classification_report(y_true, y_pred,digits=4)
        #multi_ent_report = classification_report(multi_ent_y_true, multi_ent_y_pred,digits=4)

        sentence_list = []
        test_data, imgs, _ = processor._read_mmtsv(os.path.join(args.data_dir, "test.txt"))
        output_pred_file = os.path.join(args.output_dir, "mtmner_pred.txt")
        fout = open(output_pred_file, 'w')
        for i in range(len(y_pred)):
            sentence = test_data[i][0]
            sentence_list.append(sentence)
            img = imgs[i]
            samp_pred_label = y_pred[i]
            samp_true_label = y_true[i]
            fout.write(img+'\n')
            fout.write(' '.join(sentence)+'\n')
            fout.write(' '.join(samp_pred_label)+'\n')
            fout.write(' '.join(samp_true_label)+'\n'+'\n')
            if i < 100: # we only record 100 images for the memory
                text2img_att_dict[img] = text2img_att_list[i]
                img2text_att_dict[img] = img2text_att_list[i]
                obj2text_att_dict[img] = obj2text_att_list[i]
        fout.close()
        torch.save(text2img_att_dict, os.path.join(args.output_dir, "text2img_att.pt")) # for visual
        torch.save(img2text_att_dict, os.path.join(args.output_dir, "img2text_att.pt")) # resnet for text
        torch.save(obj2text_att_dict, os.path.join(args.output_dir, "obj2text_att.pt")) # mrcnn for text

        reverse_label_map = {label: i for i, label in enumerate(label_list, 1)}
        acc, f1, p, r = evaluate(y_pred_idx, y_true_idx, sentence_list, reverse_label_map)
        print("Overall: ", p, r, f1)
        per_f1, per_p, per_r = evaluate_each_class(y_pred_idx, y_true_idx, sentence_list, reverse_label_map, 'PER')
        print("Person: ", per_p, per_r, per_f1)
        loc_f1, loc_p, loc_r = evaluate_each_class(y_pred_idx, y_true_idx, sentence_list, reverse_label_map, 'LOC')
        print("Location: ", loc_p, loc_r, loc_f1)
        org_f1, org_p, org_r = evaluate_each_class(y_pred_idx, y_true_idx, sentence_list, reverse_label_map, 'ORG')
        print("Organization: ", org_p, org_r, org_f1)
        misc_f1, misc_p, misc_r = evaluate_each_class(y_pred_idx, y_true_idx, sentence_list, reverse_label_map, 'MISC')
        print("Miscellaneous: ", misc_p, misc_r, misc_f1)

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Test Eval results *****")
            logger.info("\n%s", report)
            writer.write(report)
            writer.write("Overall: " + str(p) + ' ' + str(r) + ' ' + str(f1) + '\n')
            writer.write("Person: " + str(per_p) + ' ' + str(per_r) + ' ' + str(per_f1) + '\n')
            writer.write("Location: " + str(loc_p) + ' ' + str(loc_r) + ' ' + str(loc_f1) + '\n')
            writer.write("Organization: " + str(org_p) + ' ' + str(org_r) + ' ' + str(org_f1) + '\n')
            writer.write("Miscellaneous: " + str(misc_p) + ' ' + str(misc_r) + ' ' + str(misc_f1) + '\n')

if __name__ == "__main__":
    main()
