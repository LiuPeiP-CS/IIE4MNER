import os
import torch
from collections import Counter
import torch.nn as nn

def build_vocab(text_data_dir):
    word_list = []
    char_list = []
    label_list = []

    for text_file in ['train', 'dev', 'test']:
        with open(os.path.join(text_data_dir, text_file), 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line != '':
                    if not line.startswith('IMGID:'):
                        # print(line.split('\t'))
                        word = line.split('\t')[0]
                        label = line.split('\t')[-1]
                        # word = preprocess_word(word)
                        word_list.append(word)
                        label_list.append(label)
                        for char in word:
                            char_list.append(char)

    # word_set = list(set(word_list))
    word_counter = Counter(word_list)
    word_set = [word[0] for word in word_counter.most_common()]  # make sure the number of word>=2
    print("The word set is ", word_set)
    word2index = {each_word: word_index+2 for word_index, each_word in enumerate(word_set)}
    word2index['[PAD]'] = 0; word2index['[UNK]'] = 1
    index2word = {word_index: each_word for each_word, word_index in word2index.items()}
    print("There is the dictionary for index2word : {}".format(index2word))

    char_counter = Counter(char_list)
    char_set = [char[0] for char in char_counter.most_common()]  # make sure the number of char>=2
    print("The char set is ", char_set)
    char2index = {each_char: char_index+5 for char_index, each_char in enumerate(char_set)}
    char2index['[PAD]'] = 0; char2index['[UNK]'] = 1; char2index['[X]'] = 2; char2index['[XC]'] = 3; char2index['[XS]'] = 4
    index2char = {char_index: each_char for each_char, char_index in char2index.items()}
    print("There is the dictionary for index2char : {}".format(index2char))

    label_counter = Counter(label_list)
    label_set = [label_item[0] for label_item in label_counter.most_common()]
    print("The label set is ",label_set)
    label2index = {each_label: label_index + 5 for label_index, each_label in enumerate(label_set)}
    label2index['[PAD]'] = 0;  label2index['[UNK]'] = 1; label2index['X'] = 2; label2index['[CLS]'] = 3; label2index['[SEP]'] = 4;
    index2label = {label_index: each_label for each_label, label_index in label2index.items()}
    print("There is the dictionary for index2label : {}".format(index2label))

    # return word2index, char2index, label2index
    return word2index, char2index


class char_cnn(nn.Module):
    def __init__(self,
                 max_wordlen=30,
                 kernel_list="2,3,4",
                 filters_num=32,
                 char_vocab_size=1000,
                 char_emb_dim=30,
                 final_char_dim=50):
        super(char_cnn, self).__init__()

        self.char_emb = nn.Embedding(char_vocab_size, char_emb_dim, padding_idx=0)
        nn.init.xavier_normal_(self.char_emb.weight)

        self.kernel_list = list(map(int, kernel_list.split(','))) # '2,3,4' -> [2,3,4]

        self.convs = nn.ModuleList([ # 构建多个模块，每个模块是以kernel来定义的一系列网络Sequential
            nn.Sequential(
                nn.Conv1d(char_emb_dim, filters_num, kernel, padding=kernel//2),
                nn.Tanh(),
                nn.MaxPool1d(max_wordlen - kernel + 1),
                nn.Dropout(0.25)
            ) for kernel in self.kernel_list
        ])

        self.linear = nn.Sequential(
            nn.Linear(filters_num*len(self.kernel_list), 100),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(100, final_char_dim)
        )

    def forward(self, x):
        batch_size, max_seqlen, max_wordlen = x.shape
        x = self.char_emb(x)
        x = x.view(batch_size*max_seqlen, max_wordlen, -1)
        x = x.transpose(2, 1)

        conv_list = [conv(x) for conv in self.convs]
        conv_concat = torch.cat(conv_list, dim=-1)
        conv_concat = conv_concat.view(conv_concat.shape[0], -1)

        output = self.linear(conv_concat)
        output = output.view(batch_size, max_seqlen, -1)
        return output # (batch_size, max_seqlen, final_char_dim)


class Highway(nn.Module):
    def __init__(self, size, num_layers, f):
        super(Highway, self).__init__()
        self.num_layers = num_layers
        self.nonlinear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.f = f

    def forward(self, x):
        """
            :param x: tensor with shape of [batch_size, size]
            :return: tensor with shape of [batch_size, size]
            applies σ(x) ⨀ (f(G(x))) + (1 - σ(x)) ⨀ (Q(x)) transformation | G and Q is affine transformation,
            f is non-linear transformation, σ(x) is affine transformation with sigmoid non-linearition
            and ⨀ is element-wise multiplication
            """
        for layer in range(self.num_layers):
            gate = torch.sigmoid(self.gate[layer](x))
            nonlinear = self.f(self.nonlinear[layer](x))
            linear = self.linear[layer](x)
            x = gate * nonlinear + (1 - gate) * linear
        return x

class GMF(nn.Module):
    """GMF (Gated Multimodal Fusion)"""

    def __init__(self):
        super(GMF, self).__init__()
        self.text_linear = nn.Linear(768, 768)  # Inferred from code (dim isn't written on paper)
        self.img_linear = nn.Linear(768, 768)
        self.gate_linear = nn.Linear(768 * 2, 1)

    def forward(self, att_text_features, att_img_features):
        """
        :param att_text_features: (batch_size, max_seq_len, hidden_dim)  (16, 35, 200)
        :param att_img_features: (batch_size, max_seq_len, hidden_dim)
        :return: multimodal_features
        """
        new_img_feat = torch.tanh(self.img_linear(att_img_features))  # [batch_size, max_seq_len, hidden_dim]
        new_text_feat = torch.tanh(self.text_linear(att_text_features))  # [batch_size, max_seq_len, hidden_dim]

        gate_img = self.gate_linear(torch.cat([new_img_feat, new_text_feat], dim=-1))  # [batch_size, max_seq_len, 1]
        gate_img = torch.sigmoid(gate_img)  # [batch_size, max_seq_len, 1]
        gate_img = gate_img.repeat(1, 1, 768)  # [batch_size, max_seq_len, hidden_dim]
        multimodal_features = torch.mul(gate_img, new_img_feat) + torch.mul(1 - gate_img, new_text_feat)  # [batch_size, max_seq_len, hidden_dim]

        return multimodal_features


class FiltrationGate(nn.Module):
    """
    In this part, code is implemented in other way compare to equation on paper.
    So I mixed the method between paper and code (e.g. Add `nn.Linear` after the concatenated matrix)
    """

    def __init__(self):
        super(FiltrationGate, self).__init__()
        self.text_linear = nn.Linear(768, 768, bias=False)
        self.multimodal_linear = nn.Linear(768, 768, bias=True)
        self.gate_linear = nn.Linear(768 * 2, 1)

        self.resv_linear = nn.Linear(768, 768)
        # self.output_linear = nn.Linear(768 * 2, len(TweetProcessor.get_labels()))

    def forward(self, text_features, multimodal_features):
        """
        :param text_features: Original text feature from BiLSTM [batch_size, max_seq_len, hidden_dim]
        :param multimodal_features: Feature from GMF [batch_size, max_seq_len, hidden_dim]
        :return: output: Will be the input for CRF decoder [batch_size, max_seq_len, hidden_dim]
        """
        # [batch_size, max_seq_len, 2 * hidden_dim]
        concat_feat = torch.cat([self.text_linear(text_features), self.multimodal_linear(multimodal_features)], dim=-1)
        # This part is not written on equation, but if is needed
        filtration_gate = torch.sigmoid(self.gate_linear(concat_feat))  # [batch_size, max_seq_len, 1]
        filtration_gate = filtration_gate.repeat(1, 1, 768)  # [batch_size, max_seq_len, hidden_dim]

        reserved_multimodal_feat = torch.mul(filtration_gate,
                                             torch.tanh(self.resv_linear(multimodal_features)))  # [batch_size, max_seq_len, hidden_dim]
        output = torch.cat([text_features, reserved_multimodal_feat], dim=-1)  # [batch_size, max_seq_len, 2 * hidden_dim]

        return output
