import torch
import numpy as np
import gensim
from config import Args as aug_config

aug_model = 'XXX/data_preprocess/text_prep/aug/TPANN_supplemental/word2vec/word2vec_200dim.model'


class BaseInternalAugmentation:

    def __init__(self, args):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.args = args
        self.wv_model = gensim.models.Word2Vec.load(aug_model)

    def consine_sim(self, word):
        sim_scores = []
        sim_words = self.wv_model.wv.most_similar(word, topn=self.args.token_aug_num)
        for sim_word in sim_words:
            sim_scores.append(sim_word[1])
        return sim_scores

    def most_k_words(self, word):
        words = []
        sim_words = self.wv_model.wv.most_similar(word, topn=self.args.token_aug_num)
        for sim_word in sim_words:
            words.append(sim_word[0])
        return words

    def compute_atten_embedding(self, word):
        sim_scores = np.array(self.consine_sim(word), dtype=np.float32)
        e_x = np.exp(sim_scores - np.max(sim_scores))
        f_x = e_x / e_x.sum()  # the softmax weight

        words_embeddings = []
        for each_sim_word in self.most_k_words(word):
            words_embeddings.append(self.wv_model.wv[each_sim_word])
        similar_words_embedding = np.array(words_embeddings, dtype=np.float32)
        hard_atten_embedding = np.sum(f_x[:, None] * similar_words_embedding, axis=0)
        return hard_atten_embedding


class HardInternalAugmentation(BaseInternalAugmentation):
    def __init__(self, args, word2index):
        super(HardInternalAugmentation, self).__init__(args)
        self.word2index = word2index

    def hard_augmentation_embedding(self, hidden_dim):
        attention_embedding = np.zeros([len(self.word2index), hidden_dim], dtype=np.float32)
        for word, ind in self.word2index.items():
            if word in self.wv_model.wv.index2word:
                attention_embedding[ind, :] = self.compute_atten_embedding(word)
            else:
                # get the random augmentation embedding
                scale = np.sqrt(3.0 / hidden_dim)
                attention_embedding[ind, :] = np.random.uniform(-scale, scale, [1, hidden_dim])

        return attention_embedding

if __name__ == "__main__":
    HA = HardInternalAugmentation(aug_config, aug_config.word_dict)
    token_aug_emb_table = HA.hard_augmentation_embedding(aug_config.aug_dim)
    np.save(aug_config.token_aug_vec_addr, token_aug_emb_table)