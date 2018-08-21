from collections import defaultdict
import re
import math
import numpy as np
import jieba


def word_tokenize(sent):
    # doc = nlp(sent)
    # return [token.text for token in doc]
    return list(jieba.cut(sent))


def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print("Token {} cannot be found".format(token))
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans


def substring_indexes(substring, string):
    """
    在全文中找到所有精确匹配的内容，用于过滤
    :param substring:
    :param string:
    :return:
    """
    last_found = -1
    while True:
        last_found = string.find(substring, last_found + 1)
        if last_found == -1:
            break
        yield last_found


def get_match_size(cand_ngram: list, ref_ngram: list) -> (int, int):
    ref_set = defaultdict(int)
    cand_set = defaultdict(int)

    for ngram in ref_ngram:
        ref_set[ngram] += 1
    for ngram in cand_ngram:
        cand_set[ngram] += 1
    match_size = 0
    for ngram in cand_set:
        match_size += min(cand_set[ngram], ref_set[ngram])
    cand_size = len(cand_ngram)
    return match_size, cand_size


def get_ngram(sent: str, n_size: int) -> list:
    return [sent[left: left + n_size] for left in range(len(sent) - n_size + 1)]


def get_trim_string(string: str) -> str:
    return re.sub(r'\s+', '', string)


def word2char(str_in):
    str_out = str_in.replace(' ', '')
    return ''.join(str_out.split())


class RougeL(object):
    def __init__(self, gamma=1.2):
        self.gamma = gamma  # gamma 为常量
        self.inst_scores = []

    def lcs(self, string: str, sub: str) -> int:
        """计算最长公共子序列
        Arguments:
            string {str} -- 字符串
            sub {str} -- 字符串

        Returns:
            int -- 最长公共子序列的长度
        """
        str_length = len(string)
        sub_length = len(sub)

        lengths = np.zeros(((str_length + 1), (sub_length + 1)), dtype=np.int)
        for i in range(1, str_length + 1):
            for j in range(1, sub_length + 1):
                if string[i - 1] == sub[j - 1]:
                    lengths[i][j] = lengths[i - 1][j - 1] + 1
                else:
                    lengths[i][j] = max(lengths[i - 1][j], lengths[i][j - 1])
        return lengths[str_length, sub_length]

    def add_inst(self, cand: str, ref: str):
        """根据参考答案分析出预测答案的分数

        Arguments:
            cand {str} -- 预测答案
            ref {str} -- 参考答案
        """

        basic_lcs = self.lcs(cand, ref)
        p_denom = len(cand)
        r_denom = len(ref)
        prec = basic_lcs / p_denom if p_denom > 0. else 0.
        rec = basic_lcs / r_denom if r_denom > 0. else 0.
        if prec != 0 and rec != 0:
            score = ((1 + self.gamma ** 2) * prec * rec) / float(rec + self.gamma ** 2 * prec)
        else:
            score = 0
        self.inst_scores.append(score)

    def get_score(self) -> float:
        """计算cand预测数据的RougeL分数

        Returns:
            float -- RougeL分数
        """
        return 1. * sum(self.inst_scores) / len(self.inst_scores)


class Bleu(object):
    def __init__(self, n_size=4):
        self.match_ngram = {}
        self.candi_ngram = {}
        self.bp_r = 0
        self.bp_c = 0
        self.n_size = n_size

    def add_inst(self, cand: str, ref: str):
        """根据添加的预测答案和参考答案，更新match_gram和candi_gram

        Arguments:
            cand {str} -- 预测答案
            ref {str} -- 参考答案
        """
        for n_size in range(self.n_size):
            self.count_ngram(cand, ref, n_size + 1)
        self.count_bp(cand, ref)

    def count_ngram(self, cand: str, ref: str, n_size: int):
        """计算子序列重合的个数，并存储到字典中

        Arguments:
            cand {str} -- 预备答案
            ref {str} -- 参考答案
            n_size {int} -- 子序列的大小
        """
        cand_ngram = get_ngram(cand, n_size)
        ref_ngram = get_ngram(ref, n_size)
        if n_size not in self.match_ngram:
            self.match_ngram[n_size] = 0
            self.candi_ngram[n_size] = 0
        match_size, cand_size = get_match_size(cand_ngram, ref_ngram)
        self.match_ngram[n_size] += match_size
        self.candi_ngram[n_size] += cand_size

    def count_bp(self, cand: str, ref: str):
        """计算BP参数对应的r和c

        Arguments:
            cand {str} -- 预备答案
            ref {str} -- 参考答案

        Returns:
            float -- BP参数计算结果
        """
        self.bp_c += len(cand)
        self.bp_r += len(ref)

    def get_score(self) -> float:
        """计算字符串cand的Bleu分数, 并返回

        Returns:
            bleu_score {float} -- bleu分数
        """
        prob_list = [
            self.match_ngram[n_size + 1] / float(self.candi_ngram[n_size + 1])
            for n_size in range(self.n_size)
        ]
        bleu_score = prob_list[0]
        for n in range(1, self.n_size):
            bleu_score *= prob_list[n]
        bleu_score = bleu_score ** (1. / float(self.n_size))
        bp = math.exp(min(1 - self.bp_r / float(self.bp_c), 0))
        bleu_score = bp * bleu_score
        return bleu_score


if __name__ == '__main__':
    cand = '我是中国人'
    ref = '我是孙维松'
    cand_ngram = get_ngram(cand, 0)
    ref_ngram = get_ngram(ref, 0)
    print('cand_ngram: {}'.format(cand_ngram))
    print('ref_ngram: {}'.format(ref_ngram))
