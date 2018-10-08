import os
import tensorflow as tf
import random
import jieba
import pynlpir
import ujson as json
import numpy as np
from codecs import open
from tqdm import tqdm
from collections import Counter
from utils import word_tokenize, convert_idx, substring_indexes

'''
该文件不再使用fasttext的word embedding。

This file is taken and modified from R-Net by HKUST-KnowComp
https://github.com/HKUST-KnowComp/R-Net
'''


def process_file(filename, data_type, word_counter, char_counter, ques_limit):
    """
    从文本文件中读取内容后进行初步处理。如果数据集是train的话，需要进行内容过滤
    :param filename:
    :param data_type:
    :param word_counter:
    :param char_counter:
    :return:
    """

    print("Processing {} examples...".format(data_type))
    examples = []
    eval_examples = {}
    total = 0

    with open(filename, "r") as fh:
        source = json.load(fh)
        # TODO 预处理中进行了过滤，但后续没有办法计算spans
        for article in tqdm(source):
            content = article['article_title'] + '。' + article['article_content']
            content_tokens = word_tokenize(content)
            content_chars = [list(token) for token in content_tokens]
            spans = convert_idx(content, content_tokens)

            for token in content_tokens:
                word_counter[token] += len(article['questions'])
                for char in token:
                    char_counter[char] += len(article["questions"])

            for q in article['questions']:
                question_text = q["question"]
                answer_text = q['answer']
                question_tokens = word_tokenize(question_text)
                question_tokens = shrink_question_tokens(question_tokens, ques_limit)

                question_chars = [list(token) for token in question_tokens]
                result = list(substring_indexes(answer_text, content))

                for token in question_tokens:
                    word_counter[token] += 1
                    for char in token:
                        char_counter[char] += 1

                if len(result) == 1:
                    # 将result的字符转换成分词之后的位置，y1 y2 分别是开始的分词位置和结束的位置
                    current_pos, start_token, end_token = 0, -1, -1
                    for token_cnt, token in enumerate(content_tokens):
                        if current_pos > result[0] and start_token == -1:
                            start_token = token_cnt - 1
                        if current_pos > result[0] + len(q["answer"]):
                            end_token = token_cnt - 2
                            break
                        current_pos += len(token)
                    total += 1
                    example = {
                        "context_tokens": content_tokens,
                        "context_chars": content_chars,
                        "ques_tokens": question_tokens,
                        "ques_chars": question_chars,
                        "y1s": [start_token], "y2s": [end_token],
                        "id": total
                    }
                    eval_examples[str(total)] = {
                        "context": content,
                        "spans": spans,  # 全文的每个token与位置的对应关系
                        "answers": [answer_text],  # TODO 改成不分para的
                        "uuid": q["questions_id"]
                    }  # example中没有存储原始的问题文本信息，在这里保存了，在后续的结果展示中可以用到。
                    examples.append(example)  # 不考虑任何跨段的问题

        random.shuffle(examples)
        print("{} questions in total".format(len(examples)))
    return examples, eval_examples


def get_embedding(counter, data_type, limit=-1, emb_file=None, vec_size=None):
    print("Generating {} embedding...".format(data_type))
    embedding_dict = {}
    filtered_elements = [k for k, v in counter.items() if v > limit]
    if emb_file is not None:
        assert vec_size is not None
        with open(emb_file, "r", encoding="utf-8") as fh:
            for line in tqdm(fh):
                array = line.split()
                word = "".join(array[0:-vec_size])
                vector = list(map(float, array[-vec_size:]))
                if word in counter and counter[word] > limit:
                    embedding_dict[word] = vector
        cnt = 0
        for word in counter:
            if word not in embedding_dict:
                embedding_dict[word] = [np.random.normal(scale=0.1) for _ in range(vec_size)]
                cnt += 1
        print("{} / {} tokens have corresponding {} embedding vector".format(
            len(embedding_dict), len(filtered_elements), data_type))
        print("{} tokens are initialized randomly".format(cnt))
    else:
        assert vec_size is not None
        for token in filtered_elements:
            embedding_dict[token] = [np.random.normal(scale=0.1) for _ in range(vec_size)]
        print("{} tokens have corresponding embedding vector".format(len(filtered_elements)))

    null = "--NULL--"
    oov = "--OOV--"
    token2idx_dict = {token: idx for idx, token in enumerate(embedding_dict.keys(), 2)}
    token2idx_dict[null] = 0
    token2idx_dict[oov] = 1
    embedding_dict[null] = [0. for _ in range(vec_size)]
    embedding_dict[oov] = [0. for _ in range(vec_size)]
    idx2emb_dict = {idx: embedding_dict[token] for token, idx in token2idx_dict.items()}
    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
    return emb_mat, token2idx_dict


def build_features(config, examples, data_type, out_file, word2idx_dict, char2idx_dict):
    """
    将经过tokenize的文件加工成为tfrecords格式
    :param config:
    :param examples: 已经经过tokenize的文本，以问题为单位形成的训练语料（是的你没看错，每个预料都会把当前问题对应的原文重复一遍）。
        每个条目包含本文所有的context和当前的问题，以及答案对应的位置。
    :param data_type:
    :param out_file: 输出文件名
    :param word2idx_dict: 词表
    :param char2idx_dict:
    :return meta: 只包含一个总条目数
    """
    para_limit = config.para_limit
    ques_limit = config.ques_limit
    # ans_limit = config.ans_limit
    char_limit = config.char_limit

    def filter_func(_example):
        return len(_example["context_tokens"]) > para_limit or \
               len(_example["ques_tokens"]) > ques_limit

    print("Processing {} examples...".format(data_type))
    writer = tf.python_io.TFRecordWriter(out_file)
    total = 0
    total_ = 0
    meta = {}
    for example in tqdm(examples):
        total_ += 1

        if filter_func(example):
            continue

        total += 1
        context_idxs = np.zeros([para_limit], dtype=np.int32)
        context_char_idxs = np.zeros([para_limit, char_limit], dtype=np.int32)
        ques_idxs = np.zeros([ques_limit], dtype=np.int32)
        ques_char_idxs = np.zeros([ques_limit, char_limit], dtype=np.int32)
        y1 = np.zeros([para_limit], dtype=np.float32)
        y2 = np.zeros([para_limit], dtype=np.float32)

        def _get_word(word):
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in word2idx_dict:
                    return word2idx_dict[each]
            return 1

        def _get_char(_char):
            if _char in char2idx_dict:
                return char2idx_dict[_char]
            return 1

        for i, token in enumerate(example["context_tokens"]):
            context_idxs[i] = _get_word(token)

        for i, token in enumerate(example["ques_tokens"]):
            ques_idxs[i] = _get_word(token)

        for i, token in enumerate(example["context_chars"]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                context_char_idxs[i, j] = _get_char(char)

        for i, token in enumerate(example["ques_chars"]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                ques_char_idxs[i, j] = _get_char(char)

        start, end = example["y1s"][-1], example["y2s"][-1]
        y1[start], y2[end] = 1.0, 1.0
        record = tf.train.Example(
            features=tf.train.Features(feature={
                "context_idxs": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[context_idxs.tostring()])),
                "ques_idxs": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[ques_idxs.tostring()])),
                "context_char_idxs": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[context_char_idxs.tostring()])),
                "ques_char_idxs": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[ques_char_idxs.tostring()])),
                "y1": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y1.tostring()])),
                "y2": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y2.tostring()])),
                "id": tf.train.Feature(int64_list=tf.train.Int64List(value=[example["id"]]))
            })
        )
        writer.write(record.SerializeToString())
    print("Built {} / {} instances of features in total".format(total, total_))
    meta["total"] = total
    writer.close()
    return meta


def shrink_question_tokens(question_tokenized, question_limit):
    if len(question_tokenized) <= question_limit:
        return question_tokenized
    find, pos = False, 0
    candidate_keys = ['什么', '谁', '哪', '几', '何', '多', '是否', '怎么', '嘛', '怎样']
    for idx, token in enumerate(question_tokenized):
        for key in candidate_keys:
            if key in token:
                find, pos = True, idx
                break
        if find:
            break
    if find:
        question = ''.join(
            question_tokenized[max(0, pos - int(question_limit / 2 + 1)):
                               min(pos + int(question_limit / 2 - 1), len(question_tokenized) - 1)])
    else:
        question = ''.join(question_tokenized[len(question_tokenized) - question_limit + 1:])
    return question


def save(filename, obj, message=None):
    if message is not None:
        print("Saving {}...".format(message))
        with open(filename, "w") as fh:
            json.dump(obj, fh)


def load(filename):
    with open(filename, 'r') as fh:
        return json.load(fh)


if __name__ == "__main__":
    """
    在执行process_file函数时直接将数据分为train/dev/test
    当期的test.json和dev.json为同一个文件
    :param config:
    :return:
    """
    flags = tf.flags
    config = flags.FLAGS
    home = os.getcwd()

    pynlpir.open()
    jieba.initialize()

    # 原始数据
    train_file = os.path.join(home, "datasets", "train.json")
    dev_file = os.path.join(home, "datasets", "dev.json")
    flags.DEFINE_string("train_file", train_file, "Train source file")
    flags.DEFINE_string("dev_file", dev_file, "Dev source file")
    # fasttext_file = os.path.join(home, "datasets", "fasttext", "wiki.zh.vec")
    fasttext_file = os.path.join(home, "datasets", "word_vector_result.vec")
    flags.DEFINE_string("fasttext_file", fasttext_file, "Fasttext word embedding source file")

    # 目标数据
    target_dir = "data"
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    flags.DEFINE_string("train_record_file", os.path.join(target_dir, "train.tfrecords"), "Out file for train data")
    flags.DEFINE_string("dev_record_file", os.path.join(target_dir, "dev.tfrecords"), "Out file for dev data")
    flags.DEFINE_string("word_emb_file", os.path.join(target_dir, "word_emb.json"), "Out file for word embedding")
    flags.DEFINE_string("char_emb_file", os.path.join(target_dir, "char_emb.json"), "Out file for char embedding")
    flags.DEFINE_string("train_eval_file", os.path.join(target_dir, "train_eval.json"), "Out file for train eval")
    flags.DEFINE_string("dev_eval_file", os.path.join(target_dir, "dev_eval.json"), "Out file for dev eval")
    flags.DEFINE_string("dev_meta", os.path.join(target_dir, "dev_meta.json"), "Out file for dev meta")
    flags.DEFINE_string("word_dictionary", os.path.join(target_dir, "word_dictionary.json"), "Word dictionary")
    flags.DEFINE_string("char_dictionary", os.path.join(target_dir, "char_dictionary.json"), "Character dictionary")
    flags.DEFINE_integer("glove_dim", 300, "Embedding dimension for Glove")
    flags.DEFINE_integer("char_dim", 300, "Embedding dimension for char")

    # 处理配置
    flags.DEFINE_integer("para_limit", 1000, "Limit length for paragraph")
    flags.DEFINE_integer("ques_limit", 25, "Limit length for question")
    flags.DEFINE_integer("char_limit", 16, "Limit length for character")
    flags.DEFINE_integer("ans_limit", -1, "Limit length for answers")

    # 处理原始数据
    jieba.initialize()
    word_counter, char_counter = Counter(), Counter()
    train_examples, train_eval = process_file(config.train_file, 'train', word_counter, char_counter, config.ques_limit)
    dev_examples, dev_eval = process_file(config.dev_file, 'dev', word_counter, char_counter, config.ques_limit)

    # 处理词向量
    word_emb_mat, word2idx_dict = get_embedding(
        word_counter, "word", emb_file=config.fasttext_file, vec_size=config.glove_dim)
    char_emb_mat, char2idx_dict = get_embedding(
        char_counter, "char", emb_file=config.fasttext_file, vec_size=config.char_dim)
    build_features(config, train_examples, "train", config.train_record_file, word2idx_dict, char2idx_dict)
    dev_meta = build_features(config, dev_examples, "dev", config.dev_record_file, word2idx_dict, char2idx_dict)

    save(config.train_eval_file, train_eval, message="train eval")
    save(config.dev_eval_file, dev_eval, message="dev eval")
    save(config.word_emb_file, word_emb_mat, message="word embedding")
    save(config.char_emb_file, char_emb_mat, message="char embedding")
    save(config.dev_meta, dev_meta, message="dev meta")
    save(config.word_dictionary, word2idx_dict, message="word dictionary")
    save(config.char_dictionary, char2idx_dict, message="char dictionary")
