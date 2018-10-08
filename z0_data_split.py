import json
import pynlpir
import jieba
import pandas as pd
from tqdm import tqdm
from utils import word_tokenize


dataset_dir = 'datasets/'


def build_word_vectors(infile_name, outfile_name):
    print('building word vectors...')
    pynlpir.open()
    jieba.initialize()

    df = pd.read_json(infile_name)
    with open(outfile_name, 'w') as f:
        for content in tqdm(df.article_content):
            f.write(' '.join(word_tokenize(content)))


if __name__ == '__main__':

    # 计算向量
    build_word_vectors(dataset_dir + 'question.json', dataset_dir + 'word_vector.txt')

    # 将问题切分为两部分
    print('Split dataset...')
    with open(dataset_dir + "question.json", "r") as fh:
        source = json.load(fh)

        article_cnt = len(source)
        train_json = source[:-int(len(source) / 10)]
        dev_json = source[-int(len(source) / 10):]

        with open(dataset_dir + "train.json", "w") as fw:
            json.dump(train_json, fw)
        with open(dataset_dir + "dev.json", "w") as fw:
            json.dump(dev_json, fw)


