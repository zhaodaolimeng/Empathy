import json
import os
import tensorflow as tf
import numpy as np
from model import Model
from tqdm import tqdm
from utils import word_tokenize, Bleu, RougeL


def convert_to_features(config, data, word2idx_dict, char2idx_dict):
    example = {}
    context, question = data
    context = context.replace("''", '" ').replace("``", '" ')
    question = question.replace("''", '" ').replace("``", '" ')
    example['context_tokens'] = word_tokenize(context)
    example['ques_tokens'] = word_tokenize(question)
    example['context_chars'] = [list(token) for token in example['context_tokens']]
    example['ques_chars'] = [list(token) for token in example['ques_tokens']]

    para_limit = config.para_limit
    ques_limit = config.ques_limit
    # ans_limit = 100
    char_limit = config.char_limit

    def filter_func(_example):
        return len(_example["context_tokens"]) > para_limit or \
               len(_example["ques_tokens"]) > ques_limit

    if filter_func(example):

        raise ValueError("Context/Questions lengths are over the limit")

    context_idxs = np.zeros([para_limit], dtype=np.int32)
    context_char_idxs = np.zeros([para_limit, char_limit], dtype=np.int32)
    ques_idxs = np.zeros([ques_limit], dtype=np.int32)
    ques_char_idxs = np.zeros([ques_limit, char_limit], dtype=np.int32)

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

    return context_idxs, context_char_idxs, ques_idxs, ques_char_idxs


def get_answer(content, question, session, model, word_dictionary, char_dictionary, config):
    try:
        content_tokenized = word_tokenize(content.replace("''", '" ').replace("``", '" '))
        content = ''.join(content_tokenized[:config.para_limit])

        candidate_keys = ['什么', '谁', '哪', '几', '何', '多', '是否', '怎么', '嘛', '怎样']
        question_tokenized = word_tokenize(question.replace("''", '" ').replace("``", '" '))
        if len(question_tokenized) > config.ques_limit:
            find, pos = False, 0
            for idx, token in enumerate(question_tokenized):
                for key in candidate_keys:
                    if key in token:
                        find, pos = True, idx
                        break
                if find:
                    break
            if find:
                question = ''.join(
                    question_tokenized[max(0, pos - int(config.ques_limit/2 + 1)):
                                       min(pos + int(config.ques_limit/2 - 1), len(question_tokenized)-1)])
            else:
                question = ''.join(question_tokenized[len(question_tokenized) - config.ques_limit + 1:])

        c, ch, q, qh = convert_to_features(config, (content, question), word_dictionary, char_dictionary)
        fd = {'context:0': [c],
              'question:0': [q],
              'context_char:0': [ch],
              'question_char:0': [qh]}

        yp1, yp2 = session.run([model.yp1, model.yp2], feed_dict=fd)
        yp2[0] += 1
        return "".join(content_tokenized[yp1[0]:yp2[0]])
    except:
        print("Error triggered!")
        return None


if __name__ == '__main__':

    flags = tf.flags
    config = flags.FLAGS
    output_dir = "output"
    target_dir = "data"
    train_dir = "train"
    model_name = "empathy"
    dir_name = os.path.join(train_dir, model_name)
    save_dir = os.path.join(dir_name, "model")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    flags.DEFINE_string("eval_output_file", os.path.join(output_dir, "eval_output.json"), "Evaluation output")
    flags.DEFINE_string("eval_input_file", os.path.join(output_dir, "eval_input.json"), "Evaluation input")

    flags.DEFINE_string("save_dir", save_dir, "Directory for saving model")
    flags.DEFINE_string("word_emb_file", os.path.join(target_dir, "word_emb.json"), "Out file for word embedding")
    flags.DEFINE_string("char_emb_file", os.path.join(target_dir, "char_emb.json"), "Out file for char embedding")
    flags.DEFINE_string("word_dictionary", os.path.join(target_dir, "word_dictionary.json"), "Word dictionary")
    flags.DEFINE_string("char_dictionary", os.path.join(target_dir, "char_dictionary.json"), "Character dictionary")

    # 处理配置
    flags.DEFINE_integer("para_limit", 1000, "Limit length for paragraph")
    flags.DEFINE_integer("ques_limit", 25, "Limit length for question")
    flags.DEFINE_integer("char_limit", 16, "Limit length for character")
    flags.DEFINE_integer("ans_limit", -1, "Limit length for answers")
    flags.DEFINE_integer("hidden", 96, "Hidden size")
    flags.DEFINE_integer("glove_dim", 300, "Embedding dimension for Glove")
    flags.DEFINE_integer("char_dim", 300, "Embedding dimension for char")
    flags.DEFINE_integer("num_heads", 1, "Number of heads in self attention")
    flags.DEFINE_float("l2_norm", 3e-7, "L2 norm scale")
    flags.DEFINE_float("decay", 0.995, "Exponential moving average decay")

    print("Start to load datasets ... ")
    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.char_emb_file, "r") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)

    with open(config.word_dictionary, "r") as fh:
        word_dictionary = json.load(fh)
    with open(config.char_dictionary, "r") as fh:
        char_dictionary = json.load(fh)

    with open(config.eval_input_file, 'r') as f:
        jdata = json.load(f)

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True

    model = Model(config, None, word_mat, char_mat, trainable=False, demo=True)
    with model.graph.as_default():

        with tf.Session(config=sess_config) as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(config.save_dir))
            if config.decay < 1.0:
                sess.run(model.assign_vars)

                odata = []
                # bleu_eval, rough_eval = Bleu(), RougeL()
                for a in tqdm(jdata):
                    aid = a['article_id']
                    content = a['article_title'] + a['article_content']
                    o_question = []
                    for q in a['questions']:
                        question = q['question']
                        answer = get_answer(content, question, sess, model,
                                            word_dictionary, char_dictionary, config)
                        o_question.append({
                            'questions_id': q['questions_id'],
                            'answer': answer
                        })
                        # bleu_eval.add_inst(answer, q['answer'])
                        # rough_eval.add_inst(answer, q['answer'])
                    odata.append({
                        'article_id': aid,
                        'questions': o_question
                    })
                # print("Bleu = {}, RoughL = {}".format(bleu_eval.get_score(), rough_eval.get_score()))
                with open(config.eval_output_file, 'w') as outfile:
                    json.dump(odata, outfile, ensure_ascii=False)
