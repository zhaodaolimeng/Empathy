import tensorflow as tf
import ujson as json
import numpy as np
from tqdm import tqdm
import os

from model import Model
from utils import RougeL, Bleu


def get_record_parser(config):
    def parse(example):
        para_limit = config.para_limit
        ques_limit = config.ques_limit
        char_limit = config.char_limit
        features = tf.parse_single_example(example,
                                           features={
                                               "context_idxs": tf.FixedLenFeature([], tf.string),
                                               "ques_idxs": tf.FixedLenFeature([], tf.string),
                                               "context_char_idxs": tf.FixedLenFeature([], tf.string),
                                               "ques_char_idxs": tf.FixedLenFeature([], tf.string),
                                               "y1": tf.FixedLenFeature([], tf.string),
                                               "y2": tf.FixedLenFeature([], tf.string),
                                               "id": tf.FixedLenFeature([], tf.int64)
                                           })
        context_idxs = tf.reshape(tf.decode_raw(features["context_idxs"], tf.int32), [para_limit])
        ques_idxs = tf.reshape(tf.decode_raw(features["ques_idxs"], tf.int32), [ques_limit])
        context_char_idxs = tf.reshape(tf.decode_raw(features["context_char_idxs"], tf.int32), [para_limit, char_limit])
        ques_char_idxs = tf.reshape(tf.decode_raw(features["ques_char_idxs"], tf.int32), [ques_limit, char_limit])
        y1 = tf.reshape(tf.decode_raw(features["y1"], tf.float32), [para_limit])
        y2 = tf.reshape(tf.decode_raw(features["y2"], tf.float32), [para_limit])
        qa_id = features["id"]
        return context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, y1, y2, qa_id
    return parse


def get_batch_dataset(record_file, parser, config):
    num_threads = tf.constant(config.num_threads, dtype=tf.int32)
    dataset = tf.data.TFRecordDataset(record_file).map(
        parser, num_parallel_calls=num_threads).shuffle(config.capacity).repeat()
    if config.is_bucket:
        buckets = [tf.constant(num) for num in range(*config.bucket_range)]

        def key_func(context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, y1, y2, qa_id):
            c_len = tf.reduce_sum(tf.cast(tf.cast(context_idxs, tf.bool), tf.int32))
            t = tf.clip_by_value(buckets, 0, c_len)
            return tf.argmax(t)

        def reduce_func(key, elements):
            return elements.batch(config.batch_size)

        dataset = dataset.apply(tf.contrib.data.group_by_window(
            key_func, reduce_func, window_size=5 * config.batch_size)).shuffle(len(buckets) * 25)
    else:
        dataset = dataset.batch(config.batch_size)
    return dataset


def get_dataset(record_file, parser, config):
    num_threads = tf.constant(config.num_threads, dtype=tf.int32)
    dataset = tf.data.TFRecordDataset(record_file).map(
        parser, num_parallel_calls=num_threads).repeat().batch(config.batch_size)
    return dataset


def evaluate_batch(model, num_batches, eval_file, sess, data_type, handle, str_handle):
    """
    验证程序
    :param model:
    :param num_batches:
    :param eval_file: 验证文件，例如train_eval.json等
    :param sess:
    :param data_type:
    :param handle:
    :param str_handle:
    :return:
    """
    answer_dict = {}
    losses = []
    for _ in tqdm(range(1, num_batches + 1)):
        qa_id, loss, yp1, yp2, = sess.run(
            [model.qa_id, model.loss, model.yp1, model.yp2], feed_dict={handle: str_handle})
        answer_dict_, _ = convert_tokens(eval_file, qa_id.tolist(), yp1.tolist(), yp2.tolist())
        answer_dict.update(answer_dict_)
        losses.append(loss)
    loss = np.mean(losses)
    metrics = evaluate(eval_file, answer_dict)
    metrics["loss"] = loss
    loss_sum = tf.Summary(value=[tf.Summary.Value(tag="{}/loss".format(data_type), simple_value=metrics["loss"]), ])
    f1_sum = tf.Summary(value=[tf.Summary.Value(tag="{}/bleu".format(data_type), simple_value=metrics["bleu"]), ])
    em_sum = tf.Summary(value=[tf.Summary.Value(tag="{}/rouge".format(data_type), simple_value=metrics["rouge"]), ])
    return metrics, [loss_sum, f1_sum, em_sum]


def convert_tokens(eval_file, qa_id, pp1, pp2):
    answer_dict = {}
    remapped_dict = {}
    for qid, p1, p2 in zip(qa_id, pp1, pp2):
        context = eval_file[str(qid)]["context"]
        spans = eval_file[str(qid)]["spans"]
        uuid = eval_file[str(qid)]["uuid"]
        start_idx = spans[p1][0]
        end_idx = spans[p2][1]
        answer_dict[str(qid)] = context[start_idx: end_idx]
        remapped_dict[uuid] = context[start_idx: end_idx]
    return answer_dict, remapped_dict


def evaluate(eval_file, answer_dict):
    rouge_eval, bleu_eval = RougeL(), Bleu()
    for key, value in answer_dict.items():
        ground_truths = eval_file[key]["answers"][0]
        prediction = value
        rouge_eval.add_inst(prediction, ground_truths)
        bleu_eval.add_inst(prediction, ground_truths)
    return {'bleu': bleu_eval.get_score(), 'rouge': rouge_eval.get_score()}


if __name__ == '__main__':

    flags = tf.flags
    config = flags.FLAGS
    home = os.getcwd()

    train_file = os.path.join(home, "datasets", "junshi", "train.json")
    dev_file = os.path.join(home, "datasets", "junshi", "dev.json")

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

    # 模型训练
    train_dir = "train"
    model_name = "empathy"
    dir_name = os.path.join(train_dir, model_name)
    log_dir = os.path.join(dir_name, "event")
    save_dir = os.path.join(dir_name, "model")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    flags.DEFINE_string("target_dir", target_dir, "Target directory for out data")
    flags.DEFINE_string("log_dir", log_dir, "Directory for tf event")
    flags.DEFINE_string("save_dir", save_dir, "Directory for saving model")

    flags.DEFINE_integer("capacity", 15000, "Batch size of dataset shuffle")
    flags.DEFINE_integer("num_threads", 4, "Number of threads in input pipeline")
    flags.DEFINE_boolean("is_bucket", False, "build bucket batch iterator or not")
    flags.DEFINE_list("bucket_range", [40, 401, 40], "the range of bucket")

    # 模型参数
    flags.DEFINE_integer("para_limit", 1000, "Limit length for paragraph")
    flags.DEFINE_integer("ques_limit", 25, "Limit length for question")
    flags.DEFINE_integer("ans_limit", -1, "Limit length for answers")
    flags.DEFINE_integer("char_limit", 16, "Limit length for character")
    flags.DEFINE_integer("word_count_limit", -1, "Min count for word")
    flags.DEFINE_integer("char_count_limit", -1, "Min count for char")

    flags.DEFINE_integer("batch_size", 8, "Batch size")
    flags.DEFINE_integer("num_steps", 80000, "Number of steps")
    flags.DEFINE_integer("checkpoint", 1000, "checkpoint to save and evaluate the model")
    flags.DEFINE_integer("period", 100, "period to save batch loss")
    flags.DEFINE_integer("val_num_batches", 150, "Number of batches to evaluate the model")
    flags.DEFINE_float("dropout", 0.1, "Dropout prob across the layers")
    flags.DEFINE_float("grad_clip", 5.0, "Global Norm gradient clipping rate")
    flags.DEFINE_float("learning_rate", 0.00001, "Learning rate")
    flags.DEFINE_float("decay", 0.995, "Exponential moving average decay")
    flags.DEFINE_float("l2_norm", 3e-7, "L2 norm scale")
    flags.DEFINE_integer("hidden", 96, "Hidden size")
    flags.DEFINE_integer("num_heads", 1, "Number of heads in self attention")
    flags.DEFINE_integer("early_stop", 10, "Checkpoints for early stop")

    # 读入原始数据
    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.char_emb_file, "r") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.train_eval_file, "r") as fh:
        train_eval_file = json.load(fh)
    with open(config.dev_eval_file, "r") as fh:
        dev_eval_file = json.load(fh)
    with open(config.dev_meta, "r") as fh:
        meta = json.load(fh)

    dev_total = meta["total"]
    print("Building model...")
    parser = get_record_parser(config)
    graph = tf.Graph()
    with graph.as_default() as g:
        train_dataset = get_batch_dataset(config.train_record_file, parser, config)
        dev_dataset = get_dataset(config.dev_record_file, parser, config)
        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
        train_iterator = train_dataset.make_one_shot_iterator()
        dev_iterator = dev_dataset.make_one_shot_iterator()
        model = Model(config, iterator, word_mat, char_mat, graph=g)

        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True

        # loss_save = 100.0
        patience, best_bleu, best_rouge = 0, 0., 0.
        with tf.Session(config=sess_config) as sess:
            writer = tf.summary.FileWriter(config.log_dir)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            train_handle = sess.run(train_iterator.string_handle())
            dev_handle = sess.run(dev_iterator.string_handle())
            if os.path.exists(os.path.join(config.save_dir, "checkpoint")):
                saver.restore(sess, tf.train.latest_checkpoint(config.save_dir))
            global_step = max(sess.run(model.global_step), 1)

            for _ in tqdm(range(global_step, config.num_steps + 1)):
                global_step = sess.run(model.global_step) + 1
                loss, train_op = sess.run([model.loss, model.train_op], feed_dict={
                    handle: train_handle, model.dropout: config.dropout})
                if global_step % config.period == 0:
                    loss_sum = tf.Summary(value=[tf.Summary.Value(tag="model/loss", simple_value=loss), ])
                    writer.add_summary(loss_sum, global_step)
                if global_step % config.checkpoint == 0:
                    _, summ = evaluate_batch(
                        model, config.val_num_batches, train_eval_file, sess, "train", handle, train_handle)
                    for s in summ:
                        writer.add_summary(s, global_step)
                    metrics, summ = evaluate_batch(
                        model, dev_total // config.batch_size + 1, dev_eval_file, sess, "dev", handle, dev_handle)

                    dev_bleu = metrics["bleu"]
                    dev_rouge = metrics["rouge"]
                    if dev_bleu < best_bleu and dev_rouge < best_rouge:
                        patience += 1
                        if patience > config.early_stop:
                            break
                    else:
                        patience = 0
                        best_rouge = max(best_rouge, dev_rouge)
                        best_bleu = max(best_bleu, dev_bleu)

                    for s in summ:
                        writer.add_summary(s, global_step)
                    writer.flush()
                    filename = os.path.join(
                        config.save_dir, "model_{}.ckpt".format(global_step))
                    saver.save(sess, filename)
