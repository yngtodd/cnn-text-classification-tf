#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
from tensorflow.contrib import learn

from cnntext.text_cnn import TextCNN
from cnntext.data.dataloader import batch_iter, load_data_and_labels

from hammer.meters import OptimizationHistory
from sklearn.model_selection import train_test_split

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("train_positive_data_file", "../data/imdb/train/imdb_train.pos", "Train source for the positive data.")
tf.flags.DEFINE_string("train_negative_data_file", "../data/imdb/train/imdb_train.neg", "Train source for the negative data.")
tf.flags.DEFINE_string("test_positive_data_file", "../data/imdb/test/imdb_test.pos", "Test source for the positive data.")
tf.flags.DEFINE_string("test_negative_data_file", "../data/imdb/test/imdb_test.neg", "Test source for the negative data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 30, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS


def preprocess(downsample=0.0):
    # Data Preparation
    # ==================================================

    # Load data
    print("Loading data...")
    x_train, y_train = load_data_and_labels(FLAGS.train_positive_data_file, FLAGS.train_negative_data_file)
    x_dev, y_dev = load_data_and_labels(FLAGS.test_positive_data_file, FLAGS.test_negative_data_file)

    corpus = x_train + x_dev

    # Build vocabulary
    max_document_length = max([len(x.split(" ")) for x in corpus])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    processed_vocab = vocab_processor.fit(corpus)

    x_train = np.array(list(vocab_processor.transform(x_train)))
    x_dev = np.array(list(vocab_processor.transform(x_dev)))

    # Cut down the training set size to check sample size effect
    if downsample > 0.0:
       x_train, _, y_train, _ = train_test_split(
            x_train, y_train, test_size=downsample, random_state=42
       )

    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    return x_train, y_train, vocab_processor, x_dev, y_dev


def train(x_train, y_train, vocab_processor, x_dev, y_dev, history):
    # Training
    # ==================================================
    
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement
        )

        # Try to squeeze eval in memory
        session_conf.gpu_options.per_process_gpu_memory_fraction = 0.8
        session_conf.gpu_options.allow_growth=True

        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())


            def train_step(x_batch, y_batch, history):
                """
                A single training step
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()

                # Record history for Hammer
                history.minibatch_loss_meter.add_train_loss(loss)
                history.top1_train.update(accuracy, FLAGS.batch_size)

                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)


            def dev_step(x_batch, y_batch, history, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: 1.0
                }

                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict
                )

                time_str = datetime.datetime.now().isoformat()

                # Record validation metrics for Hammer.
                history.minibatch_loss_meter.add_valid_loss(loss)
                history.top1_valid.update(accuracy, FLAGS.batch_size)

                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)


            # Generate batches
            batches = batch_iter(
                list(zip(x_train, y_train)),
                FLAGS.batch_size, 
                FLAGS.num_epochs
            )

            test_batches - batch_iter(
                list(zip(x_dev, y_dev)),
                FLAGS.batch_size,
                1
            )

            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch, history)
                current_step = tf.train.global_step(sess, global_step)

                if current_step % FLAGS.evaluate_every == 0:
                    with tf.device('/cpu:0'):
                        for test_batch in test_batches:
                            print("\nEvaluation:")
                            dev_step(x_dev, y_dev, history, writer=dev_summary_writer)
                            print("")

                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))


def main(argv=None):
    dataloader_info = {
        'shuffle': True,
        'num_workers': 1
    }

    notes = {
        'sample_size': 'Training on full dataset'
    }
    
    num_ranks = 30
    for i in range(num_ranks):
        history = OptimizationHistory(
            savepath='/home/ygx/src/cnn-text-classification-tf/experiments/imdb',
            experiment_name='yoonkim_tf_imdb',
            device='gpu',
            dataloader_info=dataloader_info,
            notes=notes,
            rank=i
        )

        x_train, y_train, vocab_processor, x_dev, y_dev = preprocess()
        train(x_train, y_train, vocab_processor, x_dev, y_dev, history)

        history.time_meter.stop_timer()
        history.record_history()
        history.reset_meters()
        history.save()


if __name__ == '__main__':
    tf.app.run()
