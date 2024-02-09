IMPORTS = '''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
import pathlib
import random
import re
from collections import Counter

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization, 
                                    Embedding, 
                                    Input, 
                                    LSTM, 
                                    Dense, 
                                    Permute, 
                                    Multiply, 
                                    Lambda, 
                                    Concatenate, 
                                    Embedding, 
                                    RepeatVector
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

from gensim.parsing import preprocessing

from aws.aws_funcs import upload_to_s3
'''

VISUALIZATION = '''
def plot_loss_curves(history, metrics='accuracy'):
  """
  Return separate loss curves for training and validation metrics.
  """
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  accuracy = history.history[metrics]
  val_accuracy = history.history[f'val_{metrics}']

  epochs = range(1, 1 + len(history.history['loss']))

  fig, ax = plt.subplots(1, 2, figsize=(15, 4), dpi=200)

  ax[0].plot(epochs, loss, label='training_loss')
  ax[0].plot(epochs, val_loss, label='val_loss')
  ax[0].set_title('loss')
  ax[0].set_xlabel('epochs')
  ax[0].legend()

  ax[1].plot(epochs, accuracy, label=f'training_{metrics}')
  ax[1].plot(epochs, val_accuracy, label=f'val_{metrics}')
  ax[1].set_title(metrics)
  ax[1].set_xlabel('epochs')
  ax[1].legend()
'''

PREPROCESSING = '''
def preprocess_tweet_text(tweet_text: str) -> str:
    preprocessed_text = preprocessing.strip_non_alphanum(s=tweet_text)
    preprocessed_text = preprocessing.strip_multiple_whitespaces(s=preprocessed_text)
    preprocessed_text = preprocessing.strip_punctuation(s=preprocessed_text)
    preprocessed_text = preprocessing.strip_numeric(s=preprocessed_text)

    preprocessed_text = preprocessing.stem_text(text=preprocessed_text)
    preprocessed_text = preprocessing.remove_stopwords(s=preprocessed_text)

    return preprocessed_text

def to_dataset(features, labels, BATCH_SIZE=32):
    dataset_features = tf.data.Dataset.from_tensor_slices(features)
    dataset_labels = tf.data.Dataset.from_tensor_slices(labels)
    dataset = tf.data.Dataset.zip((dataset_features, dataset_labels))
    dataset = dataset.shuffle(buffer_size=len(features)).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset
'''

MODEL_METRICS = '''
def calculate_results(y_true, y_pred):
  """
  Calculates model accuracy, precision, recall and f1 score of a binary classification model.

  Args:
  ----
  y_true = true labels in the form of a 1D array
  y_pred = predicted label in the form of a 1D array

  Returns a dictionary of accuracy, precision, recall and f1-score between y_true and y_pred.
  """
  model_accuracy = accuracy_score(y_true, y_pred) * 100 
  model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
  return {
      "accuracy": model_accuracy,
      "precision": model_precision,
      "recall": model_recall,
      "f1": model_f1,
  }
'''

RESULT_FUNC = '''
    def get_df_results():
        df = pd.DataFrame()
        names = ['LSTM', 'Bidirectional', 'Conv1D + Bidirectional', 'LSTM + Attention']
        models = [model_1, model_2, model_3, model_4]
        temp = []
        for model in models:
            preds = tf.argmax(model.predict(test_sentences_vectorized), axis=1)
            temp.append(calculate_results(y_true=test_labels, y_pred=preds))
        temp = pd.DataFrame(temp)
        temp.index = names
        return temp

    def how_is_confused(model, data, labels):
        preds = np.argmax(model.predict(data), axis=1)
        
        wrong_preds_idx = np.where(preds != labels)
        wrong_preds_labels = preds[wrong_preds_idx]
        df_wrong_total = pd.Series(Counter(wrong_preds_labels))
        
        df_total = labels.value_counts()
        final = pd.concat([df_total, df_wrong_total], axis=1)
        final.columns = ['Total', 'Wrong']
        return final

    def where_is_confused(target, labels, preds):
        true_target_preds = np.where(target == labels)
        return pd.Series(Counter(preds[true_target_preds]))

    def create_confusion(model, data, labels):
        preds = np.argmax(model.predict(data), axis=1)
        series_dict = {i: where_is_confused(i, labels, preds) for i in range(len(id2label))}
        df_confusion = pd.DataFrame(series_dict).fillna(0)
        df_confusion.index, df_confusion.columns = id2label.values(), id2label.values()
        return df_confusion

'''

AMAZON_S3 = '''
# for plots
path_to_local_plots = '***'
def upload_to_aws_png(file_name):
    upload_to_s3(f'{path_to_local_plots}{file_name}.png', f'***/{file_name}.png')

def upload_plot(plot_name):
    plt.savefig(f'{path_to_local_plots}{plot_name}.png', dpi=300)
    upload_to_aws_png(plot_name)

# for csv
path_to_local_csv = '***'
def upload_to_aws_csv(file_name):
    upload_to_s3(f'{path_to_local_csv}{file_name}.csv', f'***/{file_name}.csv')

def upload_csv(df, csv_name):
    df.to_csv(f'{path_to_local_csv}{csv_name}.csv')
    upload_to_aws_csv(csv_name)
'''

DISTRIBUTION_LABELS = '''
    plt.figure(figsize=(12, 5), dpi=350)
    sns.countplot(data=emoji_df, y='Tweets');
'''

MAX_MIN_RATION = '''
    emoji_df.value_counts().max() / emoji_df.value_counts().min()
    ratio_max_others = {}
    for idx in emoji_df.value_counts().index:
        ratio_max_others[f'Sob to {idx[0]}'] = (emoji_df.value_counts().loc[["sob"]] / emoji_df.value_counts().loc[idx]).to_list()[0]
    
    plt.figure(figsize=(12, 5), dpi=350)
    plt.xticks(rotation=45)
    plt.bar(ratio_max_others.keys(), ratio_max_others.values());
'''


TWEETS_LENGTH_DISTRIBUTION = '''
    plt.figure(figsize=(12, 5), dpi=150)
    plt.hist(tweets_df['Tweets'].apply(lambda x: len(x)), bins= 50);
    plt.xlabel('Tweet Length')
    plt.ylabel('Count')
'''

PREPROCESS_TWEETS = '''
    tweets_df['tweets_preprocessed'] = tweets_df['Tweets'].apply(preprocess_tweet_text)
'''

TWEETS_LENGTH_SYMBOLS_DISTRIBUTION = '''
    plt.figure(figsize=(15, 5), dpi=350)
    sns.histplot(data=tweets_df, x=tweets_df['tweets_preprocessed'].apply(len), kde=True)
    plt.xlabel('Length of Tweet in Symbols', labelpad=10)
    plt.title('Distribution of Length of Tweets in Symbols', pad=15);
'''


TWEETS_LENGTH_WORDS_DISTRIBUTION = '''
    plt.figure(figsize=(15, 5), dpi=350)
    sns.histplot(data=tweets_df, x=tweets_df['tweets_preprocessed'].apply(lambda x: x.split(' ')).apply(len))
    plt.xlabel('Length of Tweet in Words', labelpad=10)
    plt.title('Distribution of Length of Tweets in Words', pad=15);
'''

ENCODER = '''
    encoder = OrdinalEncoder()
    emoji_df['encoded'] = encoder.fit_transform(emoji_df[['Tweets']])
'''

TRAIN_TEST_SPLIT = '''
    tweets_df_final = tweets_df.drop('Tweets', axis=1)
    df = pd.merge(emoji_df, tweets_df_final, left_index=True, right_index=True).rename(columns={'encoded': 'emoji', 'tweets_preprocessed': 'tweets'}).drop('Tweets', axis=1)
    
    train_sentences, test_sentences, train_labels, test_labels = train_test_split(
    df['tweets'],
    df['emoji'],
    test_size=0.2,
    random_state=42)
'''

TEXT_VECTORIZING = '''
    text_vectorizer = TextVectorization(
    max_tokens=10_000,
    standardize='lower_and_strip_punctuation',
    split='whitespace',
    ngrams=None,
    output_mode='int',
    output_sequence_length=30,
    pad_to_max_tokens=True)
    
    text_vectorizer.adapt(train_sentences)

    sample_sentence = "There's a flood in my street"
    text_vectorizer([sample_sentence])

    # Output: <tf.Tensor: shape=(1, 30), dtype=int64, numpy=
    #           array([[   1,    1, 3554,    1,    1,  936,    0,    0,    0,    0,    0,
    #                      0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    #                      0,    0,    0,    0,    0,    0,    0,    0]])>
'''


VOCABULARY = '''
    words_in_vocab = text_vectorizer.get_vocabulary()
    top_10_words = words_in_vocab[:10]
    bottom_10_words = words_in_vocab[-10:]
    print(f'Most common words in vocab: {top_10_words}')
    print(f'Least common words in vocab: {bottom_10_words}')
    # Output: 
    # Most common words in vocab: ['', '[UNK]', 't', 'bet', 'http', 'rt', 'starbuck', 's', 'thi', 'subwai']
    # Least common words in vocab: ['soror', 'sorchahollowai', 'sophiaeatspizza', 'sopandeb', 'soooooooooooooo', 'sooooooooo', 'soonyoung', 'soojung', 'sony', 'soniabubla']
'''

MODEL_0 = '''
    model_0 = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', MultinomialNB())])

    model_0.fit(train_sentences, train_labels)


    baseline_score = model_0.score(test_sentences, test_labels)
    f'Baseline Model achieves an accuracy of: {baseline_score * 100:.2f}%'

    # Output: 'Baseline Model achieves an accuracy of: 44.20%'
'''


TENSORFLOW_MODElS = '''
    train_sentences_vectorized = text_vectorizer(train_sentences)
    test_sentences_vectorized = text_vectorizer(test_sentences)

    dataset_train = to_dataset(train_sentences_vectorized, train_labels)
    dataset_test = to_dataset(test_sentences_vectorized, test_labels)

    earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
'''

EMBEDDINGS = '''
    embedding = layers.Embedding(
    input_dim=10_000,
    output_dim=128,
    embeddings_initializer='uniform',
    input_length=30)
'''

MODEL_1 = '''
    inputs = layers.Input(shape=(30,), dtype="int32")
    x = embedding(inputs)
    x = layers.LSTM(64, activation="tanh")(x)
    outputs = layers.Dense(10, activation="softmax")(x)
    model_1 = tf.keras.Model(inputs, outputs, name="model_1_LSTM")

    model_1.compile(loss="SparseCategoricalCrossentropy",
                    optimizer=tf.keras.optimizers.Adam(),
                    metrics=["accuracy"])

    history_1 = model_1.fit(dataset_train,
                            epochs=12,
                            validation_data=dataset_test,
                            callbacks=[reduce_lr])
'''



MODEL_2 = '''
    inputs = layers.Input(shape=(30,), dtype="int32")
    x = embedding(inputs)
    x = layers.Bidirectional(layers.LSTM(64, activation="tanh"))(x)
    outputs = layers.Dense(10, activation="softmax")(x)
    model_2 = tf.keras.Model(inputs, outputs, name="model_2_Bidirectional")

    model_2.compile(loss="SparseCategoricalCrossentropy",
                    optimizer=tf.keras.optimizers.Adam(),
                    metrics=["accuracy"])

    history_2 = model_2.fit(dataset_train,
                            epochs=12,
                            validation_data=dataset_test,
                            callbacks=[reduce_lr])
'''


MODEL_3 = '''
    inputs = layers.Input(shape=(30,), dtype="int32")
    x = embedding(inputs)
    x = layers.Conv1D(128, 5, activation="relu")(x)
    x = layers.MaxPooling1D(5)(x)
    x = layers.Bidirectional(layers.GRU(64))(x)
    outputs = layers.Dense(10, activation="softmax")(x)
    model_3 = tf.keras.Model(inputs, outputs, name="model_3_Conv1D_Bidirectional")

    model_3.compile(loss="SparseCategoricalCrossentropy",
                    optimizer=tf.keras.optimizers.Adam(),
                    metrics=["accuracy"])

    history_3 = model_3.fit(dataset_train,
                            epochs=12,
                            validation_data=dataset_test,
                            callbacks=[reduce_lr])
'''


MODEL_4 = '''
    def attention_layer(inputs, SINGLE_ATTENTION_VECTOR=False):
        num_tokens = K.int_shape(inputs)[1]
        embed_dim = K.int_shape(inputs)[2]
        a = Permute((2, 1))(inputs)
        a = Dense(num_tokens, activation='softmax')(a)
        if SINGLE_ATTENTION_VECTOR:
            a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
            a = RepeatVector(embed_dim)(a)
        a_probs = Permute((2, 1), name='attention_vec')(a)
        return Multiply()([inputs, a_probs])


    inputs = Input(shape=(30, ), dtype='int32')
    x = embedding(inputs)  
    attention_mul = attention_layer(x)
    lstm_out = LSTM(32, return_sequences=False)(attention_mul)
    output = Dense(10, activation='softmax')(lstm_out)
    model_4 = Model(inputs=[inputs], outputs=output)

    model_4.compile(loss="sparse_categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

    history_4 = model_4.fit(dataset_train,
                            epochs=15,
                            validation_data=dataset_test,
                            callbacks=[reduce_lr])

'''

RESULTS = '''
    names = ['LSTM', 'Bidirectional', 'Conv1D + Bidirectional', 'LSTM + Attention']
    models = [model_1, model_2, model_3, model_4]
    results_df = get_df_results(models, names)
'''