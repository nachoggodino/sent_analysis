import re
import xml.etree.ElementTree as ET

import numpy
import pandas as pd
from sklearn import preprocessing

from src.config import *


def fetch_data(dataset):
    if dataset == 'tass2019':
        return read_tass2019_files()
    elif dataset == 'coah':
        return read_coah_files()


def parse_xml(filepath):
    parser = ET.XMLParser(encoding='UTF-8')
    tree = ET.parse(filepath, parser=parser)
    return tree


def get_dataframe_from_coah(data):
    id, rank, abstract, review = [], [], [], []
    for tweet in data.iter('hotel_review'):
        for element in tweet.iter():
            if element.tag == 'id':
                id.append(element.text)
            elif element.tag == 'rank':
                rank.append(element.text)
            elif element.tag == 'abstract':
                abstract.append(element.text)
            elif element.tag == 'review':
                review.append(element.text)

    result_df = pd.DataFrame()
    result_df['id'] = id
    result_df['sentiment'] = rank
    result_df['abstract'] = abstract
    result_df['review'] = review
    return result_df


def get_dataframe_from_spamore(data):
    rank, summary, body = [], [], []
    for tweet in data.iter('review'):
        rank.append(tweet.find('review').attrib['rank'])
        for element in tweet.iter():
            if element.tag == 'summary':
                summary.append(element.text)
            elif element.tag == 'body':
                body.append(element.text)

    result_df = pd.DataFrame()
    result_df['sentiment'] = rank
    result_df['summary'] = summary
    result_df['body'] = body
    return result_df


def get_dataframe_from_tass2019(data, encode_label=True):
    tweet_id, user, content, day_of_week, month, hour, lang, sentiment = [], [], [], [], [], [], [], []
    for tweet in data.iter('tweet'):
        for element in tweet.iter():
            if element.tag == 'tweetid':
                tweet_id.append(element.text)
            elif element.tag == 'user':
                user.append(element.text)
            elif element.tag == 'content':
                content.append(element.text)
            elif element.tag == 'date':
                day_of_week.append(element.text[:3])
                month.append(element.text[4:7])
                hour.append(element.text[11:13])
            elif element.tag == 'lang':
                lang.append(element.text)
            elif element.tag == 'value':
                if element.text == 'NONE' or element.text == 'NEU':
                    sentiment.append(element.text)
                else:
                    sentiment.append(element.text)

    result_df = pd.DataFrame()
    result_df['tweet_id'] = tweet_id
    # result_df['user'] = user
    result_df['content'] = content
    # result_df['lang'] = lang
    result_df['day_of_week'] = day_of_week
    result_df['month'] = month
    result_df['hour'] = hour

    if encode_label:
        encoder = preprocessing.LabelEncoder()
        sentiment = encoder.fit_transform(sentiment)

    result_df['sentiment'] = sentiment

    return result_df


def get_dataframe_from_ftx_format(lang, folder, set='', train_plus_dev=False):
    result = pd.DataFrame()
    sets = list()
    if train_plus_dev:
        sets.append('train')
        sets.append('dev')
    else:
        sets.append(set)
    for mode in sets:
        with open('dataset/{}/intertass_{}_{}.txt'.format(folder, lang, mode), 'r') as file:
            dataframe = pd.DataFrame()
            lines = file.readlines()
            sentiment_list, content_list = list(), list()
            for line in lines:
                sentiment, content = line.split(' ', 1)
                if re.search('[0-3]', sentiment[-1:]) and content != '':
                    sentiment_list.append(int(sentiment[-1:]))
                    content_list.append(content.replace('\n', '').strip())
            dataframe['content'] = content_list
            dataframe['sentiment'] = sentiment_list
            result = pd.concat([result, dataframe], ignore_index=True).reset_index(drop=True)
    return result


def read_tass2019_files():
    name = 'tass2019'
    train_data = pd.read_csv('../dataset/csv/{}/{}_train.csv'.format(name, name), encoding='utf-8', sep='\t')
    dev_data = pd.read_csv('../dataset/csv/{}/{}_dev.csv'.format(name, name), encoding='utf-8', sep='\t')
    test_data = pd.read_csv('../dataset/csv/{}/{}_test.csv'.format(name, name), encoding='utf-8', sep='\t')

    if B_REDUCED:
        train_data['sentiment'] = train_data['sentiment'].transform(lambda x: reduce_labels('tass2019', x))
        dev_data['sentiment'] = dev_data['sentiment'].transform(lambda x: reduce_labels('tass2019', x))
        test_data['sentiment'] = test_data['sentiment'].transform(lambda x: reduce_labels('tass2019', x))
    return train_data, dev_data, test_data, train_data['sentiment'].unique()


def read_coah_files():
    name = 'coah'
    coah_data = pd.read_csv('../dataset/csv/{}/{}.csv'.format(name, name), encoding='utf-8', sep='\t')
    label_dict = coah_data['sentiment'].unique()

    if B_REDUCED:
        coah_data['sentiment'] = coah_data['sentiment'].transform(lambda x: reduce_labels(name, x))
    train_data, dev_data, test_data = dataframe_split(coah_data)
    return train_data, dev_data, test_data, train_data['sentiment'].unique()


def reduce_labels(dataset, x):
    if dataset == 'tass2019':
        return x - 1 if x > 1 else x
    elif dataset == 'coah':
        return 0 if x < 3 else (1 if x == 3 else 2)


def dataframe_split(dataframe):
    numpy.random.seed(SHUFFLE_SEED)
    train, dev, test = numpy.split(dataframe.sample(frac=1),
                       [int(SPLIT_SEP_1*len(dataframe)), int(SPLIT_SEP_2*len(dataframe))])
    return train.reset_index(drop=True), dev.reset_index(drop=True), test.reset_index(drop=True)


if __name__ == '__main__':
    print(get_dataframe_from_spamore(parse_xml('../dataset/corpus/corpus_spamore.xml')))
