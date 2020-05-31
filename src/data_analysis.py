import xml.etree.ElementTree as ET
import re
import nltk
import unidecode
import spacy
from textacy import keyterms
import hunspell
import utils
import tweet_preprocessing

from re import finditer
import feature_extraction

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import pandas, xgboost, numpy, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers

import nltk
nltk.download('punkt')
from nltk.tokenize.treebank import TreebankWordDetokenizer

from imblearn.over_sampling import RandomOverSampler

from collections import Counter

LANGUAGE_CODE = 'all'
dictionary = hunspell.HunSpell('./dictionaries/es_ANY.dic', "./dictionaries/es_ANY.aff")

emoji_pattern = re.compile("[" u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)

data_path = "./dataset/xml/"
test_path = "./codalab/DATASETS/public_data_task1/"
parser_dev = ET.XMLParser(encoding='utf-8')
parser_train = ET.XMLParser(encoding='utf-8')

# tree_dev = ET.parse(data_path + "intertass_" + LANGUAGE_CODE + "_dev.xml", parser=parser_dev)
# tree_train = ET.parse(data_path + "intertass_" + LANGUAGE_CODE + "_train.xml", parser=parser_train)


def read_files(sLang, bStoreFiles=False):
    train_data = pandas.DataFrame()
    dev_data = pandas.DataFrame()
    test_data = pandas.DataFrame()
    valid_data = pandas.DataFrame()

    if bStoreFiles:
        train_data = utils.get_dataframe_from_xml(utils.parse_xml('./dataset/xml/intertass_{}_train.xml'.format(sLang)))
        dev_data = utils.get_dataframe_from_xml(utils.parse_xml('./dataset/xml/intertass_{}_dev.xml'.format(sLang)))

        train_data.to_csv('./dataset/csv/intertass_{}_train.csv'.format(sLang), encoding='utf-8', sep='\t')
        dev_data.to_csv('./dataset/csv/intertass_{}_dev.csv'.format(sLang), encoding='utf-8', sep='\t')

    else:

        train_data = pandas.read_csv('./dataset/csv/intertass_{}_train.csv'.format(sLang), encoding='utf-8', sep='\t')
        dev_data = pandas.read_csv('./dataset/csv/intertass_{}_dev.csv'.format(sLang), encoding='utf-8', sep='\t')

    valid_data = pandas.read_csv('./dataset/csv/intertass_{}_valid.csv'.format(sLang), encoding='utf-8', sep='\t')
    test_data = pandas.read_csv('./dataset/csv/intertass_{}_test.csv'.format(sLang), encoding='utf-8', sep='\t')

    encoder = preprocessing.LabelEncoder()
    valid_data['sentiment'] = encoder.fit_transform(valid_data['sentiment'])
    test_data['sentiment'] = encoder.transform(test_data['sentiment'])

    return train_data, dev_data, test_data, valid_data


def perform_upsampling(dataframe):
    ros = RandomOverSampler()
    x_resampled, y_resampled = ros.fit_resample(dataframe[['tweet_id', 'content']], dataframe['sentiment'])
    df = pandas.DataFrame(data=x_resampled[0:, 0:], columns=['tweet_id', 'content'])
    df['sentiment'] = y_resampled
    df = df.sample(frac=1).reset_index(drop=True)
    return df


def extract_uppercase_feature(dataframe):
    regex = re.compile(r"\b[A-Z][A-Z]+\b")
    result = []
    for tweet in dataframe:
        result.append(len(regex.findall(tweet)))
    return result


def extract_laughter_feature(dataframe):
    laughter = False
    return [1 if re.search(r"[a-zA-Z]*jaj[a-zA-Z]*", tweet) or re.search(r"[a-zA-Z]*hah[a-zA-Z]*", tweet)
            or re.search(r"[a-zA-Z]*jej[a-zA-Z]*", tweet) or re.search(r"[a-zA-Z]*joj[a-zA-Z]*", tweet)
            or re.search(r"[a-zA-Z]*jij[a-zA-Z]*", tweet) or re.search(r"[a-zA-Z]*lol[a-zA-Z]*", tweet)
            else 0 for tweet in dataframe]


def extract_length_feature(tokenized_dataframe):
    return [len(tweet) for tweet in tokenized_dataframe]


def extract_question_mark_feature(dataframe):
    result = []
    for tweet in dataframe:
        if re.search(r"[/?/]", tweet):
            result.append(1)
        else:
            result.append(0)
    return result


def extract_exclamation_mark_feature(dataframe):
    result = []
    for tweet in dataframe:
        if re.search(r"[/!/]", tweet):
            result.append(1)
        else:
            result.append(0)
    return result


def extract_letter_repetition_feature(dataframe):
    result = []
    for tweet in dataframe:
        if re.search(r"(\w)(\1{2,})", tweet):
            result.append(1)
        else:
            result.append(0)
    return result


def extract_sent_words_feature(tokenized_data, data_feed, sentiment_feed):
    positive_voc, negative_voc = get_sentiment_vocabulary(data_feed, sentiment_feed, 3, 0)
    pos_result = []
    neg_result = []
    neutral_result = []
    none_result = []
    for index, tokenized_tweet in enumerate(tokenized_data):
        pos_count = sum(word in tokenized_tweet for word in positive_voc)
        neg_count = sum(word in tokenized_tweet for word in negative_voc)
        length = len(tokenized_tweet)

        pos_result.append(pos_count/length)
        neg_result.append(neg_count/length)
        neutral_result.append(0 if (pos_count + neg_count) == 0 else 1-(pos_count-neg_count)/(pos_count+neg_count))
        none_result.append(1-(max(neg_count, pos_count)/length))
    return pos_result, neg_result, neutral_result, none_result


def get_sentiment_vocabulary(data, sentiment_feed, positive, negative):
    print("Sentiment Vocabulary Extraction")
    pos_neg_tweets = []
    pos_neg_bool_labels = []
    for index, tweet in enumerate(data):
        sentiment = sentiment_feed[index]
        if sentiment == positive:
            pos_neg_tweets.append(tweet)
            pos_neg_bool_labels.append(True)
        elif sentiment == negative:
            pos_neg_tweets.append(tweet)
            pos_neg_bool_labels.append(False)
    positive_vocabulary, negative_vocabulary = keyterms.most_discriminating_terms(pos_neg_tweets, pos_neg_bool_labels)
    print(positive_vocabulary)
    print(negative_vocabulary)

    pos_df = pandas.read_csv('./lexicons/isol/positivas_mejorada.csv', encoding='latin-1', header=None, names=['words'])
    neg_df = pandas.read_csv('./lexicons/isol/negativas_mejorada.csv', encoding='latin-1', header=None, names=['words'])

    if False:
        return pos_df['words'].array, neg_df['words'].array
    else:
        return positive_vocabulary, negative_vocabulary


def isol_analytics(dataframe, feed, sentiment_feed):
    pos_df = pandas.read_csv('./lexicons/isol/positivas_mejorada.csv', encoding='latin-1', header=None, names=['words'])
    neg_df = pandas.read_csv('./lexicons/isol/negativas_mejorada.csv', encoding='latin-1', header=None, names=['words'])
    pos_voc = pos_df['words'].array
    neg_voc = neg_df['words'].array

    pos_count, neg_count = set(), set()
    for tweet in dataframe:
        pos_count = pos_count.union(set(tweet) & set(pos_voc))
        neg_count = neg_count.union(set(tweet) & set(neg_voc))
    print('Ocurrencias en iSol:     NEG-> {}        POS-> {}'.format(len(neg_count), len(pos_count)))
    print()

    positive_voc, negative_voc = get_sentiment_vocabulary(feed, sentiment_feed, 3, 0)
    pos_count.clear()
    neg_count.clear()
    for tweet in dataframe:
        pos_count = pos_count.union(set(tweet) & set(positive_voc))
        neg_count = neg_count.union(set(tweet) & set(negative_voc))
    print('Ocurrencias en keyterms:     NEG-> {}        POS-> {}'.format(len(neg_count), len(pos_count)))
    print()
    print('Palabras discriminantes que no están en isol:')
    print('POS-> {}'.format(numpy.setdiff1d(positive_voc, pos_voc)))
    print('NEG-> {}'.format(numpy.setdiff1d(negative_voc, neg_voc)))




def camel_case_split(identifier):
    clean_identifier = re.sub('[#]', '', identifier)
    matches = finditer(".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)", clean_identifier)
    return ' '.join([m.group(0) for m in matches])


def libreoffice_processing(tokenized_data):
    print("Libreoffice processing")
    return [[word if dictionary.spell(word) is True else next(iter(dictionary.suggest(word)), word) for word in tweet]
            for tweet in tokenized_data]



def text_preprocessing(data):
    result = data
    result = [tweet.replace('\n', '').strip() for tweet in result]  # Newline and leading/trailing spaces
    result = [emoji_pattern.sub(r'', tweet) for tweet in result]
    result = [re.sub(r"\B#\w+", lambda m: camel_case_split(m.group(0)), tweet) for tweet in result]  # Hashtag
    result = [tweet.lower() for tweet in result]  # Tweet to lowercase
    result = [re.sub(r"^.*http.*$", 'http', tweet) for tweet in result]  # Remove all http contents
    result = [re.sub(r"\B@\w+", 'username', tweet) for tweet in result]  # Remove all usernames
    result = [re.sub(r"(\w)(\1{2,})", r"\1", tweet) for tweet in result]  # Remove all letter repetitions
    result = [re.sub(r"[a-zA-Z]*jaj[a-zA-Z]*", 'jajaja', tweet) for tweet in result]  # Normalize laughs
    result = [re.sub(r"[a-zA-Z]*hah[a-zA-Z]*", 'jajaja', tweet) for tweet in result]  # Normalize laughs
    result = [re.sub(r"\d+", '', tweet) for tweet in result]  # Remove all numbers
    result = [tweet.translate(str.maketrans('', '', string.punctuation)) for tweet in result]  # Remove punctuation

    return result


def remove_accents(tokenized_sentence):
    return [unidecode.unidecode(word) for word in tokenized_sentence]



def tokenize_list(datalist):
    result = []
    for row in datalist:
        result.append(nltk.word_tokenize(row))
    return result


def remove_stopwords(tokenized_data):
    result = []
    for row in tokenized_data:
        result.append([word for word in row if word not in []])  # nltk.corpus.stopwords.words('spanish')])
    return result


def stem_list(datalist):
    stemmer = nltk.stem.SnowballStemmer('spanish')
    result = []
    for row in datalist:
        stemmed_words = [stemmer.stem(word) for word in row]
        result.append(stemmed_words)
    return result


def lemmatize_list(datalist):
    print("Lemmatizing the data. Could take a while...")
    lemmatizer = spacy.load("es_core_news_sm")
    result = []
    for i, row in enumerate(datalist):
        mini_result = [token.lemma_ for token in lemmatizer(row)]
        result.append(mini_result)
        i += 1
    return result


def print_vocabulary_analysis(tokenized_train_list, tokenized_dev_list):
    all_train_words = [item for sublist in tokenized_train_list for item in sublist]
    all_dev_words = [item for sublist in tokenized_dev_list for item in sublist]
    train_vocabulary = []
    dev_vocabulary = []
    for word in all_train_words:
        if word not in train_vocabulary:
            train_vocabulary.append(word)
    for word in all_dev_words:
        if word not in dev_vocabulary:
            dev_vocabulary.append(word)
    train_word_counter = Counter(all_train_words)
    most_common_train_words = train_word_counter.most_common(10)
    dev_word_counter = Counter(all_dev_words)
    most_common_dev_words = dev_word_counter.most_common(10)
    print("The total number of words in TRAINING_DATA is: " + str(len(all_train_words)))
    print("The length of the vocabulary in TRAINING_DATA is: " + str(len(train_vocabulary)))
    print("Most common words in TRAINING_DATA:")
    print(most_common_train_words)
    print()
    print("The total number of words in DEVELOPMENT_DATA is: " + str(len(all_dev_words)))
    print("The length of the vocabulary in DEVELOPMENT_DATA is: " + str(len(dev_vocabulary)))
    print("Most common words in DEVELOPMENT_DATA:")
    print(most_common_dev_words)
    print()

    out_of_vocabulary = []
    for word in dev_vocabulary:
        if word not in train_vocabulary:
            out_of_vocabulary.append(word)
    print("The number of Out-Of-Vocabulary words is: " + str(len(out_of_vocabulary)))
    print("Which is the " + str(len(out_of_vocabulary) / len(dev_vocabulary) * 100) + "% of the Development Vocabulary")
    print()


def print_separator(string_for_printing):
    print('//////////////////////////////////////////////////////////////////////////////////////////////////////////')
    print()
    print("                      " + string_for_printing)
    print()
    print('//////////////////////////////////////////////////////////////////////////////////////////////////////////')
    print()


def libreoffice_processing_analytics(dataframe, dictionary):
    correction_count, nothing_count = set(), set()
    for sentence in dataframe:
        for word in sentence:
            if dictionary.spell(word):
                nothing_count.add(word)
            else:
                if word not in correction_count:
                    print('{}           ---->           {}'.format(word, dictionary.suggest(word)))
                    correction_count.add(word)
                    print()
    print('palabras corregidas:{}       palabras sin tocar:{}'.format(len(correction_count), len(nothing_count)))
    return


# GET THE DATA
print("Loading Hunspell directory")
dictionary = hunspell.HunSpell('./dictionaries/es_ANY.dic', "./dictionaries/es_ANY.aff")  # In case you're using Hunspell

print("Loading Spacy Model")
lemmatizer = spacy.load("es_core_news_md")  # GLOBAL to avoid loading the model several times


train_data, dev_data, test_data, _ = utils.read_files(LANGUAGE_CODE, bStoreFiles=False)
train_dev_data = pandas.concat([train_data, dev_data], ignore_index=True).reset_index(drop=True)

train_data['preprocessed'] = tweet_preprocessing.preprocess(train_data['content'], all_prep=True)
dev_data['preprocessed'] = tweet_preprocessing.preprocess(dev_data['content'], all_prep=True)
test_data['preprocessed'] = tweet_preprocessing.preprocess(test_data['content'], all_prep=True)
train_dev_data['preprocessed'] = tweet_preprocessing.preprocess(train_dev_data['content'], all_prep=True)

# TOKENIZE
print("Tokenizing...")
train_data['tokenized'] = train_data.swifter.progress_bar(False).apply(
    lambda row: utils.tokenize_sentence(row.preprocessed), axis=1)
dev_data['tokenized'] = dev_data.swifter.progress_bar(False).apply(
    lambda row: utils.tokenize_sentence(row.preprocessed), axis=1)
test_data['tokenized'] = test_data.swifter.progress_bar(False).apply(
    lambda row: utils.tokenize_sentence(row.preprocessed), axis=1)
train_dev_data['tokenized'] = train_dev_data.swifter.progress_bar(False).apply(
    lambda row: utils.tokenize_sentence(row.preprocessed), axis=1)

# FEATURE EXTRACTION
train_data['has_uppercase'] = extract_uppercase_feature(train_data['content'])
test_data['has_uppercase'] = extract_uppercase_feature(test_data['content'])

train_data['laughter_feature'] = extract_laughter_feature(train_data['content'])
test_data['laughter_feature'] = extract_laughter_feature(test_data['content'])

train_data['length'] = extract_length_feature(train_data['tokenized'])
test_data['length'] = extract_length_feature(test_data['tokenized'])

train_data['question_mark'] = extract_question_mark_feature(train_data['content'])
test_data['question_mark'] = extract_question_mark_feature(test_data['content'])

train_data['exclamation_mark'] = extract_exclamation_mark_feature(train_data['content'])
test_data['exclamation_mark'] = extract_exclamation_mark_feature(test_data['content'])

train_data['letter_repetition'] = extract_letter_repetition_feature(train_data['content'])
test_data['letter_repetition'] = extract_letter_repetition_feature(test_data['content'])


train_data['pos_voc'], train_data['neg_voc'], train_data['neu_voc'], train_data['none_voc'] = \
    feature_extraction.extract_sent_words_feature(train_data['tokenized'], train_data['tokenized'], train_data['sentiment'],
                               lexicons=True, discriminating_terms=True,
                               discriminating_words=50)
test_data['pos_voc'], test_data['neg_voc'], test_data['neu_voc'], test_data['none_voc'] = \
    feature_extraction.extract_sent_words_feature(test_data['tokenized'], train_data['tokenized'], train_data['sentiment'],
                               lexicons=True, discriminating_terms=True,
                               discriminating_words=50)

# VOCABULARY ANALYSIS

# print("The maximum length of a Tweet in TRAINING_DATA is: " + str(max(train_data['length'])))
# print("The minimum length of a Tweet in TRAINING_DATA is: " + str(min(train_data['length'])))
# print(
#     "The average length of a Tweet in TRAINING_DATA is: " + str(sum(train_data['length']) / len(train_data['length'])))
# print()
# print("The maximum length of a Tweet in DEVELOPMENT_DATA is: " + str(max(test_data['length'])))
# print("The minimum length of a Tweet in DEVELOPMENT_DATA is: " + str(min(test_data['length'])))
# print("The average length of a Tweet in DEVELOPMENT_DATA is: " + str(sum(test_data['length']) / len(test_data['length'])))
# print()

# print_separator("Vocabulary Analysis after tokenize")
#
# print_vocabulary_analysis(train_data['tokenized'], test_data['tokenized'])
#
# print_separator("Vocabulary Analysis after Libreoffice Processing")
# libreoffice_processing_analytics(train_data['tokenized'], dictionary)


# print("LibreOffice Processing... ")
# train_dev_data['final_data'] = train_dev_data.swifter.progress_bar(True).apply(
#     lambda row: utils.libreoffice_processing(row.tokenized, dictionary), axis=1)
# test_data['final_data'] = test_data.swifter.apply(lambda row: utils.libreoffice_processing(row.tokenized, dictionary),
#                                                   axis=1)

# print_separator("ISOL and discriminating analysis")
# isol_analytics(train_dev_data['tokenized'], train_dev_data['tokenized'], train_dev_data['sentiment'])
# isol_analytics(test_data['tokenized'], train_dev_data['tokenized'], train_dev_data['sentiment'])

# print_vocabulary_analysis(train_data['final_data'], test_data['final_data'])
#
# print_separator("After lemmatizing the data...")
# print("Lemmatizing data...")
# train_data['final_data'] = train_data.swifter.apply(lambda row: utils.lemmatize_sentence(row.final_data, lemmatizer), axis=1)
# test_data['final_data'] = test_data.swifter.apply(lambda row: utils.lemmatize_sentence(row.final_data, lemmatizer), axis=1)
#
# print_vocabulary_analysis(train_data['final_data'], test_data['final_data'])
#
# print_separator("After removing accents to the data...")
# print("Removing accents...")
# train_data['final_data'] = train_data.swifter.progress_bar(False).apply(lambda row: remove_accents(row.final_data), axis=1)
# test_data['final_data'] = test_data.swifter.progress_bar(False).apply(lambda row: remove_accents(row.final_data), axis=1)
#
# print_vocabulary_analysis(train_data['final_data'], test_data['final_data'])
#
# print_separator("Label Counts:")
#
# print("Total count of each Label in TRAINING_DATA:")
# print(train_data['sentiment'].value_counts('N'))
# print()
# print("Total count of each Label in DEVELOPMENT_DATA:")
# print(dev_data['sentiment'].value_counts('N'))
# print()

print_separator("Most discriminating words analysis")

# print("Training data:")
# # train_pos_voc, train_neg_voc = get_sentiment_vocabulary(train_data['tokenized'], 'P', 'N')
# train_data['pos_voc'], train_data['neg_voc'], train_data['neu_voc'], train_data[
#     'none_voc'] = extract_sent_words_feature(train_data['tokenized'], train_data['tokenized'])
# print("The most discriminating words between P and N are:")
# print("Category P:")
# # print(train_pos_voc)
# print("Category N:")
# # print(train_neg_voc)
# print()
#
# print("Development data:")
# # dev_pos_voc, dev_neg_voc = get_sentiment_vocabulary(test_data['tokenized'], 'P', 'N')
# test_data['pos_voc'], test_data['neg_voc'], test_data['neu_voc'], test_data['none_voc'] = extract_sent_words_feature(
#     test_data['tokenized'], train_data['tokenized'])
# print("The most discriminating words between P and N are:")
# print("Category P:")
# # print(dev_pos_voc)
# print("Category N:")
# # print(dev_neg_voc)
# print()
#
# print_separator("Correlation analysis in TRAINING_DATA:")
#
with pandas.option_context('display.max_rows', None, 'display.max_columns', None):
    #     '''
    #     print("Hour VS Sentiment")
    #     print()
    #     print(train_data.groupby(['hour', 'sentiment']).size())
    #     print("Month VS Sentiment")
    #     print()
    #     print(train_data.groupby(['month', 'sentiment']).size())
    #     print("Day of the Week VS Sentiment")
    #     print()
    #     print(train_data.groupby(['day_of_week', 'sentiment']).size())
    #     print("Length VS Sentiment")
    #     print()
    #     print(train_data.groupby(['length', 'sentiment']).size())
    #     '''

    # print("Laughter VS Sentiment")
    # print(train_data.groupby(['laughter_feature', 'sentiment']).size())
    # print("Length VS Sentiment")
    # print()
    # print(train_data.groupby([pandas.cut(train_data['length'], numpy.arange(0, 1+50, 5)), 'sentiment']).size())
    # print("Uppercase VS Sentiment")
    # print()
    # print(train_data.groupby(['has_uppercase', 'sentiment']).size())
    # print("Question VS Sentiment")
    # print()
    # print(train_data.groupby(['question_mark', 'sentiment']).size())
    # print("Exclamation VS Sentiment")
    # print()
    # print(train_data.groupby(['exclamation_mark', 'sentiment']).size())
    # print("Letter Repetition VS Sentiment")
    # print()
    # print(train_data.groupby(['letter_repetition', 'sentiment']).size())
    # print("Positive Vocabulary VS Sentiment")
    # print()
    # print(train_data.groupby([pandas.cut(train_data['pos_voc'], numpy.arange(0, 1.0+4, 0.1)), 'sentiment']).size())
    # print("Negative Vocabulary VS Sentiment")
    # print()
    # print(train_data.groupby([pandas.cut(train_data['neg_voc'], numpy.arange(0, 1.0+4, 0.1)), 'sentiment']).size())
    # print("Neutral Vocabulary VS Sentiment")
    # print()
    # print(train_data.groupby([pandas.cut(train_data['neu_voc'], numpy.arange(0, 1.0+4, 0.2)), 'sentiment']).size())
    # print("None Vocabulary VS Sentiment")
    # print()
    # print(train_data.groupby([pandas.cut(train_data['none_voc'], numpy.arange(0, 1.0+4, 0.1)), 'sentiment']).size())
    #
    print("Correlation analysis in DEVELOPMENT_DATA:")
    # print()

    print("Laughter VS Sentiment")
    print()
    print(test_data.groupby(['laughter_feature', 'sentiment']).size())

    # '''
    # print("Hour VS Sentiment")
    # print()
    # print(dev_data.groupby(['hour', 'sentiment']).size())
    # print("Month VS Sentiment")
    # print()
    # print(dev_data.groupby(['month', 'sentiment']).size())
    # print("Day of the Week VS Sentiment")
    # print()
    # print(dev_data.groupby(['day_of_week', 'sentiment']).size())
    # print("Length VS Sentiment")
    # print()
    # print(dev_data.groupby(['length', 'sentiment']).size())
    # '''
    # print("Uppercase VS Sentiment")
    # print()
    # print(test_data.groupby(['has_uppercase', 'sentiment']).size())
    # print("Length VS Sentiment")
    # print()
    # print(test_data.groupby([pandas.cut(test_data['length'], numpy.arange(0, 1+50, 5)), 'sentiment']).size())
    # print("Question VS Sentiment")
    # print()
    # print(test_data.groupby(['question_mark', 'sentiment']).size())
    # print("Exclamation VS Sentiment")
    # print()
    # print(test_data.groupby(['exclamation_mark', 'sentiment']).size())
    # print("Positive Vocabulary VS Sentiment")
    # print()
    print(test_data.groupby([pandas.cut(test_data['pos_voc'], numpy.arange(0, 1.0+4, 0.1)), 'sentiment']).size())
    print("Negative Vocabulary VS Sentiment")
    print()
    print(test_data.groupby([pandas.cut(test_data['neg_voc'], numpy.arange(0, 1.0+4, 0.1)), 'sentiment']).size())
    print("Neutral Vocabulary VS Sentiment")
    print()
    print(test_data.groupby([pandas.cut(test_data['neu_voc'], numpy.arange(0, 1.0+4, 0.1)), 'sentiment']).size())
    print("None Vocabulary VS Sentiment")
    print()
    print(test_data.groupby([pandas.cut(test_data['none_voc'], numpy.arange(0, 1.0+4, 0.1)), 'sentiment']).size())
