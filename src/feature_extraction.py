import re
import pandas as pd
from textacy import keyterms

regex_uppercase = re.compile(r"\b[A-Z][A-Z]+\b")  # TODO


def extract_length_feature(sentences_list):
    return [len(tweet.split(' ')) for tweet in sentences_list]


def extract_uppercase_feature(dataframe):
    return [len(regex_uppercase.findall(tweet)) for tweet in dataframe]


def extract_hashtag_number_feature(dataframe):
    return[tweet.count('#') for tweet in dataframe]


def extract_laughter_feature(dataframe):
    return [1 if re.search(r"[a-zA-Z]*jaj[a-zA-Z]*", tweet) or re.search(r"[a-zA-Z]*hah[a-zA-Z]*", tweet)
            or re.search(r"[a-zA-Z]*jej[a-zA-Z]*", tweet) or re.search(r"[a-zA-Z]*joj[a-zA-Z]*", tweet)
            or re.search(r"[a-zA-Z]*jij[a-zA-Z]*", tweet) or re.search(r"[a-zA-Z]*lol[a-zA-Z]*", tweet)
            or re.search(r"[a-zA-Z]*lmao[a-zA-Z]*", tweet) or re.search(r"[a-zA-Z]*xd[a-zA-Z]*", tweet)
            else 0 for tweet in dataframe]


def extract_question_mark_feature(dataframe):
    return [1 if re.search(r"[/?/]", tweet) is True else 0 for tweet in dataframe]


def extract_exclamation_mark_feature(dataframe):
    return [1 if re.search(r"[/!/]", tweet) is True else 0 for tweet in dataframe]


def extract_letter_repetition_feature(dataframe):
    return [1 if re.search(r"(\w)(\1{2,})", tweet) is True else 0 for tweet in dataframe]


def extract_sent_words_feature(tokenized_data, data_feed, sentiment_feed, lexicons=True, discriminating_terms=False,
                               discriminating_words=25):
    positive_voc, negative_voc = get_sentiment_vocabulary(
        data_feed, sentiment_feed, 3, 0, lexicons, discriminating_terms, discriminating_words)
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
        neutral_result.append(0 if (pos_count + neg_count) == 0 else 1-(abs(pos_count-neg_count)/(pos_count+neg_count)))
        none_result.append(1-((neg_count+pos_count)/length))
    return pos_result, neg_result, neutral_result, none_result


def get_sentiment_vocabulary(data, sentiment_feed, positive, negative, lexicons, discriminating_terms,
                             discriminating_words):
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
    positive_vocabulary, negative_vocabulary = keyterms.most_discriminating_terms(
        pos_neg_tweets, pos_neg_bool_labels, top_n_terms=discriminating_words)

    pos_df = pd.read_csv('./lexicons/isol/positivas_mejorada.csv', encoding='latin-1', header=None, names=['words'])
    neg_df = pd.read_csv('./lexicons/isol/negativas_mejorada.csv', encoding='latin-1', header=None, names=['words'])

    positive_result, negative_result = set(), set()
    if lexicons:
        positive_result = positive_result.union(set(pos_df['words'].array))
        negative_result = negative_result.union(set(neg_df['words'].array))
    if discriminating_terms:
        positive_result = positive_result.union(set(positive_vocabulary))
        negative_result = negative_result.union(set(negative_vocabulary))
    return positive_result, negative_result