import re
import string
import emoji
import hunspell
import nltk
import numpy
import spacy
import pandas as pd
from imblearn.over_sampling import RandomOverSampler

from src import utils
from src.config import *

from unidecode import unidecode

# LOADINGS
print("Loading Hunspell directory")
dictionary = hunspell.HunSpell('../resources/dictionaries/es_ANY.dic', '../resources/dictionaries/es_ANY.aff')  # TODO ADD WORDS TO DICTIONARY

print("Loading Spacy Model")
lemmatizer = spacy.load("es_core_news_md")  # GLOBAL to avoid loading the model several times

print("Loading NLTK stuff")
stemmer = nltk.stem.SnowballStemmer('spanish')
nltk.data.path.append("nltk_data")
stopwords = nltk.corpus.stopwords.words('spanish')


def preprocess_data(data, conf='main'):
    if conf == 'main':
        data['preprocessed'] = preprocess(
            data, all_prep=PREP_ALL, emojis=PREP_EMOJI, hashtags=PREP_HASHTAGS, laughter=PREP_LAUGHTER,
            letrep=PREP_LETREP, lowercasing=PREP_LOWER, number=PREP_NUMBER, punctuation=PREP_PUNCT, xque=PREP_XQUE,
            username=PREP_USERNAME, url=PREP_URL)
    if conf == 'embedding':
        data['preprocessed'] = preprocess(
            data, all_prep=EMB_PREP_ALL, emojis=EMB_PREP_EMOJI, hashtags=EMB_PREP_HASHTAGS, laughter=EMB_PREP_LAUGHTER,
            letrep=EMB_PREP_LETREP, lowercasing=EMB_PREP_LOWER, number=EMB_PREP_NUMBER, punctuation=EMB_PREP_PUNCT,
            xque=EMB_PREP_XQUE, username=EMB_PREP_USERNAME, url=EMB_PREP_URL)
    # TOKENIZE
    data['final_data'] = [tokenize_sentence(row) for row in data.preprocessed]

    # LIBREOFFICE CORRECTION
    if B_LIBREOFFICE:
        print("LibreOffice Processing... ")
        data['final_data'] = data.swifter.apply(lambda row: libreoffice_processing(row.final_data), axis=1)

    # LEMMATIZING
    if B_LEMMATIZE:
        print("Lemmatizing data...")
        data['final_data'] = data.swifter.apply(lambda row: lemmatize_sentence(row.final_data), axis=1)

    # ACCENTS REMOVAL
    if B_REMOVE_ACCENTS:
        print("Removing accents...")
        data['final_data'] = data.swifter.progress_bar(False).apply(lambda row: remove_accents(row.final_data), axis=1)

    # STOPWORDS REMOVAL
    if B_REMOVE_STOPWORDS:
        print('Removing stopwords...')
        data['final_data'] = data.swifter.progress_bar(False).apply(lambda row: remove_stopwords(row.final_data), axis=1)

    data['final_data'] = [utils.untokenize_sentence(row) for row in data['final_data']]

    return data['final_data']


def preprocess(data, emojis=False, hashtags=False, url=False, username=False, letrep=False, laughter=False,
               xque=False, punctuation=False, number=False, lowercasing=False, all_prep=False):

    result = []
    for tweet in data:
        clean_tweet = tweet
        clean_tweet = clean_tweet.replace('\n', '').strip()
        clean_tweet = clean_tweet.replace(u'\u2018', "'").replace(u'\u2019', "'")

        if emojis or all_prep:
            # clean_tweet = " ".join([emoji_pattern.sub(r'EMOJI', word) for word in clean_tweet.split()])
            clean_tweet = emoji.demojize(clean_tweet, use_aliases=True)

        if hashtags or all_prep:
            clean_tweet = re.sub(r"\B#\w+", lambda m: camel_case_split(m.group(0)), clean_tweet)

        if url or all_prep:
            clean_tweet = re.sub(r"http\S+", "", clean_tweet)  # URL

        if username or all_prep:
            clean_tweet = re.sub(r"\B@\w+", 'USUARIO', clean_tweet)  # USERNAME

        if letrep or all_prep:
            clean_tweet = re.sub(r"(\w)(\1{2,})", r"\1", clean_tweet)  # LETTER REPETITION

        if laughter or all_prep:
            clean_tweet = re.sub(r"[a-zA-Z]*jaj[a-zA-Z]*", 'JAJAJA', clean_tweet, re.IGNORECASE)
            clean_tweet = re.sub(r"[a-zA-Z]*hah[a-zA-Z]*", 'JAJAJA', clean_tweet, re.IGNORECASE)
            clean_tweet = re.sub(r"[a-zA-Z]*jej[a-zA-Z]*", 'JAJAJA', clean_tweet, re.IGNORECASE)  # LAUGHTER NOT FULLY WORKING
            clean_tweet = re.sub(r"[a-zA-Z]*joj[a-zA-Z]*", 'JAJAJA', clean_tweet, re.IGNORECASE)
            clean_tweet = re.sub(r"[a-zA-Z]*jij[a-zA-Z]*", 'JAJAJA', clean_tweet, re.IGNORECASE)
            clean_tweet = re.sub(r"[a-zA-Z]*lol[a-zA-Z]*", 'JAJAJA', clean_tweet, re.IGNORECASE)
            clean_tweet = re.sub(r"[a-zA-Z]*hah[a-zA-Z]*", 'JAJAJA', clean_tweet, re.IGNORECASE)
            clean_tweet = re.sub(r"[a-zA-Z]*lmao[a-zA-Z]*", 'JAJAJA', clean_tweet, re.IGNORECASE)
            clean_tweet = re.sub(r"[a-zA-Z]*xd[a-zA-Z]*", 'JAJAJA', clean_tweet, re.IGNORECASE)

        if xque or all_prep:
            clean_tweet = re.sub(r"\b(x)\b", 'por', clean_tweet, re.IGNORECASE)  # x = por
            clean_tweet = re.sub(r"\b(d)\b", 'de', clean_tweet, re.IGNORECASE)  # d = de
            clean_tweet = re.sub(r"\b(q)\b", 'que', clean_tweet, re.IGNORECASE)  # q = que
            clean_tweet = re.sub(r"\b(xq)\b", 'porque', clean_tweet, re.IGNORECASE)  # xq = porque
            clean_tweet = re.sub(r"\b(pq)\b", 'porque', clean_tweet, re.IGNORECASE)  # pq = porque

        if number or all_prep:
            # clean_tweet = re.sub(r"\d+", lambda m: num2words(m.group(0)), clean_tweet)  # NUMBERS
            clean_tweet = re.sub(r"\d+", '', clean_tweet)  # NUMBERS

        if punctuation or all_prep:
            sc = {'¡', '!', '?', '¿', '#', '@', '_', ':'}
            punctuation = ''.join([c for c in string.punctuation + '¡¿' if c not in sc])  # PUNCTUATION
            clean_tweet = clean_tweet.translate(str.maketrans('', '', punctuation + '¡'))

        if lowercasing or all_prep:
            clean_tweet = clean_tweet.lower()  # LOWERCASING

        result.append(clean_tweet)

    return result


def camel_case_split(identifier):
    clean_identifier = re.sub('[#]', '', identifier)
    matches = re.finditer(".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)", clean_identifier)
    return ' '.join([m.group(0) for m in matches])


def remove_accents(tokenized_sentence):
    return [unidecode(word) for word in tokenized_sentence]


def remove_stopwords(tokenized_data):
    return [word for word in tokenized_data if word not in stopwords]


def stem_list(datalist):
    print("Applying stemming")
    return [[stemmer.stem(word) for word in row] for row in datalist]


def lemmatize_sentence(sentence):
    data = utils.untokenize_sentence(sentence)
    return [token.lemma_ for token in lemmatizer(data)]


def libreoffice_processing(tokenized_sentence):
    return [word if dictionary.spell(word) is True else next(iter(dictionary.suggest(word)), word)
            for word in tokenized_sentence]


def tokenize_sentence(sentence):
    return nltk.word_tokenize(sentence)


def perform_upsampling(dataframe):
    ros = RandomOverSampler(random_state=1234)
    x_resampled, y_resampled = ros.fit_resample(dataframe[['content']], dataframe['sentiment'])
    df = pd.DataFrame()
    df['content'] = x_resampled['content']
    df['sentiment'] = y_resampled
    df = df.sample(frac=1).reset_index(drop=True)
    return df


def preprocess_and_analyze(data):

    result = []
    hashtags, urls, usernames, letreps, laughters, numbers = list(), list(), list(), list(), list(), list()
    emojis, qque, xpor, dde, xqs, pqs = list(), list(), list(), list(), list(), list()

    for tweet in data:

        emoji_tweet = emoji.demojize(tweet, use_aliases=True)
        emojis.extend(re.findall(r":[a-z_0-9]*?:", emoji_tweet, re.IGNORECASE))

        hashtags.extend(re.findall(r"\B#\w+", tweet))  # HASHTAGS
        urls.extend(re.findall(r"http\S+", tweet))  # URL
        usernames.extend(re.findall(r"\B@\w+", tweet))  # URL
        letreps.extend(re.findall(r"(\w)(\1{2,})", tweet))  # URL

        laughters.extend(re.findall(r"[a-zA-Z]*jaj[a-zA-Z]*", tweet, re.IGNORECASE))  # URL
        laughters.extend(re.findall(r"[a-zA-Z]*hah[a-zA-Z]*", tweet, re.IGNORECASE))  # URL
        laughters.extend(re.findall(r"[a-zA-Z]*jej[a-zA-Z]*", tweet, re.IGNORECASE))  # URL
        laughters.extend(re.findall(r"[a-zA-Z]*joj[a-zA-Z]*", tweet, re.IGNORECASE))  # URL
        laughters.extend(re.findall(r"[a-zA-Z]*jij[a-zA-Z]*", tweet, re.IGNORECASE))  # URL
        laughters.extend(re.findall(r"[a-zA-Z]*lol[a-zA-Z]*", tweet, re.IGNORECASE))  # URL

        numbers.extend(re.findall(r"\d+", tweet))  # URL

        qque.extend(re.findall(r"\b(q)\b", tweet, re.IGNORECASE))  # q = que
        xpor.extend(re.findall(r"\b(x)\b", tweet, re.IGNORECASE))  # x = por
        dde.extend(re.findall(r"\b(d)\b", tweet, re.IGNORECASE))  # d = de
        xqs.extend(re.findall(r"\b(xq)\b", tweet, re.IGNORECASE))  # xq = porque
        pqs.extend(re.findall(r"\b(pq)\b", tweet, re.IGNORECASE))  # pq = porque

    return hashtags, urls, usernames, letreps, laughters, numbers, emojis, qque, xpor, dde, xqs, pqs
