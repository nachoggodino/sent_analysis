import re
import string
import emoji
import hunspell
import nltk
import spacy
import pandas as pd
from imblearn.over_sampling import RandomOverSampler

from src import utils

from num2words import num2words
from unidecode import unidecode

# LOADINGS
print("Loading Hunspell directory")
dictionary = hunspell.HunSpell('./dictionaries/es_ANY.dic', "./dictionaries/es_ANY.aff")  # TODO ADD WORDS TO DICTIONARY

print("Loading Spacy Model")
lemmatizer = spacy.load("es_core_news_md")  # GLOBAL to avoid loading the model several times

print("Loading NLTK stuff")
stemmer = nltk.stem.SnowballStemmer('spanish')
nltk.data.path.append("nltk_data")
stopwords = nltk.corpus.stopwords.words('spanish')


def preprocess(data, bEmoji=False, bHashtags=False, bURL=False, bUsername=False, bLetRep=False, bLaughter=False,
                     bXque=False, bPunctuation=False, bNumber=False, bLowercasing=False, bAll=False):

    result, emojis = [], list()
    for tweet in data:
        clean_tweet = tweet
        clean_tweet = clean_tweet.replace('\n', '').strip()
        clean_tweet = clean_tweet.replace(u'\u2018', "'").replace(u'\u2019', "'")

        if bEmoji or bAll:
            # clean_tweet = " ".join([emoji_pattern.sub(r'EMOJI', word) for word in clean_tweet.split()])
            clean_tweet = emoji.demojize(clean_tweet, use_aliases=True)

        if bHashtags or bAll:
            clean_tweet = re.sub(r"\B#\w+", lambda m: camel_case_split(m.group(0)), clean_tweet)

        if bURL or bAll:
            clean_tweet = re.sub(r"http\S+", "", clean_tweet)  # URL

        if bUsername or bAll:
            clean_tweet = re.sub(r"\B@\w+", 'USUARIO', clean_tweet)  # USERNAME

        if bLetRep or bAll:
            clean_tweet = re.sub(r"(\w)(\1{2,})", r"\1", clean_tweet)  # LETTER REPETITION

        if bLaughter or bAll:
            clean_tweet = re.sub(r"[a-zA-Z]*jaj[a-zA-Z]*", 'JAJAJA', clean_tweet, re.IGNORECASE)
            clean_tweet = re.sub(r"[a-zA-Z]*hah[a-zA-Z]*", 'JAJAJA', clean_tweet, re.IGNORECASE)
            clean_tweet = re.sub(r"[a-zA-Z]*jej[a-zA-Z]*", 'JAJAJA', clean_tweet, re.IGNORECASE)  # LAUGHTER NOT FULLY WORKING
            clean_tweet = re.sub(r"[a-zA-Z]*joj[a-zA-Z]*", 'JAJAJA', clean_tweet, re.IGNORECASE)
            clean_tweet = re.sub(r"[a-zA-Z]*jij[a-zA-Z]*", 'JAJAJA', clean_tweet, re.IGNORECASE)
            clean_tweet = re.sub(r"[a-zA-Z]*lol[a-zA-Z]*", 'JAJAJA', clean_tweet, re.IGNORECASE)
            clean_tweet = re.sub(r"[a-zA-Z]*hah[a-zA-Z]*", 'JAJAJA', clean_tweet, re.IGNORECASE)
            clean_tweet = re.sub(r"[a-zA-Z]*lmao[a-zA-Z]*", 'JAJAJA', clean_tweet, re.IGNORECASE)
            clean_tweet = re.sub(r"[a-zA-Z]*xd[a-zA-Z]*", 'JAJAJA', clean_tweet, re.IGNORECASE)

        if bXque or bAll:
            clean_tweet = re.sub(r"\b(x)\b", 'por', clean_tweet, re.IGNORECASE)  # x = por
            clean_tweet = re.sub(r"\b(d)\b", 'de', clean_tweet, re.IGNORECASE)  # d = de
            clean_tweet = re.sub(r"\b(q)\b", 'que', clean_tweet, re.IGNORECASE)  # q = que
            clean_tweet = re.sub(r"\b(xq)\b", 'porque', clean_tweet, re.IGNORECASE)  # xq = porque
            clean_tweet = re.sub(r"\b(pq)\b", 'porque', clean_tweet, re.IGNORECASE)  # pq = porque

        if bNumber or bAll:
            # clean_tweet = re.sub(r"\d+", lambda m: num2words(m.group(0)), clean_tweet)  # NUMBERS
            clean_tweet = re.sub(r"\d+", '', clean_tweet)  # NUMBERS

        if bPunctuation or bAll:
            sc = {'¡', '!', '?', '¿', '#', '@', '_', ':'}
            punctuation = ''.join([c for c in string.punctuation + '¡¿' if c not in sc])  # PUNCTUATION
            clean_tweet = clean_tweet.translate(str.maketrans('', '', punctuation + '¡'))

        if bLowercasing or bAll:
            clean_tweet = clean_tweet.lower()  # LOWERCASING

        result.append(clean_tweet)

    return result


def camel_case_split(identifier):
    clean_identifier = re.sub('[#]', '', identifier)
    matches = re.finditer(".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)", clean_identifier)
    return ' '.join([m.group(0) for m in matches])


def remove_accents(tokenized_sentence):
    return [unidecode.unidecode(word) for word in tokenized_sentence]


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
    df = pd.DataFrame(data=x_resampled[0:, 0:], columns=['content'])
    df['sentiment'] = y_resampled
    df = df.sample(frac=1).reset_index(drop=True)
    return df
