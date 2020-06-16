from sklearn import linear_model, naive_bayes, tree
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, \
    GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC

lr = linear_model.LogisticRegression()
nb = naive_bayes.MultinomialNB()
dt = tree.DecisionTreeClassifier()
svm = SVC(probability=True)
rf = RandomForestClassifier()
et = ExtraTreesClassifier()
ada = AdaBoostClassifier()
gb = GradientBoostingClassifier()
sgd = SGDClassifier()

# PHASES
B_TEST_PHASE = True  # If true, the test set is used.

# DATA FETCH
B_REDUCED = False  # If true, NEU and NONE are treated as one category. DO NOT USE
B_TRAIN_PLUS_DEV = False  # If true, the train and dev sets are merged
DATASET_ARRAY = ['tass2019']
SHUFFLE_SEED = 1234
SPLIT_SEP_1 = 0.7
SPLIT_SEP_2 = 0.85

# DATA AUGMENTATION
B_UPSAMPLING = True  # If true, upsampling is performed.
B_BACKTRANSLATION = False  # If true, data is augmented using the backtranslation strategy.

# VOCABULARY FUNCTIONS. Both can be used at the same time
B_LEXICONS = True  # If true, the sentiment vocabulary uses external lexicons.
B_DISCRIMINATING_TERMS = False  # If true the sentiment vocabulary uses the keyterms discriminating words
NUM_DISCRIMINATING_WORDS = 50

# VOCABULARY PREPROCESSINGS
B_LEMMATIZE = False  # If true, the data is lemmatized.
B_REMOVE_ACCENTS = False  # If true, the accents are removed from the data
B_REMOVE_STOPWORDS = False  # If true, stopwords are removed from the data
B_LIBREOFFICE = False  # If true, words not in the libreoffice dictionary are corrected

# CLASSIFICATION
B_FEATURES = True  # If true, the classifier is feature based, overriding the CV and TFIDF options
B_ONE_VS_REST = False  # If true, the classifier uses a One vs All strategy
B_COUNTVECTORS = False  # If true, count vectors are performed
B_TFIDF = False  # If true, tf-idf vectors are performed
B_CLASSIC_BOW = False  # If true, B_COUNTVECTORS must also be true

# CLASSIFIERS
CLASSIFIER = lr  # , nb, ada, gb]
S_CLASSIFIER_NAME = 'LR'  # , 'NB', 'ADA', 'GB']

# RESULTS
CONFUSION_MATRIX = False
PREC_AND_RECALL = False

# TODO                                      PREPROCESSING
PREP_EMOJI = False
PREP_HASHTAGS = False
PREP_USERNAME = False
PREP_URL = False
PREP_PUNCT = False
PREP_NUMBER = False
PREP_LOWER = False
PREP_LAUGHTER = False
PREP_LETREP = False
PREP_XQUE = False
PREP_ALL = True

EMB_PREP_EMOJI = False
EMB_PREP_HASHTAGS = False
EMB_PREP_USERNAME = False
EMB_PREP_URL = False
EMB_PREP_PUNCT = True
EMB_PREP_NUMBER = False
EMB_PREP_LOWER = True
EMB_PREP_LAUGHTER = False
EMB_PREP_LETREP = False
EMB_PREP_XQUE = False
EMB_PREP_ALL = False

B_STORE_PREPROCESSED = False
PREP_FILE_NAME = 'allPrep'

# TODO                                      FEATURES
B_FEAT_LENGTH = True
B_FEAT_QUESTIONMARK = True
B_FEAT_EXCLAMARK = True
B_FEAT_LET_REP = True
B_FEAT_HASHTAGS = True
B_FEAT_VOCABULARY = True
B_FEAT_UPPERCASE = True
B_FEAT_LAUGHTER = True

# TODO                                      FASTTEXT
FT_MODEL_PATH = '../fasttext/models'
FT_MODEL_NAME = 'june_test1'

FT_LEARNING_RATE = 0.05
FT_WORDGRAM = 2
FT_EPOCH = 5

# At least one of those has to be True
B_WIKIPEDIA = False
B_WIKIPEDIA_ALIGNED = False
B_COMMONCRAWL = False
B_INGEOTEC = True

# TODO                                      BERT
BERT_MODEL_PATH = '../bert/beto/models'
BERT_MODEL_NAME = 'june_test1'

# TODO                                      DEMO
B_DEMO = False
DEMO_TEXTS = tweets_to_test = ['La radio de cercanía, la radio más cercana a la realidad en @HablarporHablar con esta llamada',  # 2
                  'Buenos dias, vamos a hacer algunos recados y a empezar el dia con energia!!',  # 3
                  '@mireiaescribano justo cuando se terminan las fiestas de verano, me viene genial',  # 3
                  'No sabes cuantas horas, minutos y segundos espero para volver a ver esa sonrisa que tanto me gusta ver salir de ti',  # 0
                  '@cinthiaszc jajajaja me vas a decir a mi mi abuela cocina tan rico que mando al tacho la dieta :v',  # 0
                  '@JuanPaUrrego ¡Que lindo eres que lindo actúas!! te adoroVen a Perú pls'  # 3
                  ]
