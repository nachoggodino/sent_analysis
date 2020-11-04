
import vaderSentiment.vaderSentiment
from src.config import *
import fasttext
from flair.models import TextClassifier
from src import fasttext_embedding, tweet_preprocessing, bert_embeddings


tweets_to_test = ['La radio de cercanía, la radio más cercana a la realidad en @HablarporHablar con esta llamada',  # 2
                  'Buenos dias, vamos a hacer algunos recados y a empezar el dia con energia!!',  # 3
                  '@mireiaescribano justo cuando se terminan las fiestas de verano, me viene genial',  # 3
                  'No sabes cuantas horas, minutos y segundos espero para volver a ver esa sonrisa que tanto me gusta ver salir de ti',  # 0
                  '@cinthiaszc jajajaja me vas a decir a mi mi abuela cocina tan rico que mando al tacho la dieta :v',  # 0
                  'te adoroVen a Perú pls'  # 3
                  ]
label_dictionary = ['0', '1', '2', '3']

fasttext_path = '../fasttext/models/{}_{}'.format('intertass', FT_MODEL_NAME)
fasttext_model = fasttext.load_model(path=fasttext_path)
dev_fasttext_probabilities, dev_fasttext_predictions = fasttext_embedding.predict_with_fasttext_model(
    fasttext_model, tweets_to_test, label_dictionary)
print(dev_fasttext_probabilities)
print(dev_fasttext_predictions)

print("BERT MODEL")
bert_path = '{}/{}/{}/best-model.pt'.format(BERT_MODEL_PATH, BERT_MODEL_NAME, 'intertass')
bert_model = TextClassifier.load(bert_path)
dev_bert_probabilities, dev_bert_predictions = bert_embeddings.predict_with_bert_model(
    bert_model, tweets_to_test, label_dictionary)
print(dev_bert_probabilities)
print(dev_bert_predictions)

# sentences = ['horrible']
#
# analyzer = vaderSentiment.vaderSentiment.SentimentIntensityAnalyzer()
# for sentence in sentences:
#     vs = analyzer.polarity_scores(sentence)
#     print("{:-<65} {}".format(sentence, str(vs)))

