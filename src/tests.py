
import vaderSentiment.vaderSentiment
from src.config import *
import fasttext
from src import fasttext_embedding, tweet_preprocessing


tweets_to_test = ['La radio de cercanía, la radio más cercana a la realidad en @HablarporHablar con esta llamada',  # 2
                  'Buenos dias, vamos a hacer algunos recados y a empezar el dia con energia!!',  # 3
                  '@mireiaescribano justo cuando se terminan las fiestas de verano, me viene genial',  # 3
                  'No sabes cuantas horas, minutos y segundos espero para volver a ver esa sonrisa que tanto me gusta ver salir de ti',  # 0
                  '@cinthiaszc jajajaja me vas a decir a mi mi abuela cocina tan rico que mando al tacho la dieta :v',  # 0
                  '@JuanPaUrrego ¡Que lindo eres que lindo actúas!! te adoroVen a Perú pls'  # 3
                  ]
label_dictionary = ['0', '1', '2', '3']

fasttext_path = '../fasttext/models/{}_{}'.format('tass2019', FT_MODEL_NAME)
fasttext_model = fasttext.load_model(path=fasttext_path)
dev_fasttext_probabilities, dev_fasttext_predictions = fasttext_embedding.predict_with_fasttext_model(
    fasttext_model, tweets_to_test, label_dictionary)
print(dev_fasttext_probabilities)
print(dev_fasttext_predictions)

# sentences = ['horrible']
#
# analyzer = vaderSentiment.vaderSentiment.SentimentIntensityAnalyzer()
# for sentence in sentences:
#     vs = analyzer.polarity_scores(sentence)
#     print("{:-<65} {}".format(sentence, str(vs)))

