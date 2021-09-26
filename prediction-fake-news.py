from pyspark.sql.dataframe import DataFrame
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, StructField, StructType
from pyspark.sql.functions import col, from_json, lit

## Importing the Dependencies
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


kafka_topic_name = "topic.collection.news"
kafka_prediction_topic = "topic.prediction.news"
kafka_bootstrap_servers = '52.15.77.79:9094'

# Stemming
stem = PorterStemmer() # basically creating an object for stemming! Stemming is basically getting the root word, for eg: loved --> love! 
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ',content) # this basically replaces everything other than lower a-z & upper A-Z with a ' ', for eg apple,bananna --> apple bananna
    stemmed_content = stemmed_content.lower() # to make all text lower case
    stemmed_content = stemmed_content.split() # this basically splits the line into words with delimiter as ' '
    stemmed_content = [stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')] # basically remove all the stopwords and apply stemming to the final data
    stemmed_content = ' '.join(stemmed_content) # this basically joins back and returns the cleaned sentence
    return stemmed_content

vectorizer = CountVectorizer()
def training_model():
    # download stopwords and wordnet
    # nltk.download('stopwords')
    # nltk.download('wordnet')

    # loading the dataset to dataframe
    news_dataset = pd.read_csv('datasets/train.csv')

    # Pre-processing data
    news_dataset = news_dataset.fillna('')
    news_dataset['content'] = news_dataset['author'] + " " + news_dataset['title']
    news_dataset['content'] = news_dataset['content'].apply(stemming)
    X = news_dataset['content'].values
    y = news_dataset['label'].values
    X = vectorizer.fit_transform(X)

    # training the logistic regression model
    model = LogisticRegression(C = 100, penalty = 'l2', solver= 'newton-cg')
    model.fit(X, y)
    return model

def get_prediction(trained_model, processed_news):
    prediction = trained_model.predict(processed_news)
    if (prediction[0]==0):
        return "REAL"
    else:
        return "FAKE"

def process_received_news(batch_df: DataFrame, batch_id):
    if (batch_df.count() > 0):
        data_frame = batch_df.toPandas()
        print(f'======================> Processing Data Streaming {data_frame.iloc[0]["id"]} <======================')
        data_frame['content'] = data_frame['author'] + " " + data_frame['title']
        data_frame['content'] = data_frame['content'].apply(stemming)
        X_test = data_frame['content']
        X_test = vectorizer.transform(X_test)
        print(f'======================> Prediction Data Streaming {data_frame.iloc[0]["id"]} <======================')
        predict_value = get_prediction(model, X_test)
        batch_df = batch_df.withColumn("Label", lit(predict_value))
        print(batch_df.show())
        batch_df.selectExpr("CAST(id AS STRING) AS key", "to_json(struct(id, Label)) AS value") \
                .write.format("kafka").option("kafka.bootstrap.servers", kafka_bootstrap_servers) \
                                    .option("topic", kafka_prediction_topic).save()


sc = SparkContext(appName="app-publish-news", master="local[*]")
spark = SparkSession(sc)
# spark = SparkSession.builder.appName("app-publish-news").master("local[*]").getOrCreate()

# Construct a streaming DataFrame that reads from topic
df_message_consume = spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", kafka_bootstrap_servers) \
    .option("subscribe", kafka_topic_name) \
    .option("startingOffsets", "latest") \
    .load()

df_value = df_message_consume.selectExpr("CAST(value AS STRING)")
json_schemas = StructType([StructField("id", StringType()), StructField("title", StringType()), \
                            StructField("author", StringType()), StructField("text", StringType())])
df_news = df_value.select(from_json(col("value"),json_schemas).alias("data_news")).select("data_news.*")

# Processing news data and predict fake/real
model = training_model()

df_news.writeStream.foreachBatch(process_received_news) \
                    .start() \
                    .awaitTermination()
