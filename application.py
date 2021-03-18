from flask import Flask, render_template, request, flash, redirect, url_for, send_file
import pandas as pd
import tempfile
import base64
import nltk
import csv
nltk.data.path.append(tempfile.gettempdir())
import gensim
from nltk.corpus import stopwords
nltk.download("stopwords")
nltk.download('wordnet')
import matplotlib.pyplot as plt
from collections import Counter

stop = stopwords.words('english')
import pyLDAvis
import pyLDAvis.gensim
from io import BytesIO

import os, time
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer


nltk.download('vader_lexicon')

home = os.path.expanduser('~')
UPLOAD_FOLDER = os.path.join(home, 'Downloads')
ALLOWED_EXTENSIONS = {'xlsx', 'csv'}

application = Flask(__name__)
application.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
stemmer = SnowballStemmer("english")

import pandas as pd

import base64

import gensim
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from collections import Counter

stop = stopwords.words('english')
import pyLDAvis
import pyLDAvis.gensim
from io import BytesIO

import os, time
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

#nltk.download('vader_lexicon')

home = os.path.expanduser('~')
UPLOAD_FOLDER = os.path.join(home, 'Downloads')
ALLOWED_EXTENSIONS = {'xlsx', 'csv'}

application = Flask(__name__)
application.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
stemmer = SnowballStemmer("english")


def nps_score_calculation(ratinglist):
    Promoters = ratinglist[5]
    Detractors = ratinglist[1] + ratinglist[2] + ratinglist[3]
    Passive = ratinglist[4]
    Total = Promoters + Detractors + Passive
    NPS = ((Promoters - Detractors) / Total) * 100
    return NPS


def sentiment_scores(sentence):
    sid_obj = SentimentIntensityAnalyzer()
    sentiment_dict = sid_obj.polarity_scores(sentence)
    if sentiment_dict['compound'] >= 0.05:
        return ("Positive")
    elif sentiment_dict['compound'] <= - 0.05:
        return ("Negative")
    else:
        return ("Neutral")


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result


@application.route('/')
def upload():
    return render_template('index.html')
    
@application.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        select_name = "in_topic"
        topics = request.form.get("topic")
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            data = pd.read_csv(file)
            '''
            data.iloc[:, 0] = data.iloc[:, 0].fillna("NO REVIEW")
            reviews_text = data.iloc[:, 0].values.tolist()
            reviews_text = [re.sub(r'[^A-Za-z\s]', '', text) for text in reviews_text]
            data.iloc[:, 0] = reviews_text
            data.iloc[:, 0] = data.iloc[:, 0].apply(
                lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
            processed_docs = data.iloc[:, 0].map(preprocess)
            dictionary = gensim.corpora.Dictionary(processed_docs)
            dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
            bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
            lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=int(topics), id2word=dictionary, passes=2,
                                                   workers=2)
            top_words_per_topic = []
            for t in range(lda_model.num_topics):
                top_words_per_topic.extend([(t,) + x for x in lda_model.show_topic(t, topn=30)])
            
            #wor = pd.DataFrame(top_words_per_topic, columns=['Topic', 'Word', 'P'])
            #path = str(UPLOAD_FOLDER) + r"/topic.csv"
            #wor.to_csv(path)
            # pyLDAvis.enable_notebook()
            vis = pyLDAvis.gensim.prepare(lda_model, bow_corpus, dictionary)
            # pyLDAvis.show(vis)
            result = "templates/result.html"
            # figfile = BytesIO()
            pyLDAvis.save_html(vis, result)
            time.sleep(15)
            #return render_template("result.html")
            '''
            time.sleep(10)
            return send_file("templates/result.html", as_attachment=True)
            #return render_template('index.html', url="result.html", in_select_topic=select_name)
    return render_template('ts.html')


@application.route('/sentiment', methods=['GET', 'POST'])
def sentiment():
    select_name = "sentiment"
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            data = pd.read_csv(file)
            data["text"] = data["text"].fillna("NO REVIEW")
            ra = data["text"]
            print(ra.values)
            sscore = []
            for i in ra.values:
                a = sentiment_scores(i)
                sscore.append(a)
            scounter = Counter(sscore)
            print(scounter)
            figfile = BytesIO()
            plt.title("Reviews")
            plt.bar(scounter.keys(), scounter.values(), color=(0.2, 0.4, 0.6, 0.6))
            # plt.savefig('new_plot.png')
            # figfile.seek(0)
            plt.savefig(figfile, format='png')
            figfile.seek(0)
            figdata_png = base64.b64encode(figfile.getvalue()).decode()
            result = "data:image/png;base64,"+figdata_png
            time.sleep(10)
            return render_template('index.html', name='Sentiment', url=result, select_sentiment=select_name)
            #return render_template('pre_nps.html')
    return render_template('pre_sentiment.html')


@application.route('/analytics', methods=['GET', 'POST'])
def nps_cal():
    select_name = "nps"
    if request.method == 'POST':
        # check if the post request has the file part
        topics = request.form.get("topic")
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            data = pd.read_csv(file)
            data["rating"] = data["rating"].fillna(1)
            ratings = data["rating"]
            print(ratings.values)
            categories = data["categories"]
            cat_counter = Counter(categories.values)
            nps_scores = {}
            for i in cat_counter:
                newdf = data[(data.categories == i)]
                ratings = newdf["rating"]
                counter = Counter(ratings.values)
                n = nps_score_calculation(counter)
                nps_scores[i] = n
            npvalues={k: v for k, v in sorted(nps_scores.items(), key=lambda item: item[1])}
            data_list = {'x': npvalues}
            with open(r'npsresults.csv', "w") as infile:
                writer = csv.DictWriter(infile, fieldnames=data_list["x"].keys())
                writer.writeheader()
                writer.writerow(data_list["x"])

            figfile = BytesIO()
            plt.plot(nps_scores.values())
            # plt.show()
            plt.savefig(figfile, format='png')
            figfile.seek(0)
            figdata_png = base64.b64encode(figfile.getvalue()).decode()
            result = "data:image/png;base64," + figdata_png
            # plt.savefig('/static/images/new_plot_1.png')
            time.sleep(10)
            return send_file("npsresults.csv", as_attachment=True)
            #return render_template('index.html', name='NPS', url=result, select_nps=select_name)
    return render_template('pre_nps.html')

@application.route('/timeseries', methods=['GET', 'POST'])
def timeseries_cal():
    select_name = "timeseries"
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            df = pd.read_csv(file, parse_dates=['reviews.date'])
            df1 = df.dropna()
            df1 = df1.sort_values(by="reviews.date")
            df1['reviews.date'] = pd.to_datetime(df1['reviews.date'], dayfirst=True)
            df1['year'] = df1['reviews.date'].dt.year
            df1['month'] = df1['reviews.date'].dt.month
            df1['day'] = df1['reviews.date'].dt.day

            df1 = df1.set_index(['reviews.date'])
            df2 = df1.resample('M').mean()
            df3 = df1.dropna()
            df_reviews = df3['rating'].groupby(df3.month).agg('count')
            nps_Months = []
            for i in range(1, 13):
                rslt_df = df3[df3['month'] == i]
                n = nps_score_calculation(rslt_df["rating"])
                nps_Months.append(n)
            df_recomend = pd.DataFrame(nps_Months, columns=['NPS'])

            figfile1 = BytesIO()
            fig, axes = plt.subplots(nrows=2, ncols=1)
            plt.subplots_adjust(wspace=1.5, hspace=1.0);

            df_reviews.plot(ax=axes[0], title='No.of Reviews per Month', legend=True)
            df_recomend.plot(ax=axes[1], title='Overall NPS(recommendation) per month', legend=True)
            plt.savefig(figfile1, format='png')
            figfile1.seek(0)
            figdata_png1 = base64.b64encode(figfile1.getvalue()).decode()
            result = "data:image/png;base64," + figdata_png1
            # plt.savefig('/static/images/new_plot_1.png')
            time.sleep(10)
            return render_template('index.html', name='timeseries', url=result, select_nps=select_name)

    return render_template('pre_timeseries.html')



