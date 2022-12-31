# **************** IMPORT PACKAGES ********************
from flask import Flask, render_template, request
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.style.use('ggplot')
from handlers import Results
import nltk

nltk.download('punkt')
from logger import logger
from multiprocessing.pool import ThreadPool

pool = ThreadPool(processes=10)
# Ignore Warnings
import warnings

warnings.filterwarnings("ignore")
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ***************** FLASK *****************************
app = Flask(__name__)


# To control caching so as to save and retrieve plot figs on client side
@app.after_request
def add_header(response):
    response.headers['Pragma'] = 'no-cache'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Expires'] = '0'
    return response


@app.route('/')
def signlog():
    return render_template('signlog.html')


@app.route('/verifypl')
def verifypl():
    return render_template('verifypl.html')


@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    key = request.args.get('symbol', '')
    fields = request.args.get('topic', '')
    choose = request.args.get('subject', '')
    print("key", key)
    print("fields", fields)
    print("choose", choose)
    # args = request.args
    # name = args.get('symbol')
    async_result = pool.apply_async(Results.dashboard, (key, fields, choose,))
    result = async_result.get()
    print("result", result.head())
    # return render_template('dashboard.html',tables=[result.to_html(classes='data')], titles=result.columns.values)
    return render_template('dashboard.html', column_names=result.columns.values, row_data=list(result.values.tolist()),
                           zip=zip)


# @app.route('/dashboard_after',methods=['POST'])
# def dashboard_after():
#     # key=request.args.get('search_key','')
#     nm=request.form['symbol']
#     print('nm',nm)
#     async_result = pool.apply_async(Results.dashboard, (nm,))
#     result = async_result.get()
#     print("result",result.head())
#     return render_template('dashboard.html')


@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/about_us')
def about_us():
    return render_template('about_us.html')


@app.route('/contact_us')
def contact_us():
    return render_template('contact_us.html')


@app.route('/contact_us_mail_sending', methods=['POST'])
def contact_us_mail_sending():
    global result
    try:
        name = request.form['name']
        email = request.form['email']
        subject = request.form['subject']
        contact_message = request.form['message']
        if not name:
            raise Exception("please enter valid name")
        if not email:
            raise Exception("please enter valid email")
        if not subject:
            raise Exception("please enter valid subject")
        if not contact_message:
            raise Exception("please enter valid message")
        async_result = pool.apply_async(Results.contact_us_mail_sending, (name, email, subject, contact_message,))
        result = async_result.get()
        return render_template('contact_us.html', status=result['status'])
    except Exception as e:
        return render_template('contact_us.html', status=result['status'])


@app.route('/insertintostock', methods=['GET', 'POST'])
def insertintostock():
    nm = request.form['nm']
    # **************GET DATA ***************************************
    quote = nm
    # Try-except to check if valid stock symbol
    try:
        # Results.get_historical(quote)
        async_result = pool.apply_async(Results.get_historical, (quote,))
        result = async_result.get()       
    except:
        return render_template('index.html', not_found=True)
    else:

        # ************** PREPROCESSUNG ***********************
        df = pd.read_csv('' + quote + '.csv')
        logger.info("##############################################################################")
        logger.info("Today's" + quote + "Stock Data: ")
        today_stock = df.iloc[-1:]
        logger.info(today_stock)
        logger.info("##############################################################################")
        df = df.dropna()
        code_list = []
        for i in range(0, len(df)):
            code_list.append(quote)
        df2 = pd.DataFrame(code_list, columns=['Code'])
        df2 = pd.concat([df2, df], axis=1)
        df = df2

        async_result = pool.apply_async(Results.ARIMA_ALGO, (df,))
        arima_pred, error_arima = async_result.get()

        async_result = pool.apply_async(Results.LSTM_ALGO, (df,))
        lstm_pred, error_lstm = async_result.get()

        async_result = pool.apply_async(Results.LIN_REG_ALGO, (df,))
        df, lr_pred, forecast_set, mean, error_lr = async_result.get()

        async_result = pool.apply_async(Results.retrieving_tweets_polarity_stock, (quote,))
        polarity, tw_list, tw_pol, pos, neg, neutral, dates, final_currency = async_result.get()

        async_result = pool.apply_async(Results.recommending, (df, polarity, today_stock, mean,))
        idea, decision = async_result.get()

        logger.info("Forecasted Prices for Next 7 days:")
        logger.info(forecast_set)
        forecast = []
        for row in forecast_set:
            forecast.append(row[0])
        today_stock = today_stock.round(2)
        os.remove('' + quote + '.csv')
        return render_template('results.html', quote=quote,
                               lstm_pred=round(lstm_pred, 2),
                               open_s=today_stock['Open'].to_string(index=False),
                               close_s=today_stock['Close'].to_string(index=False),
                               adj_close=today_stock['Adj Close'].to_string(index=False),
                               tw_list=tw_list, tw_pol=tw_pol, idea=idea, decision=decision,
                               high_s=today_stock['High'].to_string(index=False),
                               low_s=today_stock['Low'].to_string(index=False),
                               vol=today_stock['Volume'].to_string(index=False),
                               forecast_set=forecast, dates=dates, final_currency=final_currency, error_lstm=round(error_lstm, 2))


if __name__ == '__main__':
    app.run()
