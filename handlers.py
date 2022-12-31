import smtplib
from email.header import Header
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.style.use('ggplot')
import math
from datetime import datetime, timedelta
import pandas_datareader as web
import yfinance as yf
import tweepy
import preprocessor as p
import re
from sklearn.linear_model import LinearRegression
from textblob import TextBlob
import constants as ct
from Tweet import Tweet
import nltk
import statsmodels.api as smapi

nltk.download('punkt')
from countryinfo import CountryInfo

# Ignore Warnings
import warnings

warnings.filterwarnings("ignore")
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from logger import logger


class Results:
    # **************** CONTACTUS MAIL SENDING ***************************
    @staticmethod
    def contact_us_mail_sending(name, email, subject, contact_message):
        try:
            sender_address = ''
            sender_password = ''
            receiver_address = ''
            recipient_name = 'Tarun Madamanchi'
            message = MIMEMultipart()
            from_header = "SF System"
            message['From'] = formataddr((str(Header(from_header, 'utf-8')), sender_address))
            message['To'] = formataddr((str(Header(recipient_name, 'utf-8')), receiver_address))
            message['Subject'] = subject
            to_address = []

            to_address = receiver_address

            html = f"""\
            <html>
              <body >
                <p>Hi,<br>
                <br>
                name: {name}<br><br>
                email: {email}<br>
                subject: {subject}<br>
                message: {contact_message}<br>
                </p>
              </body>
            </html>
            """
            body_html = MIMEText(html, 'html')
            message.attach(body_html)
            session = smtplib.SMTP('smtp.gmail.com', 587)
            # session = smtplib.SMTP_SSL('smtp.gmail.com', 587)
            session.ehlo()
            session.starttls()
            session.login(sender_address, sender_password)
            text = message.as_string()
            session.sendmail(sender_address, to_address, text)
            session.quit()
            logger.info("name :" + name + " email :" + email + " subject :" + subject + " message :" + contact_message)
            return {"status": "success"}
        except Exception as e:
            logger.error("exception raised" + str(e))
            return {"status": str(e)}

    # **************** SEARCH FOR DETAILS ********************************
    @staticmethod
    def dashboard(quote, fields, choose):
        global stock_full_form
        Yahoo_Finance_Ticker_Symbols = pd.read_csv('Yahoo-Finance-Ticker-Symbols.csv')
        crypto = pd.read_csv('cryptocurrencies - cryptocurrencies.csv')
        stock_full_form = Yahoo_Finance_Ticker_Symbols[Yahoo_Finance_Ticker_Symbols['Ticker'] == quote]
        if fields == 'Ticker' and choose == 'Stock':
            stock_full_form = Yahoo_Finance_Ticker_Symbols[Yahoo_Finance_Ticker_Symbols['Ticker'] == quote]
        elif fields == 'Name' and choose == 'Stock':
            stock_full_form = Yahoo_Finance_Ticker_Symbols[Yahoo_Finance_Ticker_Symbols['Name'] == quote]
        elif fields == 'Exchange' and choose == 'Stock':
            stock_full_form = Yahoo_Finance_Ticker_Symbols[Yahoo_Finance_Ticker_Symbols['Exchange'] == quote]
        elif fields == 'Category_Name' and choose == 'Stock':
            stock_full_form = Yahoo_Finance_Ticker_Symbols[Yahoo_Finance_Ticker_Symbols['Category_Name'] == quote]
        elif fields == 'Country' and choose == 'Stock':
            stock_full_form = Yahoo_Finance_Ticker_Symbols[Yahoo_Finance_Ticker_Symbols['Country'] == quote]
        elif fields == 'Ticker' and choose == 'Crypto':
            stock_full_form = crypto[crypto['Ticker'] == quote]
        elif fields == 'Name' and choose == 'Crypto':
            stock_full_form = crypto[crypto['Name'] == quote]

        return stock_full_form

    # **************** FUNCTIONS TO FETCH DATA ***************************
    @staticmethod
    def get_historical(quote):
        global ticker
        ticker = quote
        end = datetime.now()
        start = datetime(end.year - 5, end.month, end.day)        
        logger.info(ticker)
        #data = web.DataReader(ticker, 'yahoo', start, end)
        data = yf.download(quote, start=start, end=end)
        logger.info(data)
        df = pd.DataFrame(data=data)
        df.to_csv('' + ticker + '.csv')
        if df.empty:
            ts = TimeSeries(key='N6A6QT6IBFJOPJ70', output_format='pandas')
            data, meta_data = ts.get_daily_adjusted(symbol='NSE:' + ticker, outputsize='full')
            # Format df
            # Last 2 yrs rows => 502, in ascending order => ::-1
            data = data.head(503).iloc[::-1]
            data = data.reset_index()
            # Keep Required cols only
            df = pd.DataFrame()
            df['Date'] = data['date']
            df['Open'] = data['1. open']
            df['High'] = data['2. high']
            df['Low'] = data['3. low']
            df['Close'] = data['4. close']
            df['Adj Close'] = data['5. adjusted close']
            df['Volume'] = data['6. volume']
            df.to_csv('' + quote + '.csv', index=False)
        return 

    # ******************** ARIMA SECTION ********************
    @staticmethod
    def ARIMA_ALGO(df):
        uniqueVals = df["Code"].unique()
        len(uniqueVals)
        df = df.set_index("Code")

        # for daily basis
        def parser(x):
            return datetime.strptime(x, '%Y-%m-%d')

        def arima_model(train, test):
            history = [x for x in train]
            predictions = list()
            for t in range(len(test)):
                model = smapi.tsa.arima.ARIMA(history, order=(6, 1, 0))                
                model_fit = model.fit()
                output = model_fit.forecast()
                yhat = output[0]
                predictions.append(yhat)
                obs = test[t]
                history.append(obs)
            return predictions

        for company in uniqueVals[:10]:
            data = (df.loc[company, :]).reset_index()
            data['Price'] = data['Close']
            Quantity_date = data[['Price', 'Date']]
            Quantity_date.index = Quantity_date['Date'].map(lambda x: parser(x))
            Quantity_date['Price'] = Quantity_date['Price'].map(lambda x: float(x))
            Quantity_date = Quantity_date.fillna(Quantity_date.bfill())
            Quantity_date = Quantity_date.drop(['Date'], axis=1)
            fig = plt.figure(figsize=(7.2, 4.8), dpi=65)
            plt.plot(Quantity_date)
            plt.savefig('static/Trends.png')
            plt.close(fig)

            quantity = Quantity_date.values
            size = int(len(quantity) * 0.80)
            train, test = quantity[0:size], quantity[size:len(quantity)]
            # fit in model
            predictions = arima_model(train, test)

            # plot graph
            fig = plt.figure(figsize=(7.2, 4.8), dpi=65)
            plt.plot(test, label='Actual Price')
            plt.plot(predictions, label='Predicted Price')
            plt.legend(loc=4)
            plt.savefig('static/ARIMA.png')
            plt.close(fig)
            logger.info("##############################################################################")
            arima_pred = predictions[-2]
            logger.info(f"""Tomorrow's {ticker}  Closing Price Prediction by ARIMA:{arima_pred}""")
            # rmse calculation
            error_arima = math.sqrt(mean_squared_error(test, predictions))
            logger.info(f"""ARIMA RMSE:{error_arima}""")
            logger.info("##############################################################################")
            return arima_pred, error_arima

    # ************* LSTM SECTION **********************
    @staticmethod
    def LSTM_ALGO(df):
        # Split data into training set and test set
        dataset_train = df.iloc[0:int(0.8 * len(df)), :]
        dataset_test = df.iloc[int(0.8 * len(df)):, :]
        ############# NOTE #################
        # TO PREDICT STOCK PRICES OF NEXT N DAYS, STORE PREVIOUS N DAYS IN MEMORY WHILE TRAINING
        # HERE N=7
        ###dataset_train=pd.read_csv('Google_Stock_Price_Train.csv')
        training_set = df.iloc[:, 4:5].values  # 1:2, to store as numpy array else Series obj will be stored
        # select cols using above manner to select as float64 type, view in var explorer

        # Feature Scaling
        from sklearn.preprocessing import MinMaxScaler
        sc = MinMaxScaler(feature_range=(0, 1))  # Scaled values btween 0,1
        training_set_scaled = sc.fit_transform(training_set)
        # In scaling, fit_transform for training, transform for test

        # Creating data stucture with 7 timesteps and 1 output.
        # 7 timesteps meaning storing trends from 7 days before current day to predict 1 next output
        X_train = []  # memory with 7 days from day i
        y_train = []  # day i
        for i in range(7, len(training_set_scaled)):
            X_train.append(training_set_scaled[i - 7:i, 0])
            y_train.append(training_set_scaled[i, 0])
        # Convert list to numpy arrays
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_forecast = np.array(X_train[-1, 1:])
        X_forecast = np.append(X_forecast, y_train[-1])
        # Reshaping: Adding 3rd dimension
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))  # .shape 0=row,1=col
        X_forecast = np.reshape(X_forecast, (1, X_forecast.shape[0], 1))
        # For X_train=np.reshape(no. of rows/samples, timesteps, no. of cols/features)

        # Building RNN
        from tensorflow import keras
        # from tensorflow.keras.models import models
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.layers import Dropout
        from keras.layers import LSTM

        # Initialise RNN
        regressor = Sequential()

        # Add first LSTM layer
        regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        # units=no. of neurons in layer
        # input_shape=(timesteps,no. of cols/features)
        # return_seq=True for sending recc memory. For last layer, retrun_seq=False since end of the line
        regressor.add(Dropout(0.1))

        # Add 2nd LSTM layer
        regressor.add(LSTM(units=50, return_sequences=True))
        regressor.add(Dropout(0.1))

        # Add 3rd LSTM layer
        regressor.add(LSTM(units=50, return_sequences=True))
        regressor.add(Dropout(0.1))

        # Add 4th LSTM layer
        regressor.add(LSTM(units=50))
        regressor.add(Dropout(0.1))

        # Add o/p layer
        regressor.add(Dense(units=1))

        # Compile
        regressor.compile(optimizer='adam', loss='mean_squared_error')

        # Training
        regressor.fit(X_train, y_train, epochs=25, batch_size=32)
        # For lstm, batch_size=power of 2

        # Testing
        ###dataset_test=pd.read_csv('Google_Stock_Price_Test.csv')
        real_stock_price = dataset_test.iloc[:, 4:5].values

        # To predict, we need stock prices of 7 days before the test set
        # So combine train and test set to get the entire data set
        dataset_total = pd.concat((dataset_train['Close'], dataset_test['Close']), axis=0)
        testing_set = dataset_total[len(dataset_total) - len(dataset_test) - 7:].values
        testing_set = testing_set.reshape(-1, 1)
        # -1=till last row, (-1,1)=>(80,1). otherwise only (80,0)

        # Feature scaling
        testing_set = sc.transform(testing_set)

        # Create data structure
        X_test = []
        for i in range(7, len(testing_set)):
            X_test.append(testing_set[i - 7:i, 0])
            # Convert list to numpy arrays
        X_test = np.array(X_test)

        # Reshaping: Adding 3rd dimension
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        # Testing Prediction
        predicted_stock_price = regressor.predict(X_test)

        # Getting original prices back from scaled values
        predicted_stock_price = sc.inverse_transform(predicted_stock_price)
        fig = plt.figure(figsize=(7.2, 4.8), dpi=65)
        plt.plot(real_stock_price, label='Actual Price')
        plt.plot(predicted_stock_price, label='Predicted Price')

        plt.legend(loc=4)
        plt.savefig('static/LSTM.png')
        plt.close(fig)

        error_lstm = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))

        # Forecasting Prediction
        forecasted_stock_price = regressor.predict(X_forecast)

        # Getting original prices back from scaled values
        forecasted_stock_price = sc.inverse_transform(forecasted_stock_price)

        lstm_pred = forecasted_stock_price[0, 0]
        logger.info("##############################################################################")
        logger.info(f"""Tomorrow's {ticker} Closing Price Prediction by LSTM: {lstm_pred}""")
        logger.info(f"""LSTM RMSE:{error_lstm}""")
        logger.info("##############################################################################")
        return lstm_pred, error_lstm

    # ***************** LINEAR REGRESSION SECTION ******************
    @staticmethod
    def LIN_REG_ALGO(df):
        # No of days to be forcasted in future
        forecast_out = int(7)
        # Price after n days
        df['Close after n days'] = df['Close'].shift(-forecast_out)
        # New df with only relevant data
        df_new = df[['Close', 'Close after n days']]

        # Structure data for train, test & forecast
        # lables of known data, discard last 35 rows
        y = np.array(df_new.iloc[:-forecast_out, -1])
        y = np.reshape(y, (-1, 1))
        # all cols of known data except lables, discard last 35 rows
        X = np.array(df_new.iloc[:-forecast_out, 0:-1])
        # Unknown, X to be forecasted
        X_to_be_forecasted = np.array(df_new.iloc[-forecast_out:, 0:-1])

        # Traning, testing to plot graphs, check accuracy
        X_train = X[0:int(0.8 * len(df)), :]
        X_test = X[int(0.8 * len(df)):, :]
        y_train = y[0:int(0.8 * len(df)), :]
        y_test = y[int(0.8 * len(df)):, :]

        # Feature Scaling===Normalization
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        X_to_be_forecasted = sc.transform(X_to_be_forecasted)

        # Training
        clf = LinearRegression(n_jobs=-1)
        clf.fit(X_train, y_train)

        # Testing
        y_test_pred = clf.predict(X_test)
        y_test_pred = y_test_pred * (1.04)
        import matplotlib.pyplot as plt2
        fig = plt2.figure(figsize=(7.2, 4.8), dpi=65)
        plt2.plot(y_test, label='Actual Price')
        plt2.plot(y_test_pred, label='Predicted Price')

        plt2.legend(loc=4)
        plt2.savefig('static/LR.png')
        plt2.close(fig)

        error_lr = math.sqrt(mean_squared_error(y_test, y_test_pred))

        # Forecasting
        forecast_set = clf.predict(X_to_be_forecasted)
        forecast_set = forecast_set * (1.04)
        mean = forecast_set.mean()
        lr_pred = forecast_set[0, 0]
        logger.info("##############################################################################")
        logger.info(f"""Tomorrow's {ticker} Closing Price Prediction by Linear Regression: {lr_pred}""")
        logger.info(f"""Linear Regression RMSE:{error_lr}""")
        logger.info("##############################################################################")
        return df, lr_pred, forecast_set, mean, error_lr

    # **************** SENTIMENT ANALYSIS **************************
    @staticmethod
    def retrieving_tweets_polarity_stock(symbol):
        global country
        final_currency = ['USD']
        Yahoo_Finance_Ticker_Symbols = pd.read_csv('Yahoo-Finance-Ticker-Symbols.csv')
        crypto = pd.read_csv('cryptocurrencies - cryptocurrencies.csv')
        dates = []
        stock_full_form = Yahoo_Finance_Ticker_Symbols[Yahoo_Finance_Ticker_Symbols['Ticker'] == symbol]
        if stock_full_form.shape[0] == 0:
            for i in range(0, 10):
                date = (datetime.today() + timedelta(days=i)).strftime("%Y-%m-%d")
                if len(dates) >= 7:
                    break
                else:
                    dates.append(date)
            stock_full_form = crypto[crypto['Ticker'] == symbol]
            symbol = stock_full_form['Name'].to_list()[0][0:12]
        else:
            for i in range(0, 10):
                date = (datetime.today() + timedelta(days=i)).strftime("%Y-%m-%d")
                is_weekend = datetime.strptime(date, '%Y-%m-%d').weekday() > 4
                if len(dates) >= 7:
                    break
                elif is_weekend:
                    pass
                else:
                    dates.append(date)

            symbol = stock_full_form['Name'].to_list()[0][0:12]
            country = stock_full_form['Country'].to_list()[0][0:12]
            currency = CountryInfo(country)
            final_currency = currency.currencies()
            print("final_currency", final_currency)

        # currency=CountryInfo(country)
        # final_currency=currency.currencies()
        auth = tweepy.OAuthHandler(ct.consumer_key, ct.consumer_secret)
        auth.set_access_token(ct.access_token, ct.access_token_secret)
        user = tweepy.API(auth)

        tweets = tweepy.Cursor(user.search_tweets, q=symbol, tweet_mode='extended', lang='en', exclude_replies=True).items(
            ct.num_of_tweets)

        tweet_list = []  # List of tweets alongside polarity
        global_polarity = 0  # Polarity of all tweets === Sum of polarities of individual tweets
        tw_list = []  # List of tweets only => to be displayed on web page
        # Count Positive, Negative to plot pie chart
        pos = 0  # Num of pos tweets
        neg = 1  # Num of negative tweets
        for tweet in tweets:
            count = 20  # Num of tweets to be displayed on web page
            # Convert to Textblob format for assigning polarity
            tw2 = tweet.full_text
            tw = tweet.full_text
            # Clean
            tw = p.clean(tw)
            # print("-------------------------------CLEANED TWEET-----------------------------")
            # print(tw)
            # Replace &amp; by &
            tw = re.sub('&amp;', '&', tw)
            # Remove :
            tw = re.sub(':', '', tw)
            # print("-------------------------------TWEET AFTER REGEX MATCHING-----------------------------")
            # print(tw)
            # Remove Emojis and Hindi Characters
            tw = tw.encode('ascii', 'ignore').decode('ascii')

            # print("-------------------------------TWEET AFTER REMOVING NON ASCII CHARS-----------------------------")
            # print(tw)
            blob = TextBlob(tw)
            polarity = 0  # Polarity of single individual tweet
            for sentence in blob.sentences:

                polarity += sentence.sentiment.polarity
                if polarity > 0:
                    pos = pos + 1
                if polarity < 0:
                    neg = neg + 1

                global_polarity += sentence.sentiment.polarity
            if count > 0:
                tw_list.append(tw2)

            tweet_list.append(Tweet(tw, polarity))
            count = count - 1
        if len(tweet_list) != 0:
            global_polarity = global_polarity / len(tweet_list)
        else:
            global_polarity = global_polarity
        neutral = ct.num_of_tweets - pos - neg
        if neutral < 0:
            neg = neg + neutral
            neutral = 20
        logger.info("##############################################################################")
        logger.info(f"""Positive Tweets :{pos}Negative Tweets :{neg}Neutral Tweets :{neutral}""")
        logger.info("##############################################################################")
        labels = ['Positive', 'Negative', 'Neutral']
        sizes = [pos, neg, neutral]
        explode = (0, 0, 0)
        fig = plt.figure(figsize=(7.2, 4.8), dpi=65)
        fig1, ax1 = plt.subplots(figsize=(7.2, 4.8), dpi=65)
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle=90)
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax1.axis('equal')
        plt.tight_layout()
        plt.savefig('static/SA.png')
        plt.close(fig)
        # plt.show()
        if global_polarity > 0:
            logger.info("##############################################################################")
            logger.info("Tweets Polarity: Overall Positive")
            logger.info("##############################################################################")
            tw_pol = "Overall Positive"
        else:
            logger.info("##############################################################################")
            logger.info("Tweets Polarity: Overall Negative")
            logger.info("##############################################################################")
            tw_pol = "Overall Negative"
        return global_polarity, tw_list, tw_pol, pos, neg, neutral, dates, final_currency

    # **************** RECOMMENDING **************************
    @staticmethod
    def recommending(df, global_polarity, today_stock, mean):
        global idea, decision
        if today_stock.iloc[-1]['Close'] < mean:
            if global_polarity > 0:
                idea = "RISE"
                decision = "BUY"
                logger.info("##############################################################################")
                logger.info(
                    "According to the ML Predictions and Sentiment Analysis of Tweets, a" + idea + "in" +
                    ticker + "stock is expected => " + decision)
            elif global_polarity <= 0:
                idea = "FALL"
                decision = "SELL"
                logger.info("##############################################################################")
                logger.info(
                    "According to the ML Predictions and Sentiment Analysis of Tweets, a" + idea + "in" +
                    ticker + "stock is expected => " + decision)
        else:
            idea = "FALL"
            decision = "SELL"
            logger.info("##############################################################################")
            logger.info(
                "According to the ML Predictions and Sentiment Analysis of Tweets, a" + idea + "in" +
                ticker + "stock is expected => " + decision)
        return idea, decision
