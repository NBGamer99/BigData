import streamlit as st
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.regression import LinearRegression
import matplotlib.pyplot as plt
import datetime


st.title('Stock Market Prediction')

#Initializing Spark Session
st.header("Initializing a Spark Session")
data_load_state = st.text('Starting Session...')
spark = SparkSession.builder.master("local").appName("Stock Market Prediction").getOrCreate()
sc = spark.sparkContext
data_load_state.text("Done!")
st.write(sc.version)
st.write(sc.appName)
st.write(sc.uiWebUrl)
st.write(sc.master)
# st.markdown(result)

def preload(Symbol):
	df = spark.read.json('hdfs://localhost:9000/StockPrediction/Stock_Data.json', multiLine=True)
	# Select the "name" column
	name_column = df.select(Symbol)
	# Collect the values of the "name" column as a list of Row objects
	name_values = name_column.collect()
	# print(name_values)
	row = name_values[0].asDict()
	# print(name_values) # Prints the list of Row objects
	final = row[Symbol].asDict()
	return final

# @st.cache
def Load_data_json(Symbol):
	final = preload(Symbol)
	final_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
	def create_data(final):
		data = []
		for i in range(len(final["Date"])):
			data.append((final["Date"][i], final["Price History"]["open"][i],final["Price History"]["high"][i],final["Price History"]["low"][i], final["Price History"]["close"][i], final["volume"][i]))
		return data
	data = create_data(final)
	# print(data[0])
	final_df = spark.createDataFrame(data, final_columns)
	final_df = final_df.withColumn('Date', to_date(final_df['date'], 'yyyy-MM-dd'))
	return final_df

def InfoComp(Symbol):
	final = preload(Symbol)
	# Get Company Infos
	stock_info = {key: final[key] for key in ("Name", "Founder", 'Date of Birth', 'Industry', 'Market Value')}
	return stock_info


def Describe_Table(data):
	result = data.describe()
	result = result.select(result['summary'],
				format_number(result['Open'].cast('float'),2).alias('Open'),
				format_number(result['High'].cast('float'),2).alias('High'),
				format_number(result['Low'].cast('float'),2).alias('Low'),
				format_number(result['Close'].cast('float'),2).alias('Close'),
				result['Volume'].cast('float').alias('Volume')
				)
	return result

symbols = ["GOOGL", "AAPL", "AMZN", "MSFT", "META", "V", "INTC", "MA", "ORCL", "TSLA", "NFLX", "DIS", "MCD", "KO", "IBM", "CSCO", "AMD", "EA", "ABNB", "ACN", "ACN", "ADBE", "PYPL", "NVDA", "CRM", "UBER", "ATVI", "WMT", "SHOP"]

# Create the dropdown menu
st.subheader("Select Company Symbol")
selected_company = st.selectbox('', symbols)

# Display the selected string
# st.write('You selected:', selected_company)

# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
data = Load_data_json(selected_company)
# Notify the reader that the data was successfully loaded.
data_load_state.text("Done!")


# if st.checkbox('Show raw data'):
st.dataframe(data)

# Information about the company

st.subheader("Info About the company")
st.write(InfoComp(selected_company))

st.subheader("Calculating diffrent data statistics")

result = Describe_Table(data)

st.dataframe(result)

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

featureassembler = VectorAssembler(inputCols=["Open","High","Low"],outputCol="Features")
output = featureassembler.transform(data)

st.dataframe(output)
# output_test = featureassembler.transform(data)
output.show()

finilized_data = output.select("Date","Features","Close").sort("Date",ascending=True)

n_rows = finilized_data.count()

# # Calculate the split point
split_point = int(n_rows * 0.75)
train_data = finilized_data.limit(split_point)
finilized_data_desc = finilized_data.orderBy(finilized_data["Date"].desc())
test_data = finilized_data_desc.limit(n_rows - split_point)
test_data = test_data.orderBy(test_data["Date"].asc())


regressor = LinearRegression(featuresCol="Features", labelCol="Close")
regressor = regressor.fit(train_data)

pred = regressor.transform(test_data)
pred.select("Date","Features","Close","Prediction").show()
pred.count()

dates_ = finilized_data.select("Date").toPandas()
dates_ = dates_.values
close_ = finilized_data.select("Close").toPandas()
close_ = close_.values
preds = pred.select("Prediction").toPandas()
preds = preds.values
dates_preds = pred.select("Date").toPandas()
dates_preds = dates_preds.values


fig, ax = plt.subplots(figsize=(16, 8))

# Plot data1 and data2 on the subplot, with dates as the x-values
ax.plot(dates_, close_, label="Close")
ax.plot(dates_preds, preds , label="Prediction")

plt.legend()

# Set the x-axis label to "Date"
ax.set_xlabel("Date")

# Set the y-axis label to "Value"
ax.set_ylabel("Dollar Us")

# Show the plot
st.pyplot(fig)

import matplotlib.dates as mdates
import datetime
fig2, ax2 = plt.subplots(figsize=(16, 8))

# Plot data1 and data2 on the subplot, with dates as the x-values
ax2.plot(dates_, close_, label="Close")
ax2.plot(dates_preds, preds, label="Prediction")

plt.legend()

# ax2.set_xlim(["2015-10-02", "2016-12-30"])
ax2.set_xlim(mdates.date2num([datetime.datetime(2020, 10, 1), datetime.datetime(2023, 12, 30)]))


# ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

# Set the x-axis label to "Date"
ax2.set_xlabel("Date")

# Set the y-axis label to "Value"
ax2.set_ylabel("Dollar Us")

# Show the plot
st.pyplot(fig2)
# regressor.coefficients
# test_data = finilized_data_test
# test_data.tail(5)
# # Split the DataFrame into two smaller DataFrames
# train_data = finilized_data.limit(split_point)
# test_data = finilized_data.limit(n_rows - split_point)
# output_test.show()
# COLUMNS = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
# def get_Stocks(SYMBOL):
# 	STOCK = data[SYMBOL]
# 	PRICES = STOCK['Price History']
# 	print(type(PRICES['open']))

# 	DATA = [
# 	STOCK['Date'],
# 	PRICES['open'],
# 	PRICES['high'],
# 	PRICES['low'],
# 	PRICES['close'],
# 	STOCK['volume']
# 	]
# 	df = pd.DataFrame(np.array(DATA).T, columns=COLUMNS)
# 	return df


# DATE_COLUMN = 'date/time'
# DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
#          'streamlit-demo-data/uber-raw-data-sep14.csv.gz')

# @st.cache
# def load_data(nrows):
#     data = pd.read_csv(DATA_URL, nrows=nrows)
#     lowercase = lambda x: str(x).lower()
#     data.rename(lowercase, axis='columns', inplace=True)
#     data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
#     return data

# # Create a text element and let the reader know the data is loading.
# data_load_state = st.text('Loading data...')
# # Load 10,000 rows of data into the dataframe.
# data = load_data(10000)
# # Notify the reader that the data was successfully loaded.
# data_load_state.text("Done! (using st.cache)")

# if st.checkbox('Show raw data'):
#     st.subheader('Raw data')
#     st.write(data)

# st.subheader('Number of pickups by hour')
# hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]
# st.bar_chart(hist_values)

# hour_to_filter = st.slider('hour', 0, 23, 17)  # min: 0h, max: 23h, default: 17h
# filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]
# st.subheader(f'Map of all pickups at {hour_to_filter}:00')
# if st.checkbox('Show Map'):
#     st.map(filtered_data)