import streamlit as st
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType
from pyspark.sql.functions import *
from pyspark.ml.regression import LinearRegression
import matplotlib.pyplot as plt
import datetime
import time

st.title('Stock Market Prediction')

#Initializing Spark Session
st.header("Initializing a Spark Session")
data_load_state = st.text('Starting Session...')
spark = SparkSession.builder.master("local").appName("Stock Market Prediction").getOrCreate()
sc = spark.sparkContext
data_load_state.text("Done!")
st.write(sc.version)
st.write(sc.appName)
# st.write(sc.uiWebUrl)
st.write("http://134.209.242.40:4040")
st.write(sc.master)
# st.markdown(result)

def preload(Symbol):
	df = spark.read.json('hdfs://hadoop-master:9000/StockPrediction/Stock_Data.json', multiLine=True)
	symbols = df.columns
	# Select the "name" column
	name_column = df.select(Symbol)
	# Collect the values of the "name" column as a list of Row objects
	name_values = name_column.collect()
	# print(name_values)
	row = name_values[0].asDict()
	# print(name_values) # Prints the list of Row objects
	final = row[Symbol].asDict()
	return final, symbols

# @st.cache
def Load_data_json(Symbol):
	final, symbols = preload(Symbol)
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
	return final_df, symbols

def InfoComp(Symbol):
	final, _ = preload(Symbol)
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

# symbols = ["GOOGL", "AAPL", "AMZN", "MSFT", "META", "V", "INTC", "MA", "ORCL", "TSLA", "NFLX", "DIS", "MCD", "KO", "IBM", "CSCO", "AMD", "EA", "ABNB", "ACN", "ACN", "ADBE", "PYPL", "NVDA", "CRM", "UBER", "ATVI", "WMT", "SHOP"]
_ ,symbols = Load_data_json("GOOGL")

# Create the dropdown menu
st.subheader("Select Company Symbol")
selected_company = st.selectbox('', symbols)

# Display the selected string
# st.write('You selected:', selected_company)

my_bar = st.progress(0)
data_load_state = st.text('Loading data...')
data, _ = Load_data_json(selected_company)



for percent_complete in range(100):
	time.sleep(0.03)
	my_bar.progress(percent_complete + 1)

# Create a text element and let the reader know the data is loading.
# Load 10,000 rows of data into the dataframe.
# Notify the reader that the data was successfully loaded.
if percent_complete == 99:
	st.success(f'Data Loaded !', icon="✅")
	# data_load_state = st.text('Done !...')



# if st.checkbox('Show raw data'):
st.dataframe(data)

# Information about the company

st.subheader("Info About the company")
st.write(InfoComp(selected_company))

st.subheader("Calculating data statistics")

result = Describe_Table(data)

st.dataframe(result)

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

featureassembler = VectorAssembler(inputCols=["Open","High","Low"],outputCol="Features")
output = featureassembler.transform(data)

st.subheader("Creating Traing data Features :")
st.dataframe(output)
output.show()

finilized_data = output.select("Date","Features","Close").sort("Date",ascending=True)

n_rows = finilized_data.count()


# # Calculate the split point
split_point = int(n_rows * 0.75)
train_data = finilized_data.limit(split_point)
finilized_data_desc = finilized_data.orderBy(finilized_data["Date"].desc())




if st.checkbox('Predict For 2023'):
	schema = StructType().add("Date",'date').add("Open",'float').add("High",'float').add("Low",'float').add("Close",'float').add("Adj Close", 'float').add("Volume",'float')
	test_data = spark.read.csv('hdfs://hadoop-master:9000/StockPrediction/2023Test.csv',schema=schema,header=True,multiLine=True)
	output2 = featureassembler.transform(test_data)
	finilized_data2 = output2.select("Date","Features","Close").sort("Date",ascending=True)
	test_data = finilized_data2
	st.dataframe(test_data)
else:
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


st.subheader(f"Stock Market History for {selected_company}:")
fig, ax = plt.subplots(figsize=(16, 8))

# Plot data1 and data2 on the subplot, with dates as the x-values
ax.plot(dates_, close_, label="Close")

plt.legend()

# Set the x-axis label to "Date"
ax.set_xlabel("Date")

# Set the y-axis label to "Value"
ax.set_ylabel("Dollar Us")

# Show the plot
st.pyplot(fig)

import matplotlib.dates as mdates
import datetime

st.subheader(f"Stock Market {selected_company} Prediction :")
fig2, ax2 = plt.subplots(figsize=(16, 8))

# Plot data1 and data2 on the subplot, with dates as the x-values
ax2.plot(dates_, close_, label="Close")
ax2.plot(dates_preds, preds, label="Prediction")

plt.legend()

date_start = d = st.date_input("Start Date",datetime.date(2020, 10, 1))

date_end = d = st.date_input("End Date",datetime.date(2023, 12, 30))


ax2.set_xlim(mdates.date2num([date_start, date_end]))


# ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

# Set the x-axis label to "Date"
ax2.set_xlabel("Date")

# Set the y-axis label to "Value"
ax2.set_ylabel("Dollar Us")

# Show the plot
st.pyplot(fig2)
