# Documentation

## Setuping the Data
The project folder coutains the folowing :
- **Stock_Data.json** a json file that contains the data of over 21 companies
- **2023Test.csv** a csv file that contains averages for all the dates of 2023 to proform prediction.

# Hadoop environnement
After finishing hadoop setup and making sure it's running correctly we can start setuping for our application.

**1. Push the Data into hdfs :**
```bash
hdfs dfs -mkdir /StockPrediction
hdfs dfs -put /StockPrediction/Stock_Data.json
hdfs dfs -put /StockPrediction/2023Test.csv
```

**2. Install the Required libraries :**
```bash
pip install -r requirements.txt
```

**3. Run Streamlit :**
```bash
streamlit run index.py
```

> 🎉 congratulations *YOUR* app is ready !

just visit the link genrated by streamlit and you can get rich.

<center>⚠ This is not a financial advice</center>