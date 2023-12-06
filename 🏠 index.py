import streamlit as st
import json
import time

st.title("Stock Market Prediction Project")

url = 'https://www.youtube.com/embed/p7HKvqRI_Bo'
st.video(url)

st.subheader("Abstraction:")
tab1, tab2 = st.tabs(["Abstraction", "Schematic"])
tab1.write(r"""Le big data est la collecte de grandes quantités de données provenant de sources traditionnelles et numériques afin de déterminer les tendances et les patterns. La quantité et la diversité des données informatiques augmentent de manière exponentielle pour de nombreuses raisons. Par exemple, les détaillants créent de vastes bases de données de l'activité de vente des clients. Les organisations s'occupent de la logistique, des services financiers et les réseaux sociaux publics partagent une grande quantité d'opinions liées aux prix de vente et aux produits. Les défis du big data incluent le volume et la diversité des données structurées et non structurées. Dans ce projet, nous avons utilisé plusieurs modèles d'apprentissage automatique mis en œuvre grâce à Spark en utilisant PySpark et Keras Tensorflow, qui sont évolutifs, rapides, facilement intégrables à d'autres outils et qui ont une meilleure performance que les modèles traditionnels. Nous avons étudié les actions de plus de 25 entreprises du top, dont les données comprennent les prix historiques des actions, avec des modèles MLlib tels que la régression linéaire, nous avons entraîné les modèles avec 70% des données et essayé de prédire les 30% restants.""")
tab2.image("./Assets/schema.png")

st.subheader("Stock Market:")
st.write(r"""Le marché boursier est un lieu où les entreprises peuvent émettre et échanger des actions (parts de l'entreprise) et où les investisseurs peuvent acheter et vendre ces actions. Le prix des actions est déterminé par l'offre et la demande sur le marché, et il peut fluctuer en fonction de l'état de l'économie, des résultats financiers de l'entreprise, de l'actualité et de nombreux autres facteurs. Le marché boursier est un moyen pour les entreprises de lever des fonds en vendant des actions, et pour les investisseurs de gagner de l'argent en achetant et en vendant des actions à un prix plus élevé que celui auquel elles ont été achetées.""")
st.write("Voici quelques mots-clés couramment utilisés dans le contexte du marché boursier: ")
st.markdown("- ***Open***: Le prix d'ouverture d'une action est le prix auquel elle a été négociée lors de la première transaction de la journée de trading. Cela peut être influencé par de nombreux facteurs, tels que les résultats financiers de l'entreprise, l'actualité, les prévisions de l'économie, etc.")
st.markdown("- ***Close***: Le prix de fermeture d'une action est le prix auquel elle a été négociée lors de la dernière transaction de la journée de trading. Cela peut être influencé par les mêmes facteurs que le prix d'ouverture, ainsi que par l'évolution du cours de l'action au cours de la journée.")
st.markdown("- ***High***: Le prix le plus haut d'une action est le cours auquel elle a été négociée à son plus haut niveau au cours de la journée de trading. Cela peut être influencé par la demande pour l'action, ainsi que par l'offre et la demande sur le marché dans son ensemble.")
st.markdown("- ***Low***: Le prix le plus bas d'une action est le cours auquel elle a été négociée à son plus bas niveau au cours de la journée de trading. Cela peut être influencé par la demande pour l'action, ainsi que par l'offre et la demande sur le marché dans son ensemble.")
st.markdown("- ***Volume***: Le volume des transactions d'actions est le nombre total d'actions qui ont été achetées et vendues au cours de la journée de trading. Le volume peut être utilisé pour mesurer l'intérêt pour une action et prévoir l'évolution de son cours à l'avenir. Un volume élevé peut indiquer une forte activité sur le marché et une grande volatilité du cours de l'action.")

st.subheader("Data Set")
st.write("Dans ce projet, nous avons utilisé une seule source de données, qui est un fichier JSON nommé Stock_data.json qui contient l'historique des prix du Stock Market de +25 grandes entreprises sur les 5 dernières années.")
with open('./Data_schema.json','r') as f:
  data = json.load(f)
st.json(data)
# st.image('./Assets/frame_shema.png')

st.subheader("Data Processing")
st.selectbox("STOCK LIST :", tuple(data.keys()))
st.markdown("- ***Phase 1***: Extraire les informations de l'entreprise chaque fois que le menu déroulant est modifié.")
st.markdown("- ***Phase 2***: Extraire l'historique des prix `[Open, Close, High, Low]` plus que le `[Volume]` de l'entreprise en chaque date dans `[Date]` chaque fois que le menu déroulant est modifié.")
st.markdown("***Tools*** : pandas DataFrame et PySpark DataFrame")

st.subheader("Tools :")
st.markdown("- ***Apache Spark*** : est un moteur de traitement de données distribué open source conçu pour le traitement rapide et flexible de grandes quantités de données.")
st.markdown("- ***PySpark*** : API Python pour Apache Spark, un puissant moteur de traitement de données open source pour les big data.PySpark vous permet d'écrire des programmes Spark en Python, qui peuvent être utilisés pour construire une grande variété d'applications pour le traitement de données, l'apprentissage automatique et le traitement de graphes.")
st.markdown("- ***Hadoop Distributed File System (HDFS)***: est un système de fichiers distribué qui fait partie du projet Apache Hadoop. Il est conçu pour stocker et gérer de grandes quantités de données dans un environnement de calcul distribué.")
st.markdown("- ***Streamlit***: est un cadre de développement d'applications Web pour les scientifiques de données et les ingénieurs. Il permet de créer rapidement des applications interactives en Python et de les déployer facilement sur le Web.")
st.markdown("- ***TenserFlow*** : est un logiciel de bibliothèque open source destiné à l'apprentissage automatique en grande échelle.")


