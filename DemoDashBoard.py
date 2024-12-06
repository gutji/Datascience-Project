import streamlit as st
import pandas as pd
import datetime
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide")
'''
url = "https://raw.githubusercontent.com/Chocobo11218/Datascience-Project/refs/heads/main/merged_data.csv"
response = requests.get(url, verify=False)  # Bypassing SSL verification
with open('merged_data.csv', 'wb') as f:
    f.write(response.content)
'''


# Load or create your dataset
df = pd.read_csv('merged_data.csv')


# Streamlit elements
st.title("Keyword Trends Dashboard")


# Sidebar
st.sidebar.header("Sidebar Controls")

year = st.sidebar.selectbox("Choose year", ("2018", "2019",'2020', '2021', '2022', '2023'))
def slicing(date):
  l = date.split('/')
  return l[-1]
df['publication_date_year'] = df['publication_date'].apply(lambda x:slicing(x))
df_year = df.loc[df['publication_date_year'] == year]
x = df_year['subjectArea'].value_counts().reset_index()


col1, col2 = st.columns(2)

with col1:
  st.header('Pie chart')
  fig = px.pie(x, 
               values= 'count', 
               names='subjectArea')
  st.plotly_chart(fig, use_container_width=True)

x


