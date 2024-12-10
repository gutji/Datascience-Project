import streamlit as st
import pandas as pd
import datetime
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go


st.set_page_config(layout="wide")


df = pd.read_csv('merged_data.csv')

df_keyword = pd.read_csv('top_keywords_by_field.csv')



st.title("Keyword Trends Dashboard")



st.sidebar.header("Sidebar Controls")

year = st.sidebar.selectbox("Choose year", ("2018", "2019",'2020', '2021', '2022', '2023'))
subject = st.sidebar.selectbox("Choose field",  df_keyword['field_of_study'])

field = df_keyword.loc[df_keyword['field_of_study'] == subject]
key = field['top_keyword']

def slicing(date):
  l = date.split('/')
  return l[-1]
df['publication_date_year'] = df['publication_date'].apply(lambda x:slicing(x))
df_year = df.loc[df['publication_date_year'] == year]
x = df_year['subjectArea'].value_counts().reset_index()


col1, col2 = st.columns(2)

with col1:
  st.markdown("[Click here to visit Streamlit](https://streamlit.io)")
  st.header('Pie chart')
  fig = px.pie(x, 
               values= 'count', 
               names='subjectArea',
               template="gridon"
              )
  st.plotly_chart(fig, use_container_width=True)

with col2:
  st.write(key)
  



