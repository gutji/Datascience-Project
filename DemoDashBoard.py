import streamlit as st
import pandas as pd
import datetime
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide")

# Load or create your dataset
df = pd.read_csv('merged_data.csv')

# Streamlit elements
st.title("Keyword Trends Dashboard")

# Sidebar
st.sidebar.header("Sidebar Controls")

year = st.sidebar.selectbox("Choose year", ("2018", "2019", "2020", "2021", "2022", "2023"))

# Function to extract the year from the publication date
def get_year(date):
    l = date.split('/')
    return l[-1]

# Apply the function to create a new column 'publication_date_year'
df['publication_date_year'] = df['publication_date'].apply(lambda x: get_year(x))

# Filter data by selected year
df_year = df.loc[df['publication_date_year'] == year]

# Split the subject area and get the first part (before the comma)
df_year['subjectArea_first'] = df_year['subjectArea'].apply(lambda x: x.split(',')[0])

# Get the value counts for the first subject area
x = df_year['subjectArea_first'].value_counts().reset_index()

# Streamlit layout with columns
col1, col2 = st.columns(2)

with col1:
    st.header('Pie chart')
    fig = px.pie(x, 
                 values='count', 
                 names='subjectArea_first', 
                 title="Subject Area Distribution")
    st.plotly_chart(fig, use_container_width=True)

# Display the subject area counts
st.write(x)


