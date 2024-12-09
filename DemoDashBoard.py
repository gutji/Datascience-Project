import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide")

# Load your dataset
df = pd.read_csv('merged_data.csv')

# Streamlit elements
st.title("Keyword Trends Dashboard")

# Sidebar
st.sidebar.header("Sidebar Controls")

# Select year
year = st.sidebar.selectbox("Choose year", ("2018", "2019", "2020", "2021", "2022", "2023"))

# Function to extract the year from the publication date
def get_year(date):
    try:
        return date.split('/')[-1]
    except AttributeError:  # Handle cases where date is not a string
        return None

# Apply the function to create a new column 'publication_date_year'
df['publication_date_year'] = df['publication_date'].apply(get_year)

# Filter data by selected year
df_year = df.loc[df['publication_date_year'] == year]

# Split the subject area and get the first part (before the comma)
def get_first_subject_area(value):
    if isinstance(value, str):  # Check if the value is a string
        return value.split(',')[0]
    return "Unknown"  # Use a placeholder for missing or invalid data

df_year['subjectArea_first'] = df_year['subjectArea'].apply(get_first_subject_area)

# Get the value counts for the first subject area
x = df_year['subjectArea_first'].value_counts().reset_index()
x.columns = ['subjectArea_first', 'count']

# Add a range slider to select ranks
max_rank = len(x)
rank_range = st.sidebar.slider(
    "Select Subject Area Rank Range",
    min_value=1,
    max_value=max_rank,
    value=(1, max_rank)
)

# Filter data based on selected rank range
x_filtered = x.iloc[rank_range[0] - 1 : rank_range[1]]

# Streamlit layout with columns
#col1, col2 = st.columns(2)

#with col1:
st.header('Bar Chart')
fig = px.bar(
    x_filtered,
    x='subjectArea_first',
    y='count',
    title="Subject Area Distribution (Filtered)",
    labels={'subjectArea_first': 'Subject Area', 'count': 'Count'},
    text_auto=True  # Adds text labels on the bars
)
st.plotly_chart(fig, use_container_width=True)

# Display the filtered subject area counts
st.sidebar.write(x_filtered)
