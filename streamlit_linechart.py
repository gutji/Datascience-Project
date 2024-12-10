import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go


# Load your dataset
df = pd.read_csv('merged_data_withkeywords.csv')  # Adjust to your file path

# Convert publication_date to datetime format
df['publication_date'] = pd.to_datetime(df['publication_date'], format='%d/%m/%Y')

# Split subjectArea into multiple rows
subject_df = df.assign(subjectArea=df['subjectArea'].str.split(',')).explode('subjectArea')

# Expand One_keyword into individual rows
keywords_df = subject_df.assign(One_keyword=subject_df['One_keyword'].str.split(', ')).explode('One_keyword')

# Extract year for aggregation
keywords_df['year'] = keywords_df['publication_date'].dt.year

# Add an option for all subject areas
subject_area_map["All"] = "All Subject Areas"

# Inverse mapping for filtering
full_name_to_abbreviation = {v: k for k, v in subject_area_map.items()}

# Streamlit sidebar for subject area selection
st.sidebar.header("Filters")
available_subject_areas_full_names = sorted(subject_area_map.values())
selected_full_name = st.sidebar.selectbox("Select Subject Area", available_subject_areas_full_names)

# Convert the selected full name back to abbreviation
selected_subject_area = full_name_to_abbreviation[selected_full_name]

# Display selected subject area in subheader
st.subheader(f"Line graph for {selected_full_name} Subject Area")

# Filter data by selected subject area (or use all data)
if selected_subject_area == "All":
    filtered_subject_data = keywords_df
else:
    filtered_subject_data = keywords_df[keywords_df['subjectArea'] == selected_subject_area]

if filtered_subject_data.empty:
    st.warning(f"No data found for subject area: {selected_full_name}")
else:
    # Count total occurrences of each keyword and filter top 10
    keyword_counts = filtered_subject_data['One_keyword'].value_counts()
    top_keywords = keyword_counts.head(10).index
    filtered_data = filtered_subject_data[filtered_subject_data['One_keyword'].isin(top_keywords)]

    # Group by year and keyword
    trend_data = filtered_data.groupby(['year', 'One_keyword']).size().unstack(fill_value=0)

    # Plot the trends
    fig, ax = plt.subplots(figsize=(12, 6))
    for keyword in trend_data.columns:
        ax.plot(trend_data.index, trend_data[keyword], label=keyword)

    ax.set_title(f"Top Keyword Trends Over Time in {selected_full_name}")
    ax.set_xlabel("Year")
    ax.set_ylabel("Frequency")
    ax.legend(title="Keywords", loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(fig)