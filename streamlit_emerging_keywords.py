import pandas as pd
import streamlit as st

# Load and preprocess the data
data = pd.read_csv('emerging_keywords_statistical.csv')
data['Subject Area'] = data['Subject Area'].str.split(',')
data = data.explode('Subject Area')

st.title("Emerging Keywords by Subject Area")

# Dropdown for selecting subject areas
selected_subjects = st.multiselect(
    "Select subject areas:",
    options=data['Subject Area'].unique(),
    default=[]
)

if selected_subjects:
    # Filter keywords that are common across all selected subject areas
    filtered_data = data[data['Subject Area'].isin(selected_subjects)]
    grouped_keywords = filtered_data.groupby('Emerging Keywords')['Subject Area'].apply(list)

    # Find keywords present in all selected subject areas
    common_keywords = grouped_keywords[grouped_keywords.apply(lambda x: set(selected_subjects).issubset(set(x)))].index

    if len(common_keywords) > 0:
        st.write(f"**Keywords common in {', '.join(selected_subjects)}:**")
        # Display one keyword per line
        for keywords in common_keywords:
            for keyword in keywords.split(','):
                st.write(keyword.capitalize())
    else:
        st.write(f"No keywords are common across {', '.join(selected_subjects)}.")
else:
    st.write("Select subject areas to view common keywords.")
