import streamlit as st
import pandas as pd
import datetime
from PIL import Image
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# Sample DataFrame with keywords
data = {
    "Field of Study": [
        "MEDI", "ENGI", "CHEM", "BUSI", "BIOC", "DECI", "MATE", "COMP",
        "PHYS", "ENVI", "AGRI", "ENER", "SOCI", "VETE", "NEUR", "ECON",
        "EART", "MATH", "MULT", "IMMU", "PHAR", "DENT", "CENG", "NURS",
        "HEAL", "PSYC", "ARTS"
    ],
    "Keyword": [
        "Medicine", "Engineering", "Chemistry", "Business", "Biochemistry",
        "Decision Science", "Materials", "Computer Science", "Physics",
        "Environment", "Agriculture", "Energy", "Sociology", "Veterinary",
        "Neuroscience", "Economics", "Earth Science", "Mathematics",
        "Multidisciplinary", "Immunology", "Pharmacology", "Dentistry",
        "Chemical Engineering", "Nursing", "Health", "Psychology", "Arts"
    ]
}

keyword_to_field = {
    "Medicine": "MEDI",
    "Engineering": "ENGI",
    "Chemistry": "CHEM",
    "Business": "BUSI",
    "Biochemistry": "BIOC",
    "Decision Science": "DECI",
    "Materials": "MATE",
    "Computer Science": "COMP",
    "Physics": "PHYS",
    "Environment": "ENVI",
    "Agriculture": "AGRI",
    "Energy": "ENER",
    "Sociology": "SOCI",
    "Veterinary": "VETE",
    "Neuroscience": "NEUR",
    "Economics": "ECON",
    "Earth Science": "EART",
    "Mathematics": "MATH",
    "Multidisciplinary": "MULT",
    "Immunology": "IMMU",
    "Pharmacology": "PHAR",
    "Dentistry": "DENT",
    "Chemical Engineering": "CENG",
    "Nursing": "NURS",
    "Health": "HEAL",
    "Psychology": "PSYC",
    "Arts": "ARTS"
}

df = pd.read_csv("merged_data_withkeywords.csv")
top_keyword_df = pd.read_csv("top_keywords_by_field.csv")
# Set page configuration
st.set_page_config(page_title="Analytics Dashboard", layout="wide")

# Sidebar content
st.sidebar.title("Analytics")
analytic_option = st.sidebar.selectbox(
    "Select an analytic type",
    ["Keyword Trends", "Statistical Data", "Subject Area"]
)

# Main content based on selection
st.title("Analytics Dashboard")

if analytic_option == "Keyword Trends":
    # Year selection
    selected_year = st.sidebar.selectbox(
        "Select Year",
        ["2018", "2019", "2020", "2021", "2022", "2023"]
    )

    # Field of Study selection
    selected_field = st.sidebar.selectbox(
        "Field of Study",
        [
            "Medicine", "Engineering", "Chemistry", "Business", "Biochemistry",
        "Decision Science", "Materials", "Computer Science", "Physics",
        "Environment", "Agriculture", "Energy", "Sociology", "Veterinary",
        "Neuroscience", "Economics", "Earth Science", "Mathematics",
        "Multidisciplinary", "Immunology", "Pharmacology", "Dentistry",
        "Chemical Engineering", "Nursing", "Health", "Psychology", "Arts"
        ]
    )



    # Filter data based on selected field
    filtered_data = top_keyword_df[top_keyword_df == keyword_to_field[selected_field]]
    result = filtered_data["top_keyword"]

    # Display the table
    st.subheader(f"Keyword Trends for {selected_field} in {selected_year}")
    st.table(result)

elif analytic_option == "Statistical Data":
    st.subheader("Statistical Data")
    st.write("This section will display statistical data.")

elif analytic_option == "Subject Area":
    st.subheader("Subject Area")
    st.write("This section will display information about subject areas.")
    # Bar chart count of subject areas

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

