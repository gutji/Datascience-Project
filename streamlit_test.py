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
    
    
