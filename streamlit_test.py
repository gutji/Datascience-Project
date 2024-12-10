import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

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

# Read data from CSV files
df = pd.read_csv("merged_data_withkeywords.csv")
top_keyword_df = pd.read_csv("top_keywords_by_field.csv")

# Set page configuration
st.set_page_config(page_title="Analytics Dashboard🤩", layout="wide")

# Sidebar content
st.sidebar.title("Analytics")

analytic_option = st.sidebar.selectbox(
    "Select an analytic type",
    ["Keyword Trends📈", "Statistical Data", "Subject Area📚", "Top Keyword"]
)
st.markdown(
    """
    <style>
    /* Sidebar background color and text styling */
    [data-testid="stSidebar"] {
        background-color: #071630;
    }
    [data-testid="stSidebar"] * {
        color: white !important; /* Sidebar text color */
    }

    /* Make selectbox text black, including the selected option */
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
        color: black !important; /* Selected option text color */
    }

    /* Adjust dropdown menu text color when open */
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] .css-1hb7zxy-IndicatorsContainer {
        color: black !important; /* Dropdown indicator and text */
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Main content based on selection
st.title("Analytics Dashboard🤩")

# =========================================================================
# Keyword Trends Section
if analytic_option == "Keyword Trends📈":
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

    # Dictionary to map abbreviations to full names
    subject_area_map = {
        "MEDI": "Medicine",
        "ENGI": "Engineering",
        "CHEM": "Chemistry",
        "BUSI": "Business",
        "BIOC": "Biochemistry",
        "DECI": "Decision Sciences",
        "MATE": "Materials Science",
        "COMP": "Computer Science",
        "PHYS": "Physics",
        "ENVI": "Environmental Science",
        "AGRI": "Agricultural Science",
        "ENER": "Energy",
        "SOCI": "Sociology",
        "VETE": "Veterinary Science",
        "NEUR": "Neuroscience",
        "ECON": "Economics",
        "EART": "Earth Sciences",
        "MATH": "Mathematics",
        "MULT": "Multidisciplinary",
        "IMMU": "Immunology",
        "PHAR": "Pharmacology",
        "DENT": "Dentistry",
        "CENG": "Chemical Engineering",
        "NURS": "Nursing",
        "HEAL": "Health Sciences",
        "PSYC": "Psychology",
        "ARTS": "Arts and Humanities"
    }

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

    # Add a year filter for the WordCloud
    years_available = filtered_subject_data['year'].dropna().unique()
    if len(years_available) > 0:
        years_available = sorted(years_available)
        selected_year = st.sidebar.selectbox("Select Year for WordCloud", years_available)

        # Filter the data by the selected year
        year_filtered_data = filtered_subject_data[filtered_subject_data['year'] == selected_year]

        if year_filtered_data.empty:
            st.warning(f"No data found for the year {selected_year} in {selected_full_name}.")
        else:
            # Subheader for WordCloud
            st.subheader(f"Top Keyword Trends for {selected_full_name} in {selected_year}.")

            # Combine all keywords into one string for WordCloud generation
            all_keywords = ', '.join(year_filtered_data['Top_Three_Keywords'].dropna()).replace(", ", " ")

            # Generate WordCloud
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_keywords)

            # Display WordCloud in Streamlit
            fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
            ax_wc.imshow(wordcloud, interpolation='bilinear')
            ax_wc.axis('off')
            st.pyplot(fig_wc)
    else:
        st.warning("No years available for the selected subject area.")

# =========================================================================
# Statistical Data Section
elif analytic_option == "Statistical Data":
    st.subheader("Statistical Data")
    st.write("This section will display statistical data.")

# Subject Area Section
elif analytic_option == "Subject Area📚":
    # Bar chart count of subject areas
    st.title("Subject Area Distribution")

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
# =========================================================================
# Top Keyword Section
elif analytic_option == "Top Keyword":
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

    # Filter data based on selected field and year
    field_code = keyword_to_field.get(selected_field)  # Retrieve the field code based on selected field
    filtered_data = top_keyword_df[top_keyword_df["field_of_study"] == field_code]
    filtered_data = filtered_data[filtered_data["Year"] == selected_year]

    # Show the top keywords for the selected year and field
    st.subheader(f"Top Keywords for {selected_field} in {selected_year}")
    st.table(filtered_data[['Keyword', 'Top Keyword']])  # Adjust based on your column names
