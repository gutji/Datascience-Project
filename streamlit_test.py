import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go
from sklearn.cluster import KMeans
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('merged_data_withkeywords&vectors_dropNone.csv', low_memory=False)
field_of_study= [
        "MEDI", "ENGI", "CHEM", "BUSI", "BIOC", "DECI", "MATE", "COMP",
        "PHYS", "ENVI", "AGRI", "ENER", "SOCI", "VETE", "NEUR", "ECON",
        "EART", "MATH", "MULT", "IMMU", "PHAR", "DENT", "CENG", "NURS",
        "HEAL", "PSYC", "ARTS"
    ]


result = []

for field in field_of_study:
    field_data = df.loc[df['subjectArea'].str.contains(field)]
    field_data.reset_index(inplace=True)
    print(field)
        
    # Prepare the text data for TF-IDF
    l = []
    for i in range(300):
        l.append(f"vector{i}")

    vector = field_data[l]

    # Apply KMeans clustering (choose the number of clusters)
    num_clusters = 5  # You can modify this depending on your data
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(vector)

    # Add the cluster labels to the DataFrame for the current field of study
    field_data['cluster'] = kmeans.labels_

    # Identify the largest cluster by size
    largest_cluster = np.argmax(np.bincount(kmeans.labels_))
    #print(largest_cluster)

    # Get the indices of the largest cluster
    largest_cluster_indices = np.where(kmeans.labels_ == largest_cluster)[0]
    #print(largest_cluster_indices)

    n = len(largest_cluster_indices)
    key1 = field_data.loc[field_data.index == largest_cluster_indices[0]]['One_keyword']
    key2 = field_data.loc[field_data.index == largest_cluster_indices[int(n/2)]]['One_keyword']
    key3 = field_data.loc[field_data.index == largest_cluster_indices[-1]]['One_keyword']
    word1 = key1.iloc[0]
    word2 = key2.iloc[0]
    word3 = key3.iloc[0]

    result.append({"field_of_study": field, "top_keyword": f"{word1}, {word2}, {word3}"})

finaldf = pd.DataFrame(result)



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

# Main content based on selection

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
# =========================================================================
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

    # Mapping of abbreviations to full names
    subject_area_mapping = {
        "MEDI": "Medicine", "ENGI": "Engineering", "CHEM": "Chemistry", "BUSI": "Business", 
        "BIOC": "Biochemistry", "DECI": "Decision Sciences", "MATE": "Materials Science", 
        "COMP": "Computer Science", "PHYS": "Physics", "ENVI": "Environmental Science", 
        "AGRI": "Agricultural Science", "ENER": "Energy", "SOCI": "Sociology", "VETE": "Veterinary Science", 
        "NEUR": "Neuroscience", "ECON": "Economics", "EART": "Earth Sciences", "MATH": "Mathematics", 
        "MULT": "Multidisciplinary", "IMMU": "Immunology", "PHAR": "Pharmacology", "DENT": "Dentistry", 
        "CENG": "Chemical Engineering", "NURS": "Nursing", "HEAL": "Health Sciences", "PSYC": "Psychology", 
        "ARTS": "Arts and Humanities"
    }

    # Load the data
    df = pd.read_csv('merged_data.csv')

    # Initialize the graph
    G = nx.Graph()

    # Process the subjectArea column to create edges based on co-occurrence
    for subject_area_str in df['subjectArea']:
        # Split the subject areas in each row by comma and strip whitespace
        subject_areas = [abbr.strip() for abbr in subject_area_str.split(",")]

        # Ensure we only process abbreviations that exist in the mapping
        subject_area_full = [subject_area_mapping[abbr] for abbr in subject_areas if abbr in subject_area_mapping]
        
        # Create edges between all pairs of subject areas in each record
        for i, subject1 in enumerate(subject_area_full):
            for subject2 in subject_area_full[i+1:]:
                G.add_edge(subject1, subject2)

    # Compute betweenness centrality
    betweenness_centrality = nx.betweenness_centrality(G)

    # Create positions for nodes
    pos = nx.spring_layout(G, seed=42)

    # Streamlit app
    st.title("Co-Occurrence of Subject Areas with Betweenness Centrality")

    # Graph visualization
    def plot_interactive_network(graph, positions, betweenness_centrality):
        edge_x = []
        edge_y = []
        for edge in graph.edges():
            x0, y0 = positions[edge[0]]
            x1, y1 = positions[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )

        # Node size based on betweenness centrality
        node_x = []
        node_y = []
        node_color = []
        node_text = []
        node_size = []  # Size of nodes based on betweenness centrality

        for node in graph.nodes():
            x, y = positions[node]
            node_x.append(x)
            node_y.append(y)
            node_color.append('skyblue')
            node_text.append(node)
            # Scale betweenness centrality for node size (multiply by 1000 for visibility)
            node_size.append(betweenness_centrality[node] * 1500)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            marker=dict(
                size=node_size,
                color=node_color,
                line_width=2
            )
        )

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=0, l=0, r=0, t=40),
                            xaxis=dict(showgrid=False, zeroline=False),
                            yaxis=dict(showgrid=False, zeroline=False)
                        ))
        return fig

    # Display the graph
    fig = plot_interactive_network(G, pos, betweenness_centrality)
    st.plotly_chart(fig)

# =========================================================================
# Top Keyword Section
elif analytic_option == "Top Keyword":
    # Year selection
    datafr = pd.read_csv("extracted_kmean.csv")
    st.title("Top 3 keywords by subject area")
    st.table(datafr[['field_of_study','top_keyword']])

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
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(vector)

    # Add the PCA results to the DataFrame
    field_data['PCA1'] = reduced_vectors[:, 0]
    field_data['PCA2'] = reduced_vectors[:, 1]

    # Plot the clusters
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        x='PCA1', 
        y='PCA2', 
        hue='cluster', 
        palette='tab10', 
        data=field_data, 
        legend='full', 
        alpha=0.7
    )

    # Annotate the selected keywords
    plt.text(field_data.loc[largest_cluster_indices[0], 'PCA1'], 
            field_data.loc[largest_cluster_indices[0], 'PCA2'], 
            key1, fontsize=12, color='red')
    plt.text(field_data.loc[largest_cluster_indices[int(n/2)], 'PCA1'], 
            field_data.loc[largest_cluster_indices[int(n/2)], 'PCA2'], 
            key2, fontsize=12, color='green')
    plt.text(field_data.loc[largest_cluster_indices[-1], 'PCA1'], 
            field_data.loc[largest_cluster_indices[-1], 'PCA2'], 
            key3, fontsize=12, color='blue')

    # Customize plot
    plt.title('PCA of Clusters for ENGI Field', fontsize=16)
    plt.xlabel('PCA1', fontsize=12)
    plt.ylabel('PCA2', fontsize=12)
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.show()
    

    # Show the top keywords for the selected year and field
    # st.subheader(f"Top Keywords for {selected_field} in {selected_year}")
    # st.table(filtered_data[['Keyword', 'Top Keyword']])  # Adjust based on your column names
