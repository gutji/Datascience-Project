import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import streamlit as st

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
