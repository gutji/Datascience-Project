import streamlit as st
import networkx as nx
import pandas as pd
import plotly.graph_objects as go

# Streamlit app title
st.title("Interactive Network Graph: Subject Areas and Keywords")

# File uploader for CSV
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    # Read the CSV file into a DataFrame
    df = pd.read_csv(uploaded_file)
    
    # Ensure the columns match expectations
    if "Subject Area" in df.columns and "Emerging Keywords" in df.columns:
        # Process the data
        data = [
            (row["Subject Area"], row["Emerging Keywords"].split(","))
            for _, row in df.iterrows()
        ]
        
        # Initialize the graph
        G = nx.Graph()

        # Add nodes and edges
        for subjects, keywords in data:
            subject_nodes = subjects.split(",")
            for subject in subject_nodes:
                G.add_node(subject, type='subject')
            for keyword in keywords:
                G.add_node(keyword, type='keyword')
                for subject in subject_nodes:
                    G.add_edge(subject, keyword)

        # Create positions for nodes
        pos = nx.spring_layout(G, seed=42)

        # Function to plot the interactive network
        def plot_interactive_network(graph, positions):
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
                mode='lines')

            node_x = []
            node_y = []
            node_color = []
            node_text = []
            for node in graph.nodes():
                x, y = positions[node]
                node_x.append(x)
                node_y.append(y)
                node_color.append('skyblue' if graph.nodes[node]['type'] == 'subject' else 'orange')
                node_text.append(node)

            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                text=node_text,
                marker=dict(
                    size=10,
                    color=node_color,
                    line_width=2))

            fig = go.Figure(data=[edge_trace, node_trace],
                            layout=go.Layout(
                                showlegend=False,
                                hovermode='closest',
                                margin=dict(b=0, l=0, r=0, t=40),
                                xaxis=dict(showgrid=False, zeroline=False),
                                yaxis=dict(showgrid=False, zeroline=False)))

            return fig

        # Display the graph
        fig = plot_interactive_network(G, pos)
        st.plotly_chart(fig)
    else:
        st.error("The CSV file must contain 'Subject Area' and 'Emerging Keywords' columns.")
else:
    st.info("Please upload a CSV file to visualize the network.")
