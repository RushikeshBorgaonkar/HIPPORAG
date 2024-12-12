import networkx as nx
import os
import webbrowser
import plotly.graph_objects as go

def visualize_graph(graph, title="Knowledge Graph"):
    """
    Visualize the knowledge graph or subgraph using Plotly, showing relationships as edge labels.
    """
    # Generate positions for nodes using spring layout
    pos = nx.spring_layout(graph, seed=42)
    
    # Initialize lists for edges and nodes
    edge_x = []
    edge_y = []
    edge_labels = []
    edge_label_positions = []

    # Loop over the edges and extract x, y coordinates
    for edge in graph.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])  # None for separating lines
        edge_y.extend([y0, y1, None])

        label = edge[2].get("label", "")  # Get the edge label if it exists
        edge_labels.append(label)
        edge_label_positions.append(((x0 + x1) / 2, (y0 + y1) / 2))  # Midpoint for label

    # Initialize lists for nodes
    node_x = []
    node_y = []
    node_text = []

    for node, (x, y) in pos.items():
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

    # Create edge trace (lines between nodes)
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1, color="gray"),
        hoverinfo="none",
        mode="lines",
    )

    # Create node trace (markers with text)
    node_trace = go.Scatter(
    x=node_x,
    y=node_y,
    mode="markers+text",
    text=node_text,
    textposition="top center",
    hoverinfo="text",  # Show only the node's text on hover
    marker=dict(
        showscale=True,
        colorscale="YlGnBu",
        size=20,
        colorbar=dict(
            thickness=15,
            title="Node Degree",
            xanchor="left",
            titleside="right",
        ),
      ),
    )
    # Create annotations for edge labels
    annotations = [
        dict(
            x=position[0],
            y=position[1],
            text=label,
            showarrow=False,
            font=dict(size=10, color="red"),
        )
        for position, label in zip(edge_label_positions, edge_labels)
    ]

    # Create Plotly figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=title,
            showlegend=False,
            hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            annotations=annotations,
        ),
    )

    # Output path for the graph HTML
    output_path = os.path.join(os.getcwd(), f"{title.replace(' ', '_').lower()}.html")
    
    # Write the figure to an HTML file
    fig.write_html(output_path)
    print(f"Graph visualization has been saved to {output_path}")
    
    # Open the HTML file in the browser
    webbrowser.open("file://" + output_path)