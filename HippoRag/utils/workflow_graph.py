import networkx as nx
import os
import webbrowser
import plotly.graph_objects as go

def create_workflow_graph(workflow):
    workflow_graph = nx.DiGraph()

 
    for node_name in workflow.nodes:
        
        workflow_graph.add_node(node_name, label=node_name)

    for start_node, end_node in workflow.edges:
        workflow_graph.add_edge(start_node, end_node)
    
    return workflow_graph


def visualize_workflow(graph, title="Workflow Graph"):
      pos = nx.spring_layout(graph, seed=42)
  
      edge_x, edge_y, edge_annotations = [], [], []
  
      for edge in graph.edges():
          x0, y0 = pos[edge[0]]
          x1, y1 = pos[edge[1]]
          edge_x.extend([x0, x1, None])
          edge_y.extend([y0, y1, None])
  
          edge_annotations.append(
              dict(
                  x=(x0 + x1) / 2,
                  y=(y0 + y1) / 2,
                  text="",
                  showarrow=False,
                  font=dict(size=10, color="blue"),
              )
          )
  
      node_x, node_y, node_labels = [], [], []
  
      for node in graph.nodes(data=True):
          x, y = pos[node[0]]
          node_x.append(x)
          node_y.append(y)
          node_labels.append(node[1].get("label", node[0]))
  
      edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=2, color="gray"), mode="lines")
  
      node_trace = go.Scatter(
          x=node_x,
          y=node_y,
          mode="markers+text",
          text=node_labels,
          textposition="top center",
          marker=dict(size=20, color="skyblue", line=dict(width=2, color="black")),
      )
  
      fig = go.Figure(
          data=[edge_trace, node_trace],
          layout=go.Layout(
              title=title,
              showlegend=False,
              hovermode="closest",
              xaxis=dict(showgrid=False, zeroline=False),
              yaxis=dict(showgrid=False, zeroline=False),
          ),
      )
  
      output_path = os.path.join(os.getcwd(), f"{title.replace(' ', '_').lower()}.html")
      fig.write_html(output_path)
      print(f"Workflow visualization saved at {output_path}")
      webbrowser.open("file://" + output_path)
