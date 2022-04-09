from itertools import chain
import networkx as nx
import plotly.graph_objects as go
from network.arrow import addEdge

def drawNetworkGraph(G, graph_pos, addresses = [], df = None):
    isSubGraph = addresses and len(addresses)
    if isSubGraph:
        bfs_edges = iter(())
        for address in addresses:
            bfs_edges = chain(bfs_edges, nx.bfs_edges(G, address, reverse=True, depth_limit=1))
        graph = nx.Graph(G.edge_subgraph(bfs_edges))
        for address in addresses:
            graph.add_node(address)
    else:
        graph = G
    edge_x = []
    edge_y = []
    for edge in graph.edges():
        start = graph_pos[edge[0]]
        end = graph_pos[edge[1]]
        edge_x, edge_y = addEdge(start, end, edge_x, edge_y, 1, 'end', 0.0025, 30, 2)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in graph.nodes():
        x, y = graph_pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=False,  # set true to enable color scale of node
            # colorscale options
            #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='YlGnBu',
            cmax = 10,
            cmin = 1,
            reversescale=True,
            color=[1,2,3,4,5,6,7,8,9,10],
            size=6,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=0))

    highlight_node_x = []
    highlight_node_y = []

    if isSubGraph:
        for node in addresses:
            x, y = graph_pos[node]
            highlight_node_x.append(x)
            highlight_node_y.append(y)

    highlight_node_trace = go.Scatter(
        x=highlight_node_x, y=highlight_node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=False,  # set true to enable color scale of node
            color='red',
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
          ),
          line_width=0))

    node_adjacencies = []
    node_text = []
    node_list = list(graph.nodes())
    for node, adjacencies in enumerate(graph.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append(node_list[node] + '# of connections: '+str(len(adjacencies[1])))

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text
    fig = go.Figure(
        data=[edge_trace, node_trace, highlight_node_trace],
        layout=go.Layout(
        titlefont_size=16,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        annotations=[ 
            dict(
                showarrow=True,
                xref="paper", yref="paper",
                x=0.005, y=-0.002 ) ],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
        )

    for node in graph.nodes():
        img_df = df[df['owner_address'] == node]['owner_img_md']
        if not img_df.empty:
            x, y = graph_pos[node]
            src = img_df.iloc[0]
            src = src[len("<img src='"):]
            src = src[:-len("'/>")]
            fig.add_layout_image(
                dict(
                    source=src,
                    xref="x",
                    yref="y",
                    xanchor="center",
                    yanchor="middle",
                    x=x,
                    y=y,
                    sizex=0.01,
                    sizey=0.01,
                    sizing="contain",
                    opacity=1,
                    layer="above"
                )
            )

    return fig
