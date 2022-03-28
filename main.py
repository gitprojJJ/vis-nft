# %%
import dash
from dash import dcc, html, Dash, dash_table
import dash_bootstrap_components as dbc
from jupyter_dash import JupyterDash
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

# %%
import plotly.graph_objects as go
import plotly.express as px
import plotly

# %%
import pandas as pd
import numpy as np
import networkx as nx
import sys
import os
import copy

# %%
#g_path = "/content/drive/MyDrive/COMP7507_Group28/nft_crawler/cryptopunk"
g_path = "../data"
net_df = pd.read_csv(os.path.join(g_path,"network.csv"))

# %%
data_df = pd.read_csv(os.path.join(g_path,"out.csv"))
data_df['last_sale_total_price_unscaled'] = data_df['last_sale_total_price'].astype(float).copy()
data_df['last_sale_total_price'] = data_df['last_sale_total_price'].astype(float)/1e18
data_df['last_sale_total_price'] = data_df['last_sale_total_price'].fillna(0)
data_df['traits_list'] = data_df['traits_list'].fillna('NoTrait')
data_df['img_md'] = data_df['image_url'].apply(lambda url : f"<img src='{url}' height='30' />")


# %%
def set_buckets(df, num_buckets=4):
    # set price ranges, dividing price ranges into equal number of assets.
    # Ignoring those without sales
    # Whole population of the collection
    prices = df[df['num_sales']>0]['last_sale_total_price'].tolist()
    q = [i / num_buckets for i in range(num_buckets + 1)]
    return np.nanquantile(prices, q)

# %%
def assign_price_buckets(data_df, prices_quantile):
    for i ,upperlimitrange in enumerate(prices_quantile):
        if i == 0 :
            data_df['no_sales'] = (data_df["num_sales"] == 0).copy()
        elif i == (len(prices_quantile)-1): 
            data_df['$' * i] =  (~data_df["no_sales"] & 
                               (data_df["last_sale_total_price"] <= prices_quantile[i]) & 
                               (data_df["last_sale_total_price"] >= prices_quantile[i-1]))
        
        else:
            data_df['$' * i] =  (~data_df["no_sales"] & 
                               (data_df["last_sale_total_price"] < prices_quantile[i]) & 
                               (data_df["last_sale_total_price"] >= prices_quantile[i-1]))

# %%
prices_quantile = set_buckets(data_df, num_buckets=4)
assign_price_buckets(data_df, prices_quantile)

# %%
data_df['traits_list_aslist'] = data_df['traits_list'].astype(str).apply(lambda tl_str : tl_str.split(','))
traits_set = set(data_df['traits_list_aslist'].sum())
owners_set = set(data_df['owner_address'].tolist())
owners_id_dict = {address : idx for idx, address in enumerate(list(owners_set))}
data_df['owner_id'] = data_df['owner_address'].apply(lambda address : owners_id_dict[address])

# %%
traits_df_dict_list = []
for trait in list(traits_set):
    dict_entry = {'trait' : trait}
    df_temp = data_df[data_df['traits_list_aslist'].apply(lambda tl : trait in tl)].copy()
    dict_entry['num_sales'] = df_temp['num_sales'].sum()
    dict_entry['no_sales'] = df_temp['no_sales'].sum()
    for i in range(1,len(prices_quantile)):
        dict_entry['$'*i] = df_temp['$'*i].sum()
    dict_entry['token_id'] = df_temp['id'].tolist() 
    dict_entry['n_token'] = len(df_temp['id'].tolist())
    dict_entry['owner_id'] = list(set(df_temp['owner_id'].tolist()))
    dict_entry['n_owner'] = len(dict_entry['owner_id'])
    traits_df_dict_list.append(dict_entry)
traits_df = pd.DataFrame.from_records(traits_df_dict_list)
traits_df = traits_df.sort_values("n_token", ascending = False)

traits_list = traits_df['trait'].tolist()

# %% [markdown]
# # Gary_attribute_values_commented.

# %%
def import_data(path):
    # Remove unnecessary columns, but keeping them should be fine 
    return pd.read_csv(path, usecols=['last_sale_total_price', 'name', 'traits_list', 'traits_sex'])


def process_data(df):
    df = df.astype({'traits_list': str, 'last_sale_total_price': float})

    # uncomment if sex is also considered as trait
    # df['traits_list'] = df['traits_list'] + ',' + df['traits_sex']

    # change price in unit ETH price
    df['last_sale_total_price'] = df['last_sale_total_price'] / 10**18

    df.drop(labels='traits_sex', axis='columns', inplace=True)

    # traits_list column has a np.array datatype
    df['traits_list'] = df['traits_list'].apply(lambda x: np.array(x.split(',')))

    return df


def filter_trait(df, trait):
    # return a copy of rows having that trait
    if trait in ['all', 'All', 'ALL', '']:
        return df
    else:
        condition = df["traits_list_aslist"].apply(lambda tl: trait in tl)
        return df[condition]


def set_buckets(df, num_buckets=4):
    # set price ranges, dividing price ranges into equal number of assets.
    # Ignoring those without sales
    # Whole population of the collection
    prices = df['last_sale_total_price']
    q = [i / num_buckets for i in range(num_buckets + 1)]
    return np.nanquantile(prices, q)


def find_all_traits(df):
    # return a set of all traits
    all_traits = set()
    for traits in df['traits_list']:
        for trait in traits:
            all_traits.add(trait)
    return all_traits


def find_trait_proportions(df_trait, buckets, proportion=False):
    # proportion: as a portion of all (between 0 and 1)
    asset_count = len(df_trait)
    no_sale_count = df_trait['no_sales'].sum()
    df_with_sales = df_trait[['$','$$','$$$','$$$$']].sum().sum()
    freqs = np.array(df_trait[['$','$$','$$$','$$$$']].sum())
    if proportion:
        return [
            asset_count,
            no_sale_count / asset_count,
            [freq / asset_count for freq in freqs]
        ]
    else:
        return [asset_count, no_sale_count, freqs]


def traits_stats(df, trait_names, num_buckets=4):
    # trait_names: list of trait names
    PREC = 4
    buckets = set_buckets(df, num_buckets)
    all_asset_count = len(df)
    stats = dict()
    for trait_name in trait_names:
        df_trait = filter_trait(df, trait_name)
        count_t, no_sale_t, freq_t = find_trait_proportions(df_trait, buckets, True)
        stats[trait_name] = {
            'rarity': np.round(count_t / all_asset_count, PREC),  # The smaller, the rarer
            'sale_prop': np.round(1 - no_sale_t, PREC),  # proportion of asset (with trait) that have sales
            'freqs': [np.round(f, PREC) for f in freq_t]  # Proportions of $, $$, $$$, $$$$
        }
    return stats


def price_range_graph(df, traits, num_buckets=4, interested_traits = []):
    # main function for generating a figure (go.Figure)
    price_tags = ['No Sales'] + ['&#36;' * i for i in range(1, num_buckets + 1)]
    # &#36; is dollar signs, plain '$' will crash everything (thinking latex)
    # still thinking a way to incorporate rarity in the graph
    
    testing_stats = traits_stats(df, traits, num_buckets)
    data = []
    for n in traits:
        visible = 'legendonly'
        if n in interested_traits:
            visible = True
        go_bar = go.Bar(
                    name=n,
                    x=price_tags,
                    y=[1 - testing_stats[n]['sale_prop']]+testing_stats[n]['freqs'],
                    visible = visible,
                    customdata=[n]*len(price_tags),
                    hovertemplate="<br>".join([
                        "Price: %{x}",
                        "Proportion: %{y}",
                        "Trait: %{customdata}",
                    ]))
        data.append(go_bar)
    fig = go.Figure(data=data)

    fig.update_layout(barmode='group')
    return fig

# %%
#attribute_fig = price_range_graph(data_df, traits_list, num_buckets=4, interested_traits = traits_list[0:2])
#attribute_fig.show()

# %%
app = JupyterDash()
app.title = 'Attribute Values'

all_traits = list(find_all_traits(df))

# Change this to change the initial attributes
default_trait_list = ['Crazy Hair', 'Hoodie', 'Spots', 'Hot Lipstick']

app.layout = html.Div([
    dcc.Dropdown(all_traits, default_trait_list, multi=True, id='all_traits_dropdown'),
    dcc.Graph(id='attribute_value_graph')
])

@app.callback(
    Output('attribute_value_graph', 'figure'),
    [Input('all_traits_dropdown', 'value')]
)
def update_attribute_value_fig(value):
    if not value:
        raise PreventUpdate

    new_traits = ['All'] + value
    return price_range_graph(df,  new_traits, num_buckets=4)

# %%
#if __name__ == '__main__':
#    app.run_server(debug=True, mode='inline')

# %% [markdown]
# # Morris

# %%
# Select top 50 sales count in the bar chart
def make_price_bar_fig(df_dc):
    bar_data = df_dc.sort_values('num_sales', ascending = False).head(50)
    bar_data.reset_index()
    
    bar_color = np.full(len(bar_data), 'blue')
    price_bar_fig = px.bar(bar_data, y='last_sale_total_price', x = 'name', color_discrete_sequence=[bar_color])
    return price_bar_fig, bar_color
price_bar_fig = make_price_bar_fig(data_df)

# Group some small amount address
def make_ownership_tree(df_dc):
    tree_data = df_dc.groupby('owner_address').agg({'last_sale_total_price':'sum', 'id':'count'})
    tree_data = tree_data.rename(columns={'last_sale_total_price':'total_value', 'id':'item_count'})
    tree_data['owner_address'] = tree_data.index.values
    tree_data['owner_address_group'] = tree_data.index.values
    tree_data.loc[tree_data['total_value']<5000, 'owner_address_group'] = "Other owner (500-5000 ETH)"
    tree_data.loc[tree_data['total_value']<500, 'owner_address_group'] = "Other owner (50-500 ETH)"
    tree_data.loc[tree_data['total_value']<50, 'owner_address_group'] = "Other owner (5-50 ETH)"
    tree_data.loc[tree_data['total_value']<5, 'owner_address_group'] = "Other owner (<5 ETH)"
    figTreemap = px.treemap(tree_data, path=['owner_address_group', 'owner_address'], values='total_value')
    return figTreemap, tree_data
figTreemap, tree_data = make_ownership_tree(data_df)


# %%
# create a dash app
app = JupyterDash()

# dash layout
app.layout = html.Div(children=[
    
    dcc.Graph(
        id='figTreemap',
        figure=figTreemap
    ),
    dcc.Graph(
        id='price_bar_fig',
        figure=price_bar_fig
    ),
    
])

@app.callback(
    Output('price_bar_fig', 'figure'),
    Input('figTreemap', 'hoverData')
)

def linkPieChartToBarChart(hoverData): 
    #print(hoverData)
    # make a copy of the bar chart and color
    updateBar = copy.deepcopy(price_bar_fig)
    updateColor = copy.deepcopy(bar_color)
    
    bar_data_owner_address = bar_data['owner_address']

    if hoverData is not None and 'label' in hoverData['points'][0]: 
        
        hover_label = hoverData['points'][0]['label']
        hover_owner_address = tree_data.loc[(tree_data.owner_address_group == hover_label)]['owner_address']

        
        highlight_address = pd.Series(list(set(bar_data_owner_address) & set(hover_owner_address)))
        
        updateColor[bar_data['owner_address'].isin(hover_owner_address)] = 'red'
        updateColor[(bar_data.owner_address == hover_label)]='red'
        
    updateBar.update_traces(marker_color=updateColor)
    
    return updateBar

# %%
#app.run_server(mode='inline')

# %% [markdown]
# # William Network

# %%
tx = net_df.copy()
tx = tx[tx.Method == 'Buy Punk']
tx = tx.drop(['Txhash', 'Blockno', 'UnixTimestamp', 'DateTime', 'Method'], axis=1)
tx = tx.groupby(['From', 'To'])['Quantity'].sum().reset_index(name='value')
tx.rename(columns={'From':'source','To':'target','Quantity':'weight'}, inplace=True)

G = nx.from_pandas_edgelist(tx,edge_attr=True)
G.add_nodes_from(set(data_df.owner_address.tolist()))
graph_pos = nx.spring_layout(G)




# %%
import math
from typing import List
from itertools import chain

# Start and end are lists defining start and end points
# Edge x and y are lists used to construct the graph
# arrowAngle and arrowLength define properties of the arrowhead
# arrowPos is None, 'middle' or 'end' based on where on the edge you want the arrow to appear
# arrowLength is the length of the arrowhead
# arrowAngle is the angle in degrees that the arrowhead makes with the edge
# dotSize is the plotly scatter dot size you are using (used to even out line spacing when you have a mix of edge lengths)
def addEdge(start, end, edge_x, edge_y, lengthFrac=1, arrowPos = None, arrowLength=0.025, arrowAngle = 30, dotSize=20):

    # Get start and end cartesian coordinates
    x0, y0 = start
    x1, y1 = end

    # Incorporate the fraction of this segment covered by a dot into total reduction
    length = math.sqrt( (x1-x0)**2 + (y1-y0)**2 )
    dotSizeConversion = .0565/20 # length units per dot size
    convertedDotDiameter = dotSize * dotSizeConversion
    lengthFracReduction = convertedDotDiameter / length
    lengthFrac = lengthFrac - lengthFracReduction

    # If the line segment should not cover the entire distance, get actual start and end coords
    skipX = (x1-x0)*(1-lengthFrac)
    skipY = (y1-y0)*(1-lengthFrac)
    x0 = x0 + skipX/2
    x1 = x1 - skipX/2
    y0 = y0 + skipY/2
    y1 = y1 - skipY/2

    # Append line corresponding to the edge
    edge_x.append(x0)
    edge_x.append(x1)
    edge_x.append(None) # Prevents a line being drawn from end of this edge to start of next edge
    edge_y.append(y0)
    edge_y.append(y1)
    edge_y.append(None)

    # Draw arrow
    if not arrowPos == None:

        # Find the point of the arrow; assume is at end unless told middle
        pointx = x1
        pointy = y1

        eta = math.degrees(math.atan((x1-x0)/(y1-y0))) if y1!=y0 else 90.0

        if arrowPos == 'middle' or arrowPos == 'mid':
            pointx = x0 + (x1-x0)/2
            pointy = y0 + (y1-y0)/2

        # Find the directions the arrows are pointing
        signx = (x1-x0)/abs(x1-x0) if x1!=x0 else +1    #verify this once
        signy = (y1-y0)/abs(y1-y0) if y1!=y0 else +1    #verified

        # Append first arrowhead
        dx = arrowLength * math.sin(math.radians(eta + arrowAngle))
        dy = arrowLength * math.cos(math.radians(eta + arrowAngle))
        edge_x.append(pointx)
        edge_x.append(pointx - signx**2 * signy * dx)
        edge_x.append(None)
        edge_y.append(pointy)
        edge_y.append(pointy - signx**2 * signy * dy)
        edge_y.append(None)

        # And second arrowhead
        dx = arrowLength * math.sin(math.radians(eta - arrowAngle))
        dy = arrowLength * math.cos(math.radians(eta - arrowAngle))
        edge_x.append(pointx)
        edge_x.append(pointx - signx**2 * signy * dx)
        edge_x.append(None)
        edge_y.append(pointy)
        edge_y.append(pointy - signx**2 * signy * dy)
        edge_y.append(None)


    return edge_x, edge_y

def add_arrows(source_x: List[float], target_x: List[float], source_y: List[float], target_y: List[float],
               arrowLength=0.025, arrowAngle=30):
    pointx = list(map(lambda x: x[0] + (x[1] - x[0]) / 2, zip(source_x, target_x)))
    pointy = list(map(lambda x: x[0] + (x[1] - x[0]) / 2, zip(source_y, target_y)))
    etas = list(map(lambda x: math.degrees(math.atan((x[1] - x[0]) / (x[3] - x[2]))),
                    zip(source_x, target_x, source_y, target_y)))

    signx = list(map(lambda x: (x[1] - x[0]) / abs(x[1] - x[0]), zip(source_x, target_x)))
    signy = list(map(lambda x: (x[1] - x[0]) / abs(x[1] - x[0]), zip(source_y, target_y)))

    dx = list(map(lambda x: arrowLength * math.sin(math.radians(x + arrowAngle)), etas))
    dy = list(map(lambda x: arrowLength * math.cos(math.radians(x + arrowAngle)), etas))
    none_spacer = [None for _ in range(len(pointx))]
    arrow_line_x = list(map(lambda x: x[0] - x[1] ** 2 * x[2] * x[3], zip(pointx, signx, signy, dx)))
    arrow_line_y = list(map(lambda x: x[0] - x[1] ** 2 * x[2] * x[3], zip(pointy, signx, signy, dy)))

    arrow_line_1x_coords = list(chain(*zip(pointx, arrow_line_x, none_spacer)))
    arrow_line_1y_coords = list(chain(*zip(pointy, arrow_line_y, none_spacer)))

    dx = list(map(lambda x: arrowLength * math.sin(math.radians(x - arrowAngle)), etas))
    dy = list(map(lambda x: arrowLength * math.cos(math.radians(x - arrowAngle)), etas))
    none_spacer = [None for _ in range(len(pointx))]
    arrow_line_x = list(map(lambda x: x[0] - x[1] ** 2 * x[2] * x[3], zip(pointx, signx, signy, dx)))
    arrow_line_y = list(map(lambda x: x[0] - x[1] ** 2 * x[2] * x[3], zip(pointy, signx, signy, dy)))

    arrow_line_2x_coords = list(chain(*zip(pointx, arrow_line_x, none_spacer)))
    arrow_line_2y_coords = list(chain(*zip(pointy, arrow_line_y, none_spacer)))

    x_arrows = arrow_line_1x_coords + arrow_line_2x_coords
    y_arrows = arrow_line_1y_coords + arrow_line_2y_coords

    return x_arrows, y_arrows

# %%
from itertools import chain

def drawGraph(addresses = None):
    isSubGraph = addresses and len(addresses)
    if isSubGraph: 
        bfs_edges = iter(())
        for address in addresses:
            bfs_edges = chain(bfs_edges, nx.bfs_edges(G, address, reverse=True, depth_limit=1))
        graph = G.edge_subgraph(bfs_edges)
    else:
        graph = G
    edge_x = []
    edge_y = []
    for edge in graph.edges():
        # x0, y0 = graph_pos[edge[0]]
        # x1, y1 = graph_pos[edge[1]]
        # edge_x.append(x0)
        # edge_x.append(x1)
        # edge_x.append(None)
        # edge_y.append(y0)
        # edge_y.append(y1)
        # edge_y.append(None)
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
            showscale=True,
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
    return fig

# %%
#graph_fig = drawGraph()

# %%
#graph_fig = drawGraph(['0x914f4d27761edba6926fb02f984f430f183220ed','0xf2ef5636b38ecf46324ac97957a514beda674d7d'])

# %%
#graph_fig.show()

# %% [markdown]
# # With Tab Table

# %%
table_df = data_df[['img_md','id','num_sales',
                    'last_sale_total_price','name','owner_address',
                    'traits_sex','traits_count','traits_list']].copy()
table_df['traits_list'] = data_df['traits_list_aslist'].apply(lambda tl : (" , ").join(tl))

# %%
import time
from dash import Dash, dcc, html

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = JupyterDash(__name__, external_stylesheets=external_stylesheets)

dash_table_cols = []
for col in table_df.columns:
    col_dict = {"name": col, "id": col, "deletable": False, "selectable": False, }
    if col in ['img_md','traits_list_list']:
        col_dict['presentation'] = "markdown"
    dash_table_cols.append(col_dict)
    
token_df_filtered = pd.DataFrame()
price_min = data_df['last_sale_total_price'].min()
price_max = data_df['last_sale_total_price'].max()
submit_n_click = 0
figure_built = False

app.layout = html.Div([
    dcc.Tabs([
        dcc.Tab(label='Table', children=[
            dash_table.DataTable(
                id='datatable-interactivity',
                columns=dash_table_cols,
                data=table_df.to_dict('records'),
                editable=False,
                filter_action="native",
                filter_options = {'case' : 'insensitive'},
                sort_action="native",
                sort_mode="multi",
                column_selectable="single",
                row_selectable="multi",
                row_deletable=False,
                selected_columns=[],
                selected_rows=[],
                page_action="native",
                page_current= 0,
                page_size= 20,
                markdown_options={"html": True},  # dangerous but works
            ),
            html.Div(id='datatable-interactivity-container'),
            html.Div([
                f"Price Range (min: {price_min:.2f}, max : {price_max:.2f}) :      ",
                dcc.Input(
                    id="price_min",
                    value = 0,
                    type="number",
                    placeholder="Minimum Price",
                ),
                dcc.Input(
                    id="price_max",
                    value = 3000,
                    type="number",
                    placeholder="Maximum Price",
                ),
                html.Br(),
                dcc.Checklist(
                    id="had_sales_check",
                    options=[{"label": "Only select those had sales", "value": "had_sales"}],
                    value=['had_sales'],
                    inline = False,
                ),
                html.Br(),
                'Select Traits :',
                dcc.Checklist(
                    id="all-or-none",
                    options=[{"label": "Select All", "value": "All"}],
                    #value=[],
                    inline = False,
                ),
                dcc.Checklist(
                    options = list(traits_list),
                    value = ['Earring'],
                    id = 'traits_checklist',
                    inline = True,
                ),
                html.Br(),
                'Sort by Column : ',
                dcc.Dropdown(
                    ['num_sales','last_sale_total_price','traits_count'],
                    ['last_sale_total_price'],
                    id="dropdown_sortby",
                    multi = True,
                    searchable = True,
                    clearable = True,
                    placeholder="sort data by column names in order of" ,
                ),
                html.Br(),
                'Sort Order : Ascending or Descending : ',
                dcc.RadioItems(
                    id = 'sortby_order',
                    options = ['Ascending', 'Descending'], 
                    value = 'Descending'
                ),
                html.Br(),
                'Maximum number to NFTs  :    ',
                dcc.Slider(
                    min=1, 
                    max=30, step=1, value=28, 
                    id='max_n_nfts'
                ),
                html.Br(),
                dbc.Button(
                    'Submit',
                    id = 'submit_button',
                    color="primary",
                    disabled=False,
                    n_clicks = 0,
                ),
                #html.Br,
                
            ]),
            html.Div(id='submit_msg'),
        ]),
        dcc.Tab(label='Charts', children=[
            ###Col
            html.Div([
                html.Div([
                    #dash.html.Label("Material 1"),
                    dash.dcc.Graph(id='price_bar_fig', figure=None)
                ],className="six columns"),
                html.Div([
                    #dash.html.Label("Material 2"),
                    dash.dcc.Graph(id='tree_map_fig', figure= None), #figTreemap),
                ],style= {'height': 100},className="six columns"),
            ], className="row"),
            html.Div([
                html.Div([
                    #dash.html.Label("Material 1"),
                    dash.dcc.Graph(id = 'attribute_fig', figure = None),
                ],className="six columns"),
                html.Div([
                    #dash.html.Label("Material 2"),
                    dash.dcc.Graph(id = 'network_fig', figure = None),
                ],className="six columns"),
            ],style= {'height': 100}, className="row"),
            html.Div([
                dcc.Tooltip(id="picture_tooltip"),
            ]),
        ]),
    ])
])


#@app.callback(
#    [Output("submit_msg", "children"),
#     Output("submit_button", "disable"),
#    ],
#    [Input("submit_button", "n_clicks")],
#)
#def loading_df_fig(n_clicks):
#    return "Loading...", False
@app.callback(
    [
        Output("picture_tooltip", "show"),
        Output("picture_tooltip", "bbox"),
        Output("picture_tooltip", "children"),
        Output("price_bar_fig", "clear_on_unhover"),
        Output("tree_map_fig", "clear_on_unhover"),
        Output('attribute_fig', 'clear_on_unhover'),
        Output('network_fig', 'clear_on_unhover'),
    ],
    [
        Input('price_bar_fig', 'hoverData'),
    ],
)
def show_pic(hoverData_price):
    if hoverData_price is None:
        return False, dash.no_update, dash.no_update, True, True, True, True
    pt = hoverData_price["points"][0]
    bbox = pt["bbox"]
    nft_name = pt["x"]
    
    df_row = data_df[data_df['name'] == nft_name].iloc[0]
    img_src = df_row['image_url']
    traits = df_row['traits_list_aslist']
    price = df_row['last_sale_total_price']
    num_sales = df_row['num_sales']

    children = [
        html.Div([
            html.Img(src=img_src, style={"width": "100%"}),
            html.H2(f"{nft_name}", style={"color": "darkblue"}),
            html.P(f"{', '.join(traits)}"),
            html.P(f"price : {price:.4g}"),
            html.P(f"number of sales : {num_sales}"),
        ], style={'width': '200px', 'white-space': 'normal'})
    ]
    #print(pt,bbox,num)
    return True, bbox, children, True, True, True, True

@app.callback(
    [
        Output("submit_msg", "children"),
        Output("price_bar_fig", "figure"),
        Output("tree_map_fig", "figure"),
        Output('attribute_fig', 'figure'),
        Output('network_fig', 'figure'),
     #Output("submit_button", "disable"),
    ],
    [
        Input("submit_button", "n_clicks"),
        Input('tree_map_fig', 'hoverData'),
        Input('attribute_fig', 'hoverData'),
    ],
    [
        State("had_sales_check", "value"),
        State("price_min", "value"),
        State("price_max", "value"),
        State("traits_checklist", "value"),
        State("dropdown_sortby", "value"),
        State("sortby_order", "value"),
        State("max_n_nfts", "value"),
        
        State("price_bar_fig", "figure"),
        State("tree_map_fig", "figure"),
        State('attribute_fig', 'figure'),
        State('network_fig', 'figure'),
    ],
)
def make_hover_figures(
    n_clicks,
    hoverData_tree, hoverData_attr,
    had_sales, price_min, price_max, selected_traits_list, sortby_list, sort_order, max_n_nfts,
    price_bar_fig, tree_map_fig, attribute_fig, network_fig):
    
    #print(str(hoverData_attr),str(hoverData_tree))
    global figure_built
    global submit_n_click
    
    if ((n_clicks != submit_n_click) | (not figure_built)  ):
      
        submit_n_click = n_clicks 
       
        
        price_min = max([price_min,0])
        
        price_cond = ((data_df['last_sale_total_price'] > price_min) &
                      (data_df['last_sale_total_price'] <= price_max))

        if len(had_sales) > 0:
            had_sales_cond = data_df['num_sales'] > 0
        else :
            had_sales_cond = data_df['num_sales'] >= 0
            
         

        triats_cond = data_df['traits_list_aslist'].apply(lambda tl: len(set(tl) - (set(tl)-set(selected_traits_list))) > 0 )

        temp_data_df = data_df[price_cond & had_sales_cond & triats_cond].copy()
        temp_data_df = temp_data_df.sort_values(sortby_list, ascending = (sort_order == 'Ascending')).head(max_n_nfts)

        global token_df_filtered
        global owner_df_filtered
        global price_colorbar

        token_df_filtered = temp_data_df
        price_bar_fig, price_colorbar = make_price_bar_fig(token_df_filtered)
        
        tree_map_fig, owner_df_filtered = make_ownership_tree(token_df_filtered)
        
        owner_list = owner_df_filtered['owner_address'].tolist()
        network_fig = drawGraph(owner_list)
        all_traits_list = token_df_filtered['traits_list_aslist'].sum()
        if all_traits_list == 0:
            time.sleep(0.25)
            all_traits_list = token_df_filtered['traits_list_aslist'].sum()
        traits_list = list(set(all_traits_list))
        attribute_fig = price_range_graph(data_df, traits_list, num_buckets=4, interested_traits = selected_traits_list)
        
        figure_built = True

        return "Load Complete !", price_bar_fig, tree_map_fig, attribute_fig, network_fig  #, True
    
    elif (hoverData_tree is not None) & (hoverData_attr is None):
        #print(hoverData_tree)
        updateBar = linkTreeChartToBarChart(
            hoverData_tree,
            price_colorbar,
            price_bar_fig,
            token_df_filtered)
        return dash.no_update, updateBar, dash.no_update, dash.no_update, dash.no_update
    
    elif (hoverData_attr is not None) & (hoverData_tree is None):
        #print(hoverData_attr)
        updateBar = linkAttrChartToBarChart(
            hoverData_attr, 
            price_colorbar, 
            price_bar_fig, 
            token_df_filtered)
        return dash.no_update, updateBar, dash.no_update, dash.no_update, dash.no_update
    
    else:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
   
    
    
@app.callback(
    Output("traits_checklist", "value"),
    [Input("all-or-none", "value")],
    [State("traits_checklist", "options")],)
def select_all_none(all_selected, options):
    all_or_none = []
    all_or_none = [option for option in options if all_selected]
    return all_or_none

def linkAttrChartToBarChart(hoverData, bar_color, price_bar_fig, bar_data): 
    
    #print(hoverData)
    # make a copy of the bar chart and color
    updateBar = copy.deepcopy(price_bar_fig)
    updateBar = go.Figure(updateBar)
    updateColor = copy.deepcopy(bar_color)

    if hoverData is not None and 'customdata' in hoverData['points'][0]: 
        
        hover_label = hoverData['points'][0]['customdata']
        #print(hover_label)
        tokens_contain_trait = bar_data['traits_list_aslist'].apply(lambda tr : hover_label in tr).tolist()
        #print(tokens_contain_trait )
        updateColor = ['green' if contain_trait else updateColor[i] for i,contain_trait in enumerate(tokens_contain_trait) ]
        #print(updateColor)
    updateBar.update_traces(marker_color=updateColor)
    
    return updateBar

def linkTreeChartToBarChart(hoverData, bar_color, price_bar_fig, bar_data): 
    
    #print(hoverData)
    # make a copy of the bar chart and color
    updateBar = copy.deepcopy(price_bar_fig)
    updateBar = go.Figure(updateBar)
    updateColor = copy.deepcopy(bar_color)
    
    bar_data_owner_address = token_df_filtered['owner_address']

    if hoverData is not None and 'label' in hoverData['points'][0]: 
        
        hover_label = hoverData['points'][0]['label']
        hover_owner_address = owner_df_filtered.loc[(owner_df_filtered.owner_address_group == hover_label)]['owner_address']

        
        highlight_address = pd.Series(list(set(bar_data_owner_address) & set(hover_owner_address)), dtype = 'object')
        
        updateColor[bar_data['owner_address'].isin(hover_owner_address)] = 'red'
        updateColor[(bar_data.owner_address == hover_label)]='red'
        
    updateBar.update_traces(marker_color=updateColor)
    
    return updateBar


if __name__ == '__main__':
    app.run_server(mode = 'external')

# %%


