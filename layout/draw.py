import dash
from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc

def makeLayout(data_df, table_df, traits_list):

    dash_table_cols = []
    for col in table_df.columns:
        col_dict = {"name": col, "id": col, "deletable": False, "selectable": False, }
        if col in ['img_md','traits_list_list', 'owner_img_md']:
            col_dict['presentation'] = "markdown"
        dash_table_cols.append(col_dict)

    price_min = data_df['last_sale_total_price'].min()
    price_max = data_df['last_sale_total_price'].max()

    return html.Div([
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
                    'Maximum number of NFTs  :    ',
                    dcc.Input(
                        type="number",
                        placeholder="Number of Data Points",
                        value = 100 , # len(data_df),
                        id='max_n_nfts',
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
                html.Button('Reset',id='reset_button', n_clicks=0),
                html.Div([
                    html.Div([
                        #dash.html.Label("Material 1"),
                        dash.dcc.Graph(id='price_strip_fig', figure=None)
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
