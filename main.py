import time
from jupyter_dash import JupyterDash
import dash
from dash import html
from dash.dependencies import Input, Output, State

from data_reader import load_network_data, load_nft_data
from layout.draw import makeLayout
from network.draw import drawNetworkGraph
from prices.draw import linkTreeChartToStripChart, linkAttrChartToStripChart
from ownership.draw import make_ownership_tree
from prices.draw import make_price_strip_fig
from traits.draw import price_range_graph

#ABC20220402

data_df, table_df, traits_list = load_nft_data()
network_df, G, network_graph_pos = load_network_data(data_df)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

current_df_filtered = data_df
temp_df_filtered = current_df_filtered
temp_id_filter = None
temp_owner_filter = None
submit_n_click = 0
reset_n_click = 0
figure_built = False
point_color = None
last_clickData_price = None
last_clickData_tree = None
last_selectedData_price = None
last_clickData_attr = None
active_traits = None

app = JupyterDash(__name__, external_stylesheets=external_stylesheets)

app.layout = makeLayout(data_df, table_df, traits_list)
@app.callback(
    [
        Output("picture_tooltip", "show"),
        Output("picture_tooltip", "bbox"),
        Output("picture_tooltip", "children"),
        Output("price_strip_fig", "clear_on_unhover"),
        Output("tree_map_fig", "clear_on_unhover"),
        Output('attribute_fig', 'clear_on_unhover'),
        Output('network_fig', 'clear_on_unhover'),
    ],
    [
        Input('price_strip_fig', 'hoverData'),
    ],
)
def show_pic(hoverData_price):
    if hoverData_price is None:
        return False, dash.no_update, dash.no_update, True, True, True, True
    pt = hoverData_price["points"][0]
    bbox = pt["bbox"]
    nft_name = pt["customdata"][0]

    df_row = data_df[data_df['name'] == nft_name].iloc[0]
    img_src = df_row['image_url']
    traits = df_row['traits_list_aslist']
    price = df_row['last_sale_total_price']
    num_sales = df_row['num_sales']

    children = [
        html.Div([
            html.Img(src=img_src, style={"width": "100%"}),
            html.P(f"{nft_name}", style={"color": "darkblue",'fontSize': 11}),
            html.P(f"{', '.join(traits)}", style = {'fontSize': 8}),
            html.P(f"price : {price:.4g}", style = {'fontSize': 8}),
            html.P(f"number of sales : {num_sales}", style = {'fontSize': 8}),
        ], style={'width': '100px', 'white-space': 'normal'})
    ]
    #print(pt,bbox,num)
    return True, bbox, children, True, True, True, True


def updateFigureFromDf(token_df_filtered, active_traits):
    price_strip_fig, point_color = make_price_strip_fig(token_df_filtered)

    tree_map_fig, owner_df_filtered = make_ownership_tree(token_df_filtered)

    owner_list = owner_df_filtered['owner_address'].tolist()
    network_fig = drawNetworkGraph(G, network_graph_pos, owner_list)
    all_traits_list = token_df_filtered['traits_list_aslist'].sum()
    if all_traits_list == 0:
        time.sleep(0.25)
        all_traits_list = token_df_filtered['traits_list_aslist'].sum()
    traits_list = list(set(all_traits_list))
    attribute_fig = price_range_graph(data_df, traits_list, num_buckets=4, interested_traits=active_traits)
    return price_strip_fig, point_color, tree_map_fig, network_fig, attribute_fig, owner_df_filtered


def update_from_table(had_sales_check, price_min, price_max, active_traits, sortby_list, sort_order, max_n_nfts):
    price_min = max([price_min,0])

    price_cond = ((data_df['last_sale_total_price'] > price_min) &
                    (data_df['last_sale_total_price'] <= price_max))

    if len(had_sales_check) > 0:
        had_sales_cond = data_df['num_sales'] > 0
    else :
        had_sales_cond = data_df['num_sales'] >= 0

    traits_cond = data_df['traits_list_aslist'].apply(lambda tl: len(set(tl) - (set(tl)-set(active_traits))) > 0)

    temp_data_df = data_df[price_cond & had_sales_cond & traits_cond].copy()
    temp_data_df = temp_data_df.sort_values(sortby_list, ascending = (sort_order == 'Ascending')).head(max_n_nfts)
    return temp_data_df


@app.callback(
    [
        Output("submit_msg", "children"),
        Output("price_strip_fig", "figure"),
        Output("tree_map_fig", "figure"),
        Output('attribute_fig', 'figure'),
        Output('network_fig', 'figure'),
     #Output("submit_button", "disable"),
    ],
    [
        Input("submit_button", "n_clicks"),
        Input("reset_button", "n_clicks"),
        Input('tree_map_fig', 'hoverData'),
        Input('attribute_fig', 'hoverData'),
        Input('tree_map_fig', 'clickData'),
        Input('attribute_fig', 'clickData'),
        Input('price_strip_fig', 'clickData'),
        Input('price_strip_fig', 'selectedData'),
    ],
    [
        State("had_sales_check", "value"),
        State("price_min", "value"),
        State("price_max", "value"),
        State("traits_checklist", "value"),
        State("dropdown_sortby", "value"),
        State("sortby_order", "value"),
        State("max_n_nfts", "value"),

        State("price_strip_fig", "figure"),
        State("tree_map_fig", "figure"),
        State('attribute_fig', 'figure'),
        State('network_fig', 'figure'),
    ],
)
def make_hover_figures(
    n_clicks, reset_clicks,
    hoverData_tree, hoverData_attr, clickData_tree, clickData_attr, clickData_price, selectedData_price,
    had_sales_check, price_min, price_max, selected_traits_list, sortby_list, sort_order, max_n_nfts,
    price_strip_fig, tree_map_fig, attribute_fig, network_fig):

    #print(str(hoverData_attr),str(hoverData_tree))
    global figure_built
    global submit_n_click
    global reset_n_click
    global current_df_filtered
    global current_owner_df_filtered
    global temp_df_filtered
    global point_color
    global last_clickData_price
    global last_clickData_tree
    global last_clickData_attr
    global last_selectedData_price
    global active_traits

    isClickEvent = (last_clickData_tree != clickData_tree) or (last_clickData_price != clickData_price) or (last_clickData_attr != clickData_attr)
    isSelectEvent = (last_selectedData_price != selectedData_price)
    isHoverEvent = hoverData_tree is not None or hoverData_attr is not None
    if (reset_clicks > reset_n_click):
        active_traits = [trait for trait in selected_traits_list]
        reset_n_click = reset_clicks
        temp_df_filtered = current_df_filtered
        price_strip_fig, point_color, tree_map_fig, network_fig, attribute_fig, owner_df_filtered = updateFigureFromDf(current_df_filtered, active_traits)
        current_owner_df_filtered = owner_df_filtered
        return dash.no_update, price_strip_fig, tree_map_fig, attribute_fig, network_fig
    elif ((n_clicks > submit_n_click) or (not figure_built)  ):
        active_traits = [trait for trait in selected_traits_list]
        submit_n_click = n_clicks
        current_df_filtered = update_from_table(had_sales_check, price_min, price_max, selected_traits_list, sortby_list, sort_order, max_n_nfts)
        temp_df_filtered = current_df_filtered
        price_strip_fig, point_color, tree_map_fig, network_fig, attribute_fig, owner_df_filtered = updateFigureFromDf(current_df_filtered, active_traits)
        current_owner_df_filtered = owner_df_filtered
        figure_built = True
        return "Load Complete !", price_strip_fig, tree_map_fig, attribute_fig, network_fig  #, True
    elif isClickEvent:
        shouldUpdate = False
        if clickData_price is not None and last_clickData_price != clickData_price:
            last_clickData_price = clickData_price
            token_id = clickData_price['points'][0]['customdata'][0]
            temp_df_filtered = current_df_filtered[current_df_filtered.name == token_id]
            active_traits = temp_df_filtered['traits_list_aslist'].sum()
            price_strip_fig, point_color, tree_map_fig, network_fig, attribute_fig, owner_df_filtered = updateFigureFromDf(temp_df_filtered, active_traits)
            shouldUpdate = True
        elif clickData_tree is not None and last_clickData_tree != clickData_tree:
            last_clickData_tree = clickData_tree
            owner_id = clickData_tree['points'][0]['label']
            temp_df_filtered = current_df_filtered[current_df_filtered.owner_address == owner_id]
            active_traits = temp_df_filtered['traits_list_aslist'].sum()
            price_strip_fig, point_color, tree_map_fig, network_fig, attribute_fig, owner_df_filtered = updateFigureFromDf(temp_df_filtered, active_traits)
            shouldUpdate = True
        elif clickData_attr is not None and last_clickData_attr != clickData_attr:
            last_clickData_attr = clickData_attr
            clicked_trait = clickData_attr['points'][0]['customdata'][0]
            active_traits = [clicked_trait]
            condition = current_df_filtered["traits_list_aslist"].apply(lambda tl: clicked_trait in tl)
            temp_df_filtered = current_df_filtered[condition]
            price_strip_fig, point_color, tree_map_fig, network_fig, attribute_fig, owner_df_filtered = updateFigureFromDf(temp_df_filtered, active_traits)
            shouldUpdate = True
        if (shouldUpdate):
            return dash.no_update, price_strip_fig, tree_map_fig, attribute_fig, network_fig  #, True
    elif isSelectEvent:
        shouldUpdate = False
        if selectedData_price is not None and last_selectedData_price != selectedData_price:
            last_selectedData_price = selectedData_price
            token_ids = [x['customdata'][0] for x in selectedData_price['points']]
            temp_df_filtered = current_df_filtered[current_df_filtered['name'].isin(token_ids)]
            price_strip_fig, point_color, tree_map_fig, network_fig, attribute_fig, owner_df_filtered = updateFigureFromDf(temp_df_filtered, selected_traits_list)
            shouldUpdate = True
        if (shouldUpdate):
            return dash.no_update, price_strip_fig, tree_map_fig, attribute_fig, network_fig  #, True
    elif isHoverEvent:
        if (hoverData_tree is not None) & (hoverData_attr is None):
            #print(hoverData_tree)
            updateStrip = linkTreeChartToStripChart(
                hoverData_tree,
                point_color,
                price_strip_fig,
                temp_df_filtered)
            return dash.no_update, updateStrip, dash.no_update, dash.no_update, dash.no_update
        elif (hoverData_attr is not None) & (hoverData_tree is None):
            #print(hoverData_attr)
            updateStrip = linkAttrChartToStripChart(
                hoverData_attr,
                point_color,
                price_strip_fig,
                temp_df_filtered)
            return dash.no_update, updateStrip, dash.no_update, dash.no_update, dash.no_update
    return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

@app.callback(
    Output("traits_checklist", "value"),
    [Input("all-or-none", "value")],
    [State("traits_checklist", "options")],)
def select_all_none(all_selected, options):
    all_or_none = []
    all_or_none = [option for option in options if all_selected]
    return all_or_none

if __name__ == '__main__':
    app.run_server(debug = True, mode = 'external')