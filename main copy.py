app.layout = 
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