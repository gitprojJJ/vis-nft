import plotly.graph_objects as go
import pandas as pd
import copy

def linkTreeChartToBarChart(hoverData, bar_color, price_bar_fig, token_df_filtered, owner_df_filtered):

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

        updateColor[token_df_filtered['owner_address'].isin(hover_owner_address)] = 'red'
        updateColor[(token_df_filtered.owner_address == hover_label)]='red'

    updateBar.update_traces(marker_color=updateColor)

    return updateBar

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