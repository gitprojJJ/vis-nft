from turtle import fillcolor
import plotly.graph_objects as go
import pandas as pd
import copy
import numpy as np

def linkTreeChartToStripChart(hoverData, point_color, price_strip_fig, token_df_filtered, owner_df_filtered):

    updateStrip = copy.deepcopy(price_strip_fig)
    updateStrip = go.Figure(updateStrip)
    updateColor = copy.deepcopy(point_color)

    # point_data_owner_address = token_df_filtered['owner_address']

    if hoverData is not None and 'label' in hoverData['points'][0]:

        hover_label = hoverData['points'][0]['label']
        hover_owner_address = owner_df_filtered.loc[(owner_df_filtered.owner_address_group == hover_label)]['owner_address']

        # highlight_address = pd.Series(list(set(bar_data_owner_address) & set(hover_owner_address)), dtype = 'object')

        updateColor[token_df_filtered['owner_address'].isin(hover_owner_address)] = 'red'
        updateColor[(token_df_filtered.owner_address == hover_label)]='red'

    updateStrip.update_traces(marker_color=updateColor)

    return updateStrip

def linkAttrChartToStripChart(hoverData, point_color, price_strip_fig, strip_data):
    updateStrip = copy.deepcopy(price_strip_fig)
    updateStrip = go.Figure(updateStrip)
    updateColor = copy.deepcopy(point_color)
    # print(updateColor, len(updateColor), type(updateColor))

    if hoverData is not None and 'customdata' in hoverData['points'][0]:

        hover_label = hoverData['points'][0]['customdata']
        # print(hover_label)
        tokens_contain_trait = strip_data['traits_list_aslist'].apply(lambda tr : hover_label in tr).tolist()
        # print(tokens_contain_trait )
        updateColor = np.array(['green' if contain_trait else updateColor[i] for i,contain_trait in enumerate(tokens_contain_trait)])
        # print(updateColor)
    updateStrip.update_traces(marker_color=updateColor)

    return updateStrip