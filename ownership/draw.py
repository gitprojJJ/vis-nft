
import plotly.express as px
import numpy as np

# Select top 50 sales count in the bar chart
def make_price_bar_fig(df_dc):
    bar_data = df_dc.sort_values('num_sales', ascending = False).head(50)
    bar_data.reset_index()

    bar_color = np.full(len(bar_data), 'blue')
    price_bar_fig = px.bar(bar_data, y='last_sale_total_price', x = 'name', color_discrete_sequence=[bar_color])
    return price_bar_fig, bar_color


def make_price_strip_fig(df_dc):
    
    point_data = df_dc.sort_values('num_sales', ascending = False).head(50)
    point_data.reset_index()

    # print(df_dc.columns)
    # point_data = df_dc

    point_color = ['blue']
    price_strip_fig = px.strip(point_data, y='last_sale_total_price', x='traits_count', color_discrete_sequence=point_color)

    return price_strip_fig, point_color


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
