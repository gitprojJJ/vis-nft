
import plotly.express as px
import numpy as np

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
