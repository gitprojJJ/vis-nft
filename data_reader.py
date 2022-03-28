import os
import pandas as pd
import numpy as np
import networkx as nx

g_path = "./data"

def set_buckets(df, num_buckets=4):
    # set price ranges, dividing price ranges into equal number of assets.
    # Ignoring those without sales
    # Whole population of the collection
    prices = df['last_sale_total_price']
    q = [i / num_buckets for i in range(num_buckets + 1)]
    return np.nanquantile(prices, q)

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

def make_trait_list(data_df, traits_set, prices_quantile): 
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
  return traits_list


def load_nft_data():
  data_df = pd.read_csv(os.path.join(g_path,"out.csv"))
  data_df['last_sale_total_price_unscaled'] = data_df['last_sale_total_price'].astype(float).copy()
  data_df['last_sale_total_price'] = data_df['last_sale_total_price'].astype(float)/1e18
  data_df['last_sale_total_price'] = data_df['last_sale_total_price'].fillna(0)
  data_df['traits_list'] = data_df['traits_list'].fillna('NoTrait')
  data_df['img_md'] = data_df['image_url'].apply(lambda url : f"<img src='{url}' height='30' />")

  prices_quantile = set_buckets(data_df, num_buckets=4)
  assign_price_buckets(data_df, prices_quantile)

  data_df['traits_list_aslist'] = data_df['traits_list'].astype(str).apply(lambda tl_str : tl_str.split(','))
  traits_set = set(data_df['traits_list_aslist'].sum())
  owners_set = set(data_df['owner_address'].tolist())
  owners_id_dict = {address : idx for idx, address in enumerate(list(owners_set))}
  data_df['owner_id'] = data_df['owner_address'].apply(lambda address : owners_id_dict[address])
  traits_list = make_trait_list(data_df, traits_set, prices_quantile)

  table_df = data_df[['img_md','id','num_sales',
                      'last_sale_total_price','name','owner_address',
                      'traits_sex','traits_count','traits_list']].copy()
  table_df['traits_list'] = data_df['traits_list_aslist'].apply(lambda tl : (" , ").join(tl))
  return data_df, table_df, traits_list

def load_network_data(data_df):
  df = pd.read_csv(os.path.join(g_path,"network.csv"))
  tx = df.copy()
  tx = tx[tx.Method == 'Buy Punk']
  tx = tx.drop(['Txhash', 'Blockno', 'UnixTimestamp', 'DateTime', 'Method'], axis=1)
  tx = tx.groupby(['From', 'To'])['Quantity'].sum().reset_index(name='value')
  tx.rename(columns={'From':'source','To':'target','Quantity':'weight'}, inplace=True)
  network_df =tx

  G = nx.from_pandas_edgelist(tx,edge_attr=True)
  G.add_nodes_from(set(data_df.owner_address.tolist()))
  network_graph_pos = nx.spring_layout(G)
  return network_df, G, network_graph_pos
