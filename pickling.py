import pickle
import os

from data_reader import load_network_data, load_nft_data

data_df, table_df, traits_list = load_nft_data()
network_df, G, network_graph_pos = load_network_data(data_df)

outpath = "data"

pickle.dump(data_df, open(os.path.join(outpath, "data_df.p"), "wb"))
pickle.dump(table_df, open(os.path.join(outpath, "table_df.p"), "wb"))
pickle.dump(traits_list, open(os.path.join(outpath, "traits_list.p"), "wb"))
pickle.dump(network_df, open(os.path.join(outpath, "network_df.p"), "wb"))
pickle.dump(G, open(os.path.join(outpath, "G.p"), "wb"))
pickle.dump(network_graph_pos, open(os.path.join(outpath, "network_graph_pos.p"), "wb"))