import numpy as np

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
