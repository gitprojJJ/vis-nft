from traits.util import traits_stats
import plotly.graph_objects as go

def price_range_graph(df, traits, num_buckets=4, interested_traits = []):
    # main function for generating a figure (go.Figure)
    price_tags = ['No Sales'] + ['&#36;' * i for i in range(1, num_buckets + 1)]
    # &#36; is dollar signs, plain '$' will crash everything (thinking latex)
    # still thinking a way to incorporate rarity in the graph

    testing_stats = traits_stats(df, traits, num_buckets)
    data = []
    for n in traits:
        visible = 'legendonly'
        if n in interested_traits:
            visible = True
        go_bar = go.Bar(
                    name=n,
                    x=price_tags,
                    y=[1 - testing_stats[n]['sale_prop']]+testing_stats[n]['freqs'],
                    visible = visible,
                    customdata=[n]*len(price_tags),
                    hovertemplate="<br>".join([
                        "Price: %{x}",
                        "Proportion: %{y}",
                        "Trait: %{customdata}",
                    ]))
        data.append(go_bar)
    fig = go.Figure(data=data)

    fig.update_layout(barmode='group')
    return fig
