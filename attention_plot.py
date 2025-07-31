import matplotlib.pyplot as plt

def plot_attention_heatmap(model, tokenizer, input_seq):
    fig, ax = plt.subplots()
    ax.imshow([[0.1, 0.2], [0.3, 0.4]], cmap="viridis")
    ax.set_title("Dummy Attention Map")
    return fig