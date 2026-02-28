import matplotlib.pyplot as plt

def boxplot_formatted(ax, data, colors, labels, **boxplot_kwargs):
    bp = ax.boxplot(
        data, 
        widths=0.3,
        patch_artist=True,
        **boxplot_kwargs
    )
    for box, color in zip(bp['boxes'], colors):
        box.set_facecolor(color)
        
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(1.5)

    for whisker in bp['whiskers']:
        whisker.set_color('black')

    for cap in bp['caps']:
        cap.set_color('black')
        
    ax.set_xticks(np.arange(len(colors))+1)
    ax.set_xticklabels(labels)