import matplotlib.pyplot as pyplot

from sklearn.manifold import TSNE

import seaborn

from time import perf_counter


def visualize_tsne(data,
                   classes,
                   n_classes,
                   n_components=2,
                   figsize=(16, 10),
                   init='pca',
                   random_state=0):

    class_color_palette = seaborn.color_palette("husl", n_classes)
    print('Classes colors:')
    seaborn.palplot(class_color_palette)
    colors = [class_color_palette[c] for c in classes]
    print(len(colors))

    fig = pyplot.figure(figsize=figsize)

    tsne_start_t = perf_counter()
    tsne = TSNE(n_components=n_components, init=init, random_state=random_state)

    Y = tsne.fit_transform(data)
    tsne_end_t = perf_counter()

    print("t-SNE done in {} secs".format(tsne_end_t - tsne_start_t))

    pyplot.scatter(Y[:, 0], Y[:, 1], c=colors, cmap=pyplot.cm.Spectral)

    pyplot.show()

    return Y, fig
