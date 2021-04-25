import numpy as np

from matplotlib.widgets import LassoSelector
from matplotlib.path import Path


class SelectFromCollection:
    """
    Select indices from a matplotlib collection using `LassoSelector`.

    Selected indices are saved in the `ind` attribute. This tool fades out the
    points that are not part of the selection (i.e., reduces their alpha
    values). If your collection has alpha < 1, this tool will permanently
    alter the alpha values.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        Axes to interact with.
    collection : `matplotlib.collections.Collection` subclass
        Collection you want to select from.
    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to *alpha_other*.
    """

    def __init__(self, ax, collection, alpha_other=0.3):
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.alpha_other = alpha_other

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

        self.lasso = LassoSelector(ax, onselect=self.onselect)
        self.ind = []

    def onselect(self, verts):
        path = Path(verts)
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        self.fc[:, -1] = self.alpha_other
        self.fc[self.ind, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

    def disconnect(self):
        self.lasso.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

from visualize_events import save_play
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from NETS.data_utils.get_data import get_classifier_data
from matplotlib.widgets  import RectangleSelector

X_tsne = np.load('./data/tsne.npy')
X_train, y_train, X_val, y_val, X_test, y_test, n_classes, encoder =\
    get_classifier_data('./data/middle_all.pkl')
X = np.vstack([X_train, X_val, X_test])
y = np.concatenate([y_train, y_val, y_test])


fig, ax = plt.subplots(figsize=(8,5))

def accept(event):
    if event.key == "enter":
        plt.close()
        plays = X[selector.ind]
        for idx, play in enumerate(plays):
            if idx > 25:
                return
            save_play(play, f'./animations/lasso/{idx}')

fig.canvas.mpl_connect("key_press_event", accept)

pts = ax.scatter(
    X_tsne[:,0], X_tsne[:,1],
    s=1, c=y
)
ax.clear()

for label in np.unique(y):
    ax.scatter(
        X_tsne[y == label, 0], X_tsne[y == label, 1],
        s=1, label=encoder.inverse_transform([label])[0]
    )

selector = SelectFromCollection(ax, pts)

ax.legend(bbox_to_anchor=(1, 0.5), fancybox=True, loc='center left')
plt.tight_layout()
plt.show()

for label in np.unique(y):
    plt.cla()
    plt.scatter(X_tsne[y != label, 0], X_tsne[y != label, 1], s=1, c='gray')
    plt.scatter(X_tsne[y == label, 0], X_tsne[y == label, 1], s=1, c='blue')
    title = encoder.inverse_transform([label])[0]
    plt.title(title)
    plt.savefig(f'{title}.png')