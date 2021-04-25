import pickle as pkl
import numpy as np

from tqdm import tqdm
import matplotlib.pyplot as plt

(X_test, team, X_emb_flat, X_tsne) = pkl.load( open('saved_data/Clippers_Lakers_tsne.pkl', 'rb'))

idx_clippers = np.argwhere(team==0) # clippers
idx_lakers = np.argwhere(team==1) # lakers

plt.figure(1)
plt.clf()
plt.scatter(X_tsne[idx_clippers, 0], X_tsne[idx_clippers, 1], c='b', s=2, label='Clippers')
plt.scatter(X_tsne[idx_lakers, 0], X_tsne[idx_lakers, 1], c='goldenrod', s=2, label='Lakers')
plt.legend()

# %%
from sklearn.cluster import AgglomerativeClustering

cluster10 = AgglomerativeClustering(n_clusters=12, affinity='euclidean', linkage='ward')
preds = cluster10.fit_predict(X_tsne)

plt.figure(2, figsize=(10, 7))
plt.clf()
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=cluster10.labels_, cmap='tab20')


# %%
from sklearn.svm import SVC
from scipy.spatial import ConvexHull
    
dend_model = AgglomerativeClustering(distance_threshold=0, n_clusters=None, affinity='euclidean', linkage='ward')
dend_model = dend_model.fit(X_tsne)

counts = np.zeros(dend_model.children_.shape[0])
idx_list_of_list = []
n_samples = len(dend_model.labels_)
for i, merge in enumerate(dend_model.children_):
    current_count = 0
    idx_list = []
    for child_idx in merge:
        if child_idx < n_samples:
            current_count += 1  # leaf node
            idx_list.append(child_idx)
        else:
            current_count += counts[child_idx - n_samples]
            idx_list = idx_list + idx_list_of_list[child_idx - n_samples]
    counts[i] = current_count
    idx_list_of_list.append(idx_list)

linkage_matrix = np.column_stack([dend_model.children_, dend_model.distances_,
                                  counts]).astype(float)

score_list = []
for idx_list in tqdm(idx_list_of_list):
    X_cl = X_tsne[idx_list]
    y_cl = team[idx_list]
    
    if y_cl.shape[0] > 3 and np.unique(y_cl).shape[0] > 1:
        clf = SVC(kernel='linear')
        clf.fit(X_cl, y_cl)
        score_list.append( clf.score(X_cl, y_cl) )
    else:
        score_list.append(1.)

def stretchHull(poly, stretchCoef):
    center = poly.mean(0)
    stretched_poly = poly - center
    stretched_poly = stretched_poly*stretchCoef
    stretched_poly = stretched_poly + center
    return stretched_poly

plt.figure(3, figsize=(10,7))
plt.clf()

count = 0
plot_list = []
for i, score in enumerate(score_list):
    if score > 0.8 and counts[i] > 20:
        idxs = idx_list_of_list[i]
        points = X_tsne[idxs]
        
        # https://stackoverflow.com/questions/62376042/calculating-and-displaying-a-convexhull
        hull = ConvexHull(points)
        polygon = stretchHull(points[hull.vertices], stretchCoef=1.05)
        polygon = np.append(polygon, [polygon[0]], axis=0)        
        count += 1
        
        plot_list.append((count, i, polygon))

# remove regions within other regions
clean_list = []
for element in reversed(plot_list):
    e_idx = idx_list_of_list[ element[1] ][0]
    is_good = True
    for c in clean_list:
        parent_idx = idx_list_of_list[ c[1] ]
        if e_idx in parent_idx:
            is_good = False
    if is_good:
        clean_list.append(element)

for element in clean_list:
    (count, i, polygon) = element
    plt.gca().plot(polygon[:, 0], polygon[:, 1], 'k', alpha=0.9, lw=2.)
    print(f'ROI {count}: score = {score_list[i]:.3f} n = {counts[i]}')

plt.scatter(X_tsne[idx_clippers, 0], X_tsne[idx_clippers, 1], c='b', label='Clippers')
plt.scatter(X_tsne[idx_lakers, 0], X_tsne[idx_lakers, 1], c='goldenrod', label='Lakers')
plt.legend(prop={'size': 14})




