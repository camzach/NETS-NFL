import pickle as pkl
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from data_utils.get_data import get_classifier_data
import matplotlib.pyplot as plt

X_train, y_train, X_val, y_val, X_test, y_test, n_classes, encoder =\
    get_classifier_data('../data/middle_all.pkl')

X = np.vstack([X_train, X_val, X_test])
y = np.concatenate([y_train, y_val, y_test])

model = torch.load('saved_models/with_pretraining (85%).pth')

batch_size = 256
n_train = X.shape[0]
X_emb = []
for i in range(0, n_train, batch_size):
    model.eval()
    
    batch_x, batch_y = X[i:i+batch_size], y[i:i+batch_size]
    batch_x = torch. torch.from_numpy(batch_x).to(0)
    
    embedd = model.get_embedding(batch_x.float(), layer_depth=-1)
    X_emb.append(embedd.detach().cpu().numpy())
X_emb = np.concatenate( X_emb, axis=0 )
X_emb = X_emb.reshape( (n_train, -1) )

torch.cuda.empty_cache()

# PCA followed by TSNE
pca = PCA(n_components=64, random_state=1337)
X_pca = pca.fit_transform(X_emb)

tsne = TSNE(n_components=2, verbose=1, n_iter=5000, learning_rate=50.0, random_state=1337)
X_tsne = tsne.fit_transform(X_pca)

plt.figure(figsize=(8, 5))
for label in np.unique(y):
    plt.scatter(
        X_tsne[y == label,0], X_tsne[y == label,1],
        s=1, label=encoder.inverse_transform([label])[0]
    )
plt.legend(bbox_to_anchor=(1, 0.5), fancybox=True, loc='center left')
plt.tight_layout()
np.save('tsne', X_tsne)
plt.savefig('tsne.png')
