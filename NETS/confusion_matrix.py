import pickle as pkl
import numpy as np
import torch
from data_utils.get_data import get_classifier_data
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

X_train, y_train, X_val, y_val, X_test, y_test, n_classes, encoder =\
    get_classifier_data('../data/middle.pkl')

model = torch.load('saved_models/with_pretraining (85%).pth')

# batching to fit into memory
batch_size = 256
n_train = X_val.shape[0]
predicted_labels = []
model.eval()
for i in range(0, n_train, batch_size):
    batch_x, batch_y = X_val[i:i+batch_size], y_val[i:i+batch_size]
    batch_x = torch. torch.from_numpy(batch_x).to(0)
    
    predictions = model(batch_x.float())
    predicted_labels.append(predictions.detach().cpu().numpy())

predicted_labels = np.concatenate(predicted_labels)
predicted_labels = np.argmax(predicted_labels, axis=1)

cmatrix = confusion_matrix(predicted_labels, y_val)

accuracy = np.trace(cmatrix) / float(np.sum(cmatrix))
misclass = 1 - accuracy

plt.figure(figsize=(8, 6))
plt.imshow(cmatrix, interpolation='nearest', cmap=plt.get_cmap('Blues'))
plt.colorbar()

def recall(idx):
    return cmatrix[idx,idx] / cmatrix[idx].sum() * 100
def percision(idx):
    return cmatrix[idx,idx] / cmatrix[:,idx].sum() * 100

xlabels = [f'{label}\n{recall(idx):.2f}%' for idx, label in enumerate(encoder.classes_)]
ylabels = [f'{label}\n{percision(idx):.2f}%' for idx, label in enumerate(encoder.classes_)]

tick_marks = np.arange(len(encoder.classes_))
plt.xticks(tick_marks, xlabels, rotation=45)
plt.yticks(tick_marks, ylabels)

thresh = cmatrix.max() / 2
for i, j in itertools.product(range(cmatrix.shape[0]), range(cmatrix.shape[1])):
    plt.text(j, i, f'{cmatrix[i,j]}',
                horizontalalignment='center',
                verticalalignment='center',
                color='white' if cmatrix[i,j] > thresh else 'black')

plt.ylabel('True label / precision')
plt.xlabel(f'Predicted label / recall\naccuracy={accuracy:0.4f}; misclass={misclass:0.4f}')
plt.tight_layout()
plt.savefig('confusion_matrix_val.png')
