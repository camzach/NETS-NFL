import pandas as pd
import numpy as np
import os
import pickle as pkl
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

feature_list = [0, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
def append_snippet(df, index, start, end, out_list):
    copy = df.iloc[start:end].values[:, feature_list]
    if copy.shape[0] != (end-start):
        Exception('bad play length')
        return
    
    out_list.append({'index': index, 'play': copy})

def get_sequences(data_folder, time_length, stride):
    # time_length: number of frames to save
    # stride: how many time steps to skip
    
    file_list = os.listdir(data_folder)
    
    data_list = []    
    for file in tqdm(file_list):
        if file.endswith(".pkl"):
            load_file = data_folder + "/" + file
            game_df = pkl.load(open(load_file, "rb"))
            
            for play_index, entry in game_df.iterrows():
                possession_df = entry['possession']
                T = len(possession_df.index)
                N = T//stride
                for n in range(N):
                    start = n*stride
                    end = start + time_length
                    append_snippet(possession_df, play_index, start, end, data_list)
    
    trajectory_df = pd.DataFrame(data_list)
    return trajectory_df

def save_sequences(save_path, data_folder, time_length, stride):
    # save path: where to save plays
    # time_length: number of frames to save
    # stride: how many time steps to skip
    
    save_df = get_sequences(data_folder, time_length, stride)
    
    pkl.dump((save_df), open(save_path, 'wb'))


def extract_from_df(df, train, test, val):
    train_end = int(len(df) * train)
    test_end = int(train_end + len(df) * test)
    val_end = int(test_end + len(df) * val)
    return np.stack(df.iloc[:train_end]['play']), np.stack(df.iloc[train_end:test_end]['play']), np.stack(df.iloc[test_end:val_end]['play'])


def get_classifier_data(load_path, train=0.8, test=0.1, val=0.1):    
    df = pd.read_pickle(load_path)
    encoder = LabelEncoder()
    df['label'] = encoder.fit_transform(df['label'].to_numpy().reshape(-1,1))

    events = {event: extract_from_df(df[df.label == event], train, test, val) for event in df.label.unique()}
    n_classes = len(events.keys())
    # none_event = encoder.transform(['None'])[0]
    
    X_train, X_test, X_val = [], [], []
    y_train, y_val, y_test = [], [], []
    for event, (train, test, val) in events.items():
        # if event == none_event:
        #     continue
        X_train.append(np.nan_to_num(train, nan=-1))
        X_test.append(np.nan_to_num(test, nan=-1))
        X_val.append(np.nan_to_num(val, nan=-1))
        y_train.append([event] * train.shape[0])
        y_test.append([event] * test.shape[0])
        y_val.append([event] * val.shape[0])
    
    # train_samples = sum([len(cls) for cls in X_train])
    # test_samples = sum([len(cls) for cls in X_train])
    # val_samples = sum([len(cls) for cls in X_train])
    # train, test, val = events[none_event]
    # X_train.append(train[np.random.choice(train.shape[0], train_samples, replace=False), :])
    # X_test.append(test[np.random.choice(test.shape[0], test_samples, replace=False), :])
    # X_val.append(val[np.random.choice(val.shape[0], val_samples, replace=False), :])
    # y_train.append([none_event] * train_samples)
    # y_test.append([none_event] * test_samples)
    # y_val.append([none_event] * val_samples)

    return (
        np.vstack(X_train),
        np.concatenate(y_train),
        np.vstack(X_val),
        np.concatenate(y_val),
        np.vstack(X_test),
        np.concatenate(y_test),
        n_classes,
        encoder
    )
