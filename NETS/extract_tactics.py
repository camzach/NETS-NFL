import pandas as pd
import os
import pickle as pkl
from tqdm import tqdm

from data_utils.detect_pick_and_roll import detect_pick_and_roll
from data_utils.detect_handoff import detect_handoff


def append_snippet(possession_df, off_id, def_id, index, start, end, out_list):
    copy = possession_df.iloc[start:end].copy()
    if len(copy.index) != (end-start):
        Exception('bad indexes')
    copy.reset_index(inplace=True)
    
    out_list.append({'index': index, 'play': copy, 'off_id': off_id, 'def_id': def_id})

def generate_data(file_path, save_name):
    file_list = os.listdir(file_path)
    
    window_length = 5 # on both sides: 2 * 5 * 0.12s = 1.2s
    seq_length = 2*window_length
    minimum_time_between = 10
    
    # save pick&rolls, handoffs and others in a list
    saved_pr = []
    saved_ho = []
    saved_negatives = []
    
    # go through all files, get the prepared pickle files
    for file in tqdm(file_list):
        if file.endswith(".pkl"):
            load_file = file_path + "/" + file
            game_df = pkl.load(open(load_file, "rb"))
            
            # go through every possession
            for play_index, entry in game_df.iterrows():
                possession_df = entry['possession']
                T = len(possession_df.index)
                
                # label pick&rolls and handoffs
                detect_pick_and_roll(possession_df)
                detect_handoff(possession_df)                
                pr_list = possession_df.index[possession_df['is_pick_and_roll'] == True].tolist()
                ho_list = possession_df.index[possession_df['is_handoff'] == True].tolist()
                
                # check if there's a pick&roll or handoff, if not add it to background class
                if len(pr_list) + len(ho_list) > 0:
                    last_frame = -99
                    t_frame = 0
                    while t_frame < T-window_length:
                        # exclude events that are very close together
                        if abs(t_frame - last_frame) < minimum_time_between:
                            t_frame += 1
                            continue
                        
                        #check if whole sequence fits
                        start = t_frame+1-window_length
                        end = t_frame+1+window_length
                        if start < 0 or end > len(possession_df.index):
                            t_frame += 1
                            continue
                        
                        if t_frame in ho_list:
                            last_frame = t_frame
                            append_snippet(possession_df, entry['offense_id'], entry['defense_id'],
                                           play_index, start, end, saved_ho)
                        elif t_frame in pr_list:
                            last_frame = t_frame
                            append_snippet(possession_df, entry['offense_id'], entry['defense_id'],
                                           play_index, start, end, saved_pr)
                        t_frame += 1
                else:
                    # no plays of interest, make snippets for background data
                    for t_frame in range(0, T-seq_length, seq_length):
                        append_snippet(possession_df, entry['offense_id'], entry['defense_id'],
                                       play_index, 
                                       t_frame, t_frame+seq_length, 
                                       saved_negatives)
    
    print(f'found {len(saved_pr)} pick&rolls, {len(saved_ho)} handoffs and {len(saved_negatives)} others')
    
    pr_df = pd.DataFrame(saved_pr)
    ho_df = pd.DataFrame(saved_ho)
    negative_df = pd.DataFrame(saved_negatives)
    
    pkl.dump((pr_df, ho_df, negative_df), open(save_name, 'wb'))
    
if __name__ == '__main__':
    generate_data('saved_data/tactics_weaklabeled.pkl')
    
    