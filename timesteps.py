import pandas as pd
import random
import numpy as np
import os
import math

PAST_FRAMES = 5
FUTURE_FRAMES = 20
TOTAL_FRAMES = PAST_FRAMES + FUTURE_FRAMES

plays = pd.read_pickle('./data/processed.pkl')
out_df = pd.DataFrame()
for index, play in plays.iterrows():
  row = {}
  row['old_idx'] = index
  data = play['play'].reshape(-1,18,2)
  extra_frames = data.shape[0] % TOTAL_FRAMES
  if extra_frames > 0:
    extra_data = data[-extra_frames:].copy()
    data = data[:-extra_frames]
  try:
    data = np.split(data, data.shape[0] / TOTAL_FRAMES)
    for q in data:
      row['past'] = q[:PAST_FRAMES]
      row['future'] = q[-FUTURE_FRAMES:]
      out_df = out_df.append(row, ignore_index=True)
  except:
    pass
  # try:
  #   pad_len = TOTAL_FRAMES - extra_data.shape[0]
  #   extra_data = np.pad(extra_data, ((0, pad_len),(0,0),(0,0)), mode='constant', constant_values=np.nan)
  #   row['past'] = extra_data[:PAST_FRAMES]
  #   row['future'] = extra_data[-FUTURE_FRAMES:]
  #   out_df = out_df.append(row, ignore_index=True)
  # except:
  #   if extra_frames != 0:
  #     print('something broke')
  #     exit()
  #   pass

out_df.to_pickle('./data/sequences.pkl')
