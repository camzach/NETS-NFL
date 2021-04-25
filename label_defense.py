import pandas as pd
import numpy as np
from visualize import save_play

df = pd.read_pickle('data/processed.pkl')

test = df.iloc[0]

df = df[(df['away'] == test.away) & (df['home'] == test.home) & (df['week'] == test.week)]

threshold = 10

for _, play in df.iterrows():
    yardline = play.yardlineNumber
    timeframe = play.play[5]
    defenders = timeframe[np.where(~np.isnan(timeframe))][14:].reshape(-1,2)
    avg_x = np.average(defenders[:,0])
    if avg_x < yardline:
        yardline = 100 - yardline
    deep_defenders = sum(defenders[:,0] - 10 - yardline > 10)
    play.players = [('','','')] * len(play.players)
    save_play(play.play, play.players, f'animations/labels/{deep_defenders if deep_defenders < 3 else "other"}/{play.gameClock}-{yardline}')
