import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import numpy as np
import glob
import os

def save_play(play_data, players, location):
  length = play_data.shape[0]
  play_data = play_data.reshape(play_data.shape[0], -1, 2)
  play_data = np.transpose(play_data, (1,0,2))

  fig, ax = plt.subplots()

  def animate(i):
    ax.clear()
    ax.imshow(plt.imread('./field.png'), extent=[0,120,0,53.3])
    ax.plot(play_data[0,i,0], play_data[0,i,1], 'o', color='brown')
    for idx, player in enumerate(play_data[1:,i]):
      position = players[idx][2] if players[idx] else ''
      ax.plot(player[0], player[1], 'o', color='red' if idx < 6 else 'blue')
      ax.annotate(position, (player[0], player[1]))

  anim = FuncAnimation(fig, animate, frames=length, interval=100)
  anim.save(f'{location}.mp4')
  plt.close(fig)

if __name__ == '__main__':
  data = pd.read_pickle('./data/processed.pkl')
  for index, play in data.iterrows():
    play_data = play['play']
    quarter = play['quarter']
    gameTime = play['gameClock']
    weekname = int(play['week'])
    home = play['home']
    away = play['away']
    players = play['players']
    gamename = f'{away}@{home}'
    if not os.path.exists(f'./animations/{weekname}/{gamename}'):
      os.makedirs(f'./animations/{weekname}/{gamename}')
    if not os.path.exists(f'./animations/{weekname}/{gamename}/{quarter}'):
      os.makedirs(f'./animations/{weekname}/{gamename}/{quarter}')
    location = f'./animations/{weekname}/{gamename}/{quarter}/{gameTime}'
    print(location)
    save_play(play_data, players, location)
