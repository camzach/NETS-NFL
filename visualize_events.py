import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import numpy as np
import glob
import os

def save_play(play_data, location):
  length = play_data.shape[0]
  play_data = play_data.reshape(play_data.shape[0], -1, 2)
  play_data = np.transpose(play_data, (1,0,2))

  fig, ax = plt.subplots()

  def animate(i):
    ax.clear()
    ax.imshow(plt.imread('./field.png'), extent=[0,120,0,53.3])
    for idx, player in enumerate(play_data[1:,i]):
      ax.plot(player[0], player[1], 'o', color='red' if idx < 6 else 'blue')
    ax.plot(play_data[0,i,0], play_data[0,i,1], 'o', color='brown')

  anim = FuncAnimation(fig, animate, frames=length, interval=100)
  anim.save(f'{location}.mp4')
  plt.close(fig)

if __name__ == '__main__':
  data = pd.read_pickle('./data/middle.pkl')
  for label in data.label.unique():
      for idx, play in data[data.label == label].sample(5).iterrows():
          save_play(play['play'], f'./animations/{label}/{idx}')
