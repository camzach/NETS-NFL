import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import numpy as np
import glob
import os

OFFENSE_COLOR = '#ff7f00'
DEFENSE_COLOR = '#377eb8'
BALL_COLOR = '#a65628'
TRAIL_COLOR = '#984ea3'

def save_play(past, future, predicted, location):
  past = past.reshape((-1,18,2))
  length = past.shape[0] + future.shape[0]

  fig, ax = plt.subplots()

  def animate(i):
    ax.clear()
    ax.imshow(plt.imread('./field.png'), extent=[0,120,0,53.3])

    # Plot trails in past data
    ax.plot(past[:i,:,0], past[:i,:,1], linewidth=2, color=TRAIL_COLOR, alpha=0.5)

    if i < past.shape[0]:
      data = past[i]

      # Plot past data 
      ax.plot(data[0,0], data[0,1], marker='o', color=BALL_COLOR)
      ax.scatter(data[1:7,0], data[1:7,1], marker='o', color=OFFENSE_COLOR)
      ax.scatter(data[7:,0], data[7:,1], marker='o', color=DEFENSE_COLOR)
    else:
      i -= past.shape[0]
      _future = future[i]
      _predicted = predicted[i]

      # Plot trails for future data
      ax.plot(future[:i,0,0], future[:i,0,1], linewidth=2, color=BALL_COLOR, alpha=0.5)
      ax.plot(future[:i,1:7,0], future[:i,1:7,1], linewidth=2, color=OFFENSE_COLOR, alpha=0.5)
      ax.plot(future[:i,7:,0], future[:i,7:,1], linewidth=2, color=DEFENSE_COLOR, alpha=0.5)

      # Plot trails for predicted data
      ax.plot(predicted[:i,:,0], predicted[:i,:,1], linewidth=2, color=TRAIL_COLOR, alpha=0.5)

      # Plot future data
      ax.plot(_future[0,0], _future[0,1], marker='o', color=BALL_COLOR)
      ax.scatter(_future[1:7,0], _future[1:7,1], marker='o', color=OFFENSE_COLOR)
      ax.scatter(_future[7:,0], _future[7:,1], marker='o', color=DEFENSE_COLOR)

      # Plot predicted data
      ax.plot(_predicted[0,0], _predicted[0,1], marker='x', color='black')
      ax.scatter(_predicted[1:7,0], _predicted[1:7,1], marker='x', color='black')
      ax.scatter(_predicted[7:,0], _predicted[7:,1], marker='x', color='black')

  anim = FuncAnimation(fig, animate, frames=length, interval=100)
  anim.save(f'{location}.mp4')
  plt.close(fig)

data = pd.read_pickle('NETS/saved_data/NFLsequences_with_predictions.pkl')
for index, play in data.head(25).iterrows():
  past = play['past']
  future = play['future']
  predicted = play['predictions']
  location = f'./animations/predictions/1_week/{index}'
  print(location)
  save_play(past, future, predicted, location)

data = pd.read_pickle('NETS/saved_data/NFLsequences_with_predictions_2.pkl')
for index, play in data.head(25).iterrows():
  past = play['past']
  future = play['future']
  predicted = play['predictions']
  location = f'./animations/predictions/17_weeks/{index}'
  print(location)
  save_play(past, future, predicted, location)
