import pandas as pd
import random
import numpy as np
import glob
import os

games = pd.read_csv('./data/raw/games.csv')
games.set_index('gameId', inplace=True)
players = pd.read_csv('./data/raw/players.csv')
players.set_index('nflId', inplace=True)
plays_meta = pd.read_csv('./data/raw/plays.csv')
plays_meta.set_index(['gameId', 'playId'], inplace=True)
week_files = glob.iglob('./data/raw/week*.csv')

positions = players['position'].unique()
positions = np.append(positions, ['BALL'])

output_df = pd.DataFrame()
for file in week_files:
  week = pd.read_csv(file)
  week.set_index(['gameId', 'playId'], inplace=True)
  week.sort_index(inplace=True)
  print(f'{file}...')
  for gameId, playId in week.index.unique():
    play_data = week.loc[gameId, playId][['frameId', 'event', 'nflId', 'team', 'x', 'y', 'playDirection']]
    play_metadata = plays_meta.loc[gameId, playId].copy()
    game_metadata = games.loc[gameId]
    offenseTeam, defenseTeam = ('home', 'away') \
      if play_metadata['possessionTeam'] == game_metadata['homeTeamAbbr']\
      else ('away', 'home')
    play_data['team'] = play_data['team'] \
      .map({'football': 0, offenseTeam: 1, defenseTeam: 2})
    play_data = play_data.sort_values(['team', 'nflId', 'frameId'])
    offensive_players = play_data[play_data.team == 1]['nflId'].unique()
    defensive_players = play_data[play_data.team == 2]['nflId'].unique()
    play_data.set_index('nflId', inplace=True)
    if len(offensive_players) != 6 or len(defensive_players) < 6:
      continue
    first_frame = play_data[play_data['event'] == 'ball_snap'].iloc[0]['frameId']
    last_frame = play_data['frameId'].max()
    n_frames = last_frame - first_frame + 1
    def get_loc_array(playerId):
      q = play_data.loc[playerId:playerId][['x', 'y', 'frameId']].copy()
      q.set_index('frameId', inplace=True)
      try:
        q = q.reindex(range(1, last_frame + 1), fill_value=np.nan)
      except:
        q = q[~q.index.duplicated(keep='first')]
        q = q.reindex(range(1, last_frame + 1), fill_value=np.nan)
      return q.to_numpy()[first_frame - 1:]
    table = np.array([get_loc_array(np.nan)])
    for player in offensive_players:
      table = np.append(table, [get_loc_array(player)], axis=0)
    for player in defensive_players:
      table = np.append(table, [get_loc_array(player)], axis=0)
    spare = np.tile(np.nan, (11 - len(defensive_players), n_frames, 2))
    table = np.append(table, spare, axis=0)
    table = np.transpose(table, (1,0,2)).reshape((n_frames, -1))
    if play_data.iloc[0]['playDirection'] == 'left':
      table[:,::2] = np.tile(120, (n_frames, 18)) - table[:,::2]
    offenseTeam, defenseTeam = \
      (game_metadata['homeTeamAbbr'], game_metadata['visitorTeamAbbr']) \
      if offenseTeam == 'home' \
      else (game_metadata['visitorTeamAbbr'], game_metadata['homeTeamAbbr'])
    def id_to_player(id):
      player = players.loc[id]
      position = np.where(positions == player['position'], 1, 0)
      return(player['displayName'], id, position)
    football = ('Football', np.nan, np.where(positions == 'BALL', 1, 0))
    player_list = \
      [football] + \
      list(map(id_to_player, offensive_players)) + \
      list(map(id_to_player, defensive_players)) + \
      [None] * (11 - len(defensive_players))
    play_metadata['play'] = table
    play_metadata['offense'] = offenseTeam
    play_metadata['defense'] = defenseTeam
    play_metadata['players'] = player_list
    play_metadata['home'] = game_metadata['homeTeamAbbr']
    play_metadata['away'] = game_metadata['visitorTeamAbbr']
    play_metadata['week'] = game_metadata['week']
    play_metadata['events'] = play_data.groupby('frameId').max()['event'].to_numpy()[first_frame-1:]
    play_metadata.drop(columns=[['possessionTeam']])
    output_df = output_df.append(play_metadata, ignore_index=True)

output_df.to_pickle('./data/processed.pkl')
