import pandas as pd
import numpy as np

data = pd.read_pickle('./data/processed.pkl')

# events = ['None', 'pass_forward', 'pass_outcome_caught', 'play_action', 'tackle', 'first_contact', 'pass_arrived']
events = np.unique(np.concatenate(data.events))

def get_index(events, event, padding_before=5, padding_after=5):
    indices, = np.where(events == event)
    for idx in indices:
        if idx > padding_before and idx < events.size - padding_after:
            return idx
    return None

def get_snippet(index, play, padding_before=5, padding_after=5):
    return play[int(index) - padding_before:int(index) + padding_after + 1]

middle_df = pd.DataFrame()
for event in events:
    plays_with_event = data.copy()
    plays_with_event['event_index'] = data.events.apply(get_index, args=(event,))
    plays_with_event.dropna(subset=['event_index'], inplace=True)
    if plays_with_event.size < 4000:
        print(f'Couldn\'t get {event}')
        continue
    out = pd.DataFrame()
    out['play'] = plays_with_event.apply(lambda row: get_snippet(row.event_index, row.play), axis=1).copy()
    out['label'] = event
    middle_df = middle_df.append(out)
middle_df.to_pickle('./data/middle_all.pkl')

end_df = pd.DataFrame()
for event in events:
    plays_with_event = data.copy()
    plays_with_event['event_index'] = data.events.apply(get_index, args=(event,10,0))
    plays_with_event.dropna(subset=['event_index'], inplace=True)
    if plays_with_event.size < 4000:
        print(f'Couldn\'t get {event}')
        continue
    out = pd.DataFrame()
    out['play'] = plays_with_event.apply(lambda row: get_snippet(row.event_index, row.play, 10, 0), axis=1).copy()
    out['label'] = event
    end_df = end_df.append(out)
end_df.to_pickle('./data/end_all.pkl')
