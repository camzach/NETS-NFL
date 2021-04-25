import numpy as np
from itertools import groupby

def argsort(seq):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)

def ball_handler(df):
    dt = 0.04 * 3
    ball_handler_list = []
    closest_teammate_list = []

    for t, data in df.iterrows():
        if t > 0:
            vx_ball = (df.iloc[t].x_ball - df.iloc[t-1].x_ball)/dt
            vy_ball = (df.iloc[t].y_ball - df.iloc[t-1].y_ball)/dt
            v_ball = np.sqrt(vx_ball**2 + vy_ball**2)
            if v_ball > 25 or df.iloc[t].z_ball > 10:
                ball_handler_list.append(-1)
                closest_teammate_list.append(-1)
                continue
        
        # find offensive
        dist_o = []
        for p in range(5):
            # offense
            dx = df.iloc[t].x_ball - df.iloc[t]['x_off_%i' % p]
            dy = df.iloc[t].y_ball - df.iloc[t]['y_off_%i' % p]
            dist_o.append(np.sqrt(dx ** 2 + dy ** 2))

        # Calling my function to find second closest offender to the ball
        # from the dist_o list, storing it in answer, then grabbing the index
        idx_sorted = argsort(dist_o)
        ball_handler = idx_sorted[0]
        smallest_dist = dist_o[ball_handler]
        if smallest_dist > 5:
            ball_handler = -1
            
        ball_handler_list.append(ball_handler)
        closest_teammate_list.append(idx_sorted[1])

    # make sure ball handler has possession for at least 5 frames
    groups = groupby(ball_handler_list)
    tuple_list = [(label, sum(1 for _ in group)) for label, group in groups]
    for i, (key, reps) in enumerate(tuple_list):
        if reps < 5:
            tuple_list[i] = (-1, reps)
    ball_handler_list = [key for key, reps in tuple_list for i in range(reps)]
        
    df['ball_handler'] = ball_handler_list
    df['closest_teammate'] = closest_teammate_list


def is_in_the_paint(x, y):
    if y > 33 or y < 17 or x > 15:
        return False
    return True

def detect_handoff(df):

    ball_handler(df)

    bool_list = [False] * len(df.index)

    counter = 0
    min_possession_frames = 5
    is_time_step_hand_off = False

    for t, data in df.iterrows():
        currentBallHandler = df.ball_handler.values

        if t == 0 or currentBallHandler[t] < 0 or currentBallHandler[t-1] < 0:
            continue

        if currentBallHandler [t - 1] != currentBallHandler[t]:

            is_time_step_hand_off = False

            if counter < min_possession_frames:
                counter = 0
            else:
                counter = 0

                if is_in_the_paint(data['x_ball'], data['y_ball']):
                    continue

                p1 = currentBallHandler[t-1]
                p2 = currentBallHandler[t]

                # Calculate distances of the two player closest to the ball
                dx = df.iloc[t]['x_off_%i' % p1] - df.iloc[t]['x_off_%i' % p2]
                dy = df.iloc[t]['y_off_%i' % p1] - df.iloc[t]['y_off_%i' % p2]
                distance = np.sqrt(dx ** 2 + dy ** 2)

                if distance < 6.5:
                    is_time_step_hand_off = True

        else:
            counter += 1

            if is_time_step_hand_off and counter > min_possession_frames:
                bool_list[t-min_possession_frames] = True
                is_time_step_hand_off = False

    df['is_handoff'] = bool_list
