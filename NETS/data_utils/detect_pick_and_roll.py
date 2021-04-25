import numpy as np
from scipy.optimize import linear_sum_assignment
from itertools import groupby

def argsort(seq):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)

def is_in_the_paint(x, y):
    if y > 33 or y < 17 or x > 15:
        return False
    return True


def find_assignments(data):
    x_off = []
    y_off = []
    for j in range(5):
        x_off.append(data['x_off_%i' % j])
        y_off.append(data['y_off_%i' % j])
    x_def = []
    y_def = []
    for j in range(5):
        x_def.append(data['x_def_%i' % j])
        y_def.append(data['y_def_%i' % j])

    cost = []
    for i in range(5):
        sublist = []
        for k in range(5):
            dx = x_def[i] - x_off[k]
            dy = y_def[i] - y_off[k]
            dist_square = dx * dx + dy * dy
            sublist.append(dist_square)
        cost.append(sublist)
    _, assignments = linear_sum_assignment(cost)

    return assignments.tolist(), np.sqrt(cost).tolist()


def add_assignments(df):
    T = len(df.index)
    assignment_list = []
    for t in range(T):
        data = df.iloc[t]
        assignments, costs = find_assignments(data)
        assignment_list.append(assignments)
    df['assignments'] = assignment_list

# instead of using current d1, use assigned defender to ball handler
# use in_the_paint to determine pick and rolls, and generate positive if that boolean is positive


def findSecondClosest(array):
    # Just initialize first two variable for further comparison. I made their values
    # extra high just to be safe
    first = 1000
    second = 1000

    # loop through whole offensive team
    for player in array:

        # if current player is closer than the first, then swap first and second,
        # and make first the closest player (which would be the current one)
        if player < first:
            second = first
            first = player

        # if the current player in the loop is the third closest
        elif first < player < second:
            second = player

    return second


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


def detect_pick_and_roll(df):

    ball_handler(df)
    add_assignments(df)

    #Assignment info:
    #Defenders are default array index, offenders are array values
    #add_assignments(df)

    bool_list = [False] * len(df.index)

    for t, data in df.iterrows():

        if is_in_the_paint(data['x_ball'], data['y_ball']) or data['ball_handler'] < 0:
            continue

        # find offensive
        dist_o = []
        dist_d = []
        for p in range(5):
            # offense
            dx = df.iloc[t].x_ball - df.iloc[t]['x_off_%i' % p]
            dy = df.iloc[t].y_ball - df.iloc[t]['y_off_%i' % p]
            dist_o.append(np.sqrt(dx ** 2 + dy ** 2))

            # defense
            dx = df.iloc[t].x_ball - df.iloc[t]['x_def_%i' % p]
            dy = df.iloc[t].y_ball - df.iloc[t]['y_def_%i' % p]
            dist_d.append(np.sqrt(dx ** 2 + dy ** 2))

        currentBallHandler = df.ball_handler.values
        closeTeammate = df.closest_teammate.values

        #assignments
        current_assignments = df.iloc[t]['assignments']

        o1 = currentBallHandler[t]
        o2 = closeTeammate[t]
        d1 = current_assignments.index(o1)

        # Calculate distances of the two player closest to the ball
        dx = df.iloc[t]['x_off_%i' % o1] - df.iloc[t]['x_off_%i' % o2]
        dy = df.iloc[t]['y_off_%i' % o1] - df.iloc[t]['y_off_%i' % o2]
        dist_o1_o2 = np.sqrt(dx ** 2 + dy ** 2)

        # Calculate distances between o1 and d1
        dx = df.iloc[t]['x_off_%i' % o1] - df.iloc[t]['x_def_%i' % d1]
        dy = df.iloc[t]['y_off_%i' % o1] - df.iloc[t]['y_def_%i' % d1]
        dist_o1_d1 = np.sqrt(dx ** 2 + dy ** 2)

        # Calculate distances between o1 and d1
        dx = df.iloc[t]['x_off_%i' % o2] - df.iloc[t]['x_def_%i' % d1]
        dy = df.iloc[t]['y_off_%i' % o2] - df.iloc[t]['y_def_%i' % d1]
        dist_o2_d1 = np.sqrt(dx ** 2 + dy ** 2)

        distance_list = []

        distance_list.append(dist_o1_o2)
        distance_list.append(dist_o1_d1)
        distance_list.append(dist_o2_d1)

        if dist_o1_o2 < 6 and dist_o1_d1 < 6 and dist_o2_d1 < 3:
            bool_list[t] = True

    df['is_pick_and_roll'] = bool_list


"""

Test out new program, then additional condition.
make adjustments otherwise.

"""
