import numpy as np

def count_streaks(data):
    count_streak = [0]
    for trial, option in enumerate(data):
        if trial==0:
            count_streak.append(1)
        else:
            if option == data[trial-1]:
                count_streak.append(count_streak[-1] + 1)
            else:
                count_streak.append(1)
    return count_streak



def streak_lengths(samples):
    
    lengths = []
    current_length = 0
    for trial, option in enumerate(samples):

        if trial==0:
            current_length = 1
        else:
            if option == samples[trial - 1]:
                current_length += 1
            else:
                lengths.append(current_length)
                current_length = 1
    lengths.append(current_length)
    return np.array(lengths)


