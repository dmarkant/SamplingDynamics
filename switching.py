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

