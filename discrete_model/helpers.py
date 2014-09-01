def add_problem_labels(ax, label_h, ylim):
    ax.plot([21, 21], ylim, 'k--')
    ax.plot([42, 42], ylim, 'k--')
    ax.plot([63, 63], ylim, 'k--')
    t = ax.text(10, label_h, "Gain, HH-LL", ha="center", va="center", size=15)
    t = ax.text(31, label_h, "Gain, HL-LH", ha="center", va="center", size=15)
    t = ax.text(52, label_h, "Loss, HH-LL", ha="center", va="center", size=15)
    t = ax.text(74, label_h, "Loss, HL-LH", ha="center", va="center", size=15)
    return ax
