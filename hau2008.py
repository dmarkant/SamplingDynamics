import numpy as np
import pandas as pd


GAMBLES_STUDY_1 = [[np.array([[3, 1.], [0., 0.]]),
                    np.array([[4, .8], [0., .2]])],
                   [np.array([[3, .25], [0., .75]]),
                    np.array([[4, .2], [0., .8]])],
                   [np.array([[-32, .1], [0., .9]]),
                    np.array([[-3, 1.], [0., 0]])],
                   [np.array([[-4, .8], [0., .2]]),
                    np.array([[-3, 1.], [0., 0]])],
                   [np.array([[3, 1.], [0., 0]]),
                    np.array([[32, .1], [0., 0.9]])],
                   [np.array([[3, .25], [0., .75]]),
                    np.array([[32, .025], [0., .975]])]
                    ]



pth = "/Users/markant/code/SamplingDynamics/data/Hau2008/"


files = ["Hau08_s1.sampling_117.0.txt",
         "Hau08_s2.sampling_118.0.txt",
         "Hau08_s3.sampling_119.0.txt"]


def load_study1():
    return pd.read_table(pth+files[0], sep=' ')



