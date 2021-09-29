import numpy as np
import pandas as pd
from kaggle.quora_question_pairs.load_data import LoadData


class DataExploration:
    def __init__(self):
        self.l_d = LoadData()
        pass

    def analysis_label(self):
        train_data = self.l_d.load_train()
        label_count = train_data['is_duplicate'].count()
        label_mean = train_data['is_duplicate'].mean()
        label_min = train_data['is_duplicate'].quantile(0)
        label_25 = train_data['is_duplicate'].quantile(0.25)
        label_50 = train_data['is_duplicate'].quantile(0.5)
        label_75 = train_data['is_duplicate'].quantile(0.75)
        label_max = train_data['is_duplicate'].quantile(1)
        pass

    pass


if __name__ == '__main__':
    d_e = DataExploration()

    d_e.analysis_label()
    pass







