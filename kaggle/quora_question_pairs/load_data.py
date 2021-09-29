import pandas as pd


class LoadData:
    def __init__(self):
        pass

    @staticmethod
    def load_train():
        path = './data/' + 'train.csv'
        data = pd.read_csv(path)
        return data
        pass

    @staticmethod
    def load_test():
        path = './data/' + 'test_inner.csv'
        data = pd.read_csv(path)
        return data
        pass
    pass




