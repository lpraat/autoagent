class Dataset:
    """
    A generic dataset
    """
    def __getitem__(self, index):
        raise NotImplementedError()

    def size(self):
        raise NotImplementedError()
