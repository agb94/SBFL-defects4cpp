import numpy as np

def diversity(X):
    N, _ = X.shape
    _, counts = np.unique(X > 0, axis=0, return_counts=True)
    if N > 1:
        value = 1 - (counts * (counts - 1)).sum()/(N * (N - 1))
    else:
        value = 1.
    assert 0 <= value <= 1
    return value

def density(X):
    N, M = X.shape
    value = np.sum(X > 0) / (N * M)
    assert 0 <= value <= 1
    return value

def uniqueness(X):
    _, M = X.shape
    unique_elems = np.unique(X > 0, axis=1)
    value = unique_elems.shape[1] / M
    assert 0 <= value <= 1
    return value

def DDU(X):
    return diversity(X) * density(X) * uniqueness(X)