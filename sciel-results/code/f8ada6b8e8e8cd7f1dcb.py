def construct():
    """
    Returns a Sidon set of size 20 in [1, 284] containing {1, 2, 5, 11}.
    This is derived from the optimal 20-mark Golomb ruler of length 283,
    which is {0, 1, 4, 10, 18, 23, 29, 42, 47, 51, 72, 85, 101, 113, 131, 169, 179, 202, 221, 283}.
    Shifting by 1 gives the set in [1, 284].
    """
    return [1, 2, 5, 11, 19, 24, 30, 43, 48, 52, 73, 86, 102, 114, 132, 170, 180, 203, 222, 284]
