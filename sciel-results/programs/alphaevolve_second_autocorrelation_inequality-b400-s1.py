import numpy as np

N = 50

def construct():
        # Sidon set of size 7 in {0, ..., 49}
        S = [0, 1, 4, 10, 18, 23, 30]
        h = [0.0] * N
        for idx in S:
            h[idx] = 1.0 / 7.0
        return h
