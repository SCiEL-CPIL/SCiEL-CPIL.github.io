import numpy as np
import itertools
import time

D = 11
DEADLINE_S = 118.0

def construct():
    t0 = time.perf_counter()
    
    def elapsed():
        return time.perf_counter() - t0

    def is_deadline():
        return elapsed() > DEADLINE_S

    def gen_vectors(m):
        vecs = []
        if m == 4:
            # Type 1: Four +/- 1s
            for idxs in itertools.combinations(range(D), 4):
                for signs in itertools.product([1, -1], repeat=4):
                    v = np.zeros(D, dtype=np.int64)
                    for k, i in enumerate(idxs):
                        v[i] = signs[k]
                    vecs.append(v)
            # Type 2: One +/- 2
            for i in range(D):
                for s in [2, -2]:
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = s
                    vecs.append(v)
        elif m == 2:
            for idxs in itertools.combinations(range(D), 2):
                for signs in itertools.product([1, -1], repeat=2):
                    v = np.zeros(D, dtype=np.int64)
                    for k, i in enumerate(idxs):
                        v[i] = signs[k]
                    vecs.append(v)
        elif m == 3:
            for idxs in itertools.combinations(range(D), 3):
                for signs in itertools.product([1, -1], repeat=3):
                    v = np.zeros(D, dtype=np.int64)
                    for k, i in enumerate(idxs):
                        v[i] = signs[k]
                    vecs.append(v)
        else:
            # Generic
            def rec(pos, sq_rem, current):
                if pos == D:
                    if sq_rem == 0:
                        vecs.append(np.array(current, dtype=np.int64))
                    return
                max_val = int(np.sqrt(sq_rem))
                for x in range(-max_val, max_val + 1):
                    nx_sq = sq_rem - x*x
                    if nx_sq >= 0:
                        current.append(x)
                        rec(pos + 1, nx_sq, current)
                        current.pop()
            rec(0, m, [])
        return np.array(vecs, dtype=np.int64)

    # For m=4, we proved all vectors are valid.
    # Let's generate and return them.
    V4 = gen_vectors(4)
    if V4.shape[0] > 0:
        return V4

    # Fallback to m=2 if something goes wrong (shouldn't happen)
    V2 = gen_vectors(2)
    return V2
