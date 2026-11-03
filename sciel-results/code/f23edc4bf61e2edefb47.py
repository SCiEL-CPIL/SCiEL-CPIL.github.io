"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def construct():
    deadline = time.time() + 140
    
    def norm_sq(v):
        return int(np.dot(v, v))
    
    def check_constraint(vecs):
        """Check if the set satisfies the constraint. Return (valid, M)."""
        if not vecs:
            return True, 0
        M = max(norm_sq(v) for v in vecs)
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                d = norm_sq(vecs[i]) + norm_sq(vecs[j]) - 2 * int(np.dot(vecs[i], vecs[j]))
                if d < M:
                    return False, M
        return True, M
    
    def can_add(vecs, v, M):
        """Check if v can be added to vecs with current max norm M. Returns new M if valid."""
        nv = norm_sq(v)
        new_M = max(M, nv)
        for ui in vecs:
            d = norm_sq(ui) + nv - 2 * int(np.dot(ui, v))
            if d < new_M:
                return None
        return new_M
    
    # Start with the seed set: all ±e_i ± e_j
    vecs = []
    M = 0
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                new_M = can_add(vecs, v, M)
                if new_M is not None:
                    vecs.append(v)
                    M = new_M
    
    best_count = len(vecs)
    
    # Try adding vectors with 3 nonzero entries
    for i, j, k in itertools.combinations(range(D), 3):
        if time.time() > deadline:
            break
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    v[k] = sk
                    new_M = can_add(vecs, v, M)
                    if new_M is not None:
                        vecs.append(v)
                        M = new_M
                        if len(vecs) > best_count:
                            best_count = len(vecs)
    
    # Try adding vectors with 4 nonzero entries
    for i, j, k, l in itertools.combinations(range(D), 4):
        if time.time() > deadline:
            break
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    for sl in (1, -1):
                        v = np.zeros(D, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        v[k] = sk
                        v[l] = sl
                        new_M = can_add(vecs, v, M)
                        if new_M is not None:
                            vecs.append(v)
                            M = new_M
                            if len(vecs) > best_count:
                                best_count = len(vecs)
    
    # Local search with random perturbations
    rng = np.random.default_rng(123)
    for iteration in range(200000):
        if time.time() > deadline:
            break
        v = rng.integers(-2, 3, size=D)
        if np.all(v == 0):
            continue
        new_M = can_add(vecs, v, M)
        if new_M is not None:
            vecs.append(v)
            M = new_M
    
    return np.array(vecs, dtype=np.int64)

# EVOLVE-BLOCK-END
