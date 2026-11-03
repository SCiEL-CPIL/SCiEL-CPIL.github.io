import time
import random
import numpy as np

def construct(q, n, radius):
    total = q ** n
    deadline = time.time() + 112  # safety margin under the 150s limit

    # ---- precompute the radius-1 neighborhood of every word ----
    words = np.array(np.unravel_index(np.arange(total), (q,) * n), dtype=np.int8).T
    neigh = np.empty((total, 1 + radius * (q - 1) * n), dtype=np.int32)
    for i in range(total):
        w = words[i]
        idxs = [i]
        for pos in range(n):
            ov = w[pos]
            for nv in range(q):
                if nv == ov:
                    continue
                w2 = w.copy()
                w2[pos] = nv
                idxs.append(int(np.ravel_multi_index(tuple(w2), (q,) * n)))
        neigh[i] = idxs

    def to_lists(idx):
        out = []
        for i in idx:
            w = words[i]
            out.append([int(w[0]), int(w[1]), int(w[2]), int(w[3]),
                        int(w[4]), int(w[5]), int(w[6])])
        return out

    # ---- greedy seed (vectorized gain) ----
    def greedy(seed):
        rng = random.Random(seed)
        covered = np.zeros(total, dtype=np.int32)
        code = []
        for _ in range(total):
            if np.all(covered > 0):
                break
            gains = np.zeros(total, dtype=np.int32)
            for c in code:
                gains[neigh[c]] -= 1
            gains[covered == 0] = 1
            if gains.max() <= 0:
                break
            m = gains.max()
            cands = np.where(gains == m)[0]
            c = int(cands[rng.randrange(len(cands))])
            code.append(c)
            covered[neigh[c]] += 1
        return code

    # ---- local search: repeatedly delete any codeword that covers no exclusive word ----
    def local_search(code, seed):
        code = list(code)
        counts = np.zeros(total, dtype=np.int32)
        for c in code:
            counts[neigh[c]] += 1
        changed = True
        while changed:
            changed = False
            # find codewords that cover at least one word they are the sole coverer of
            keep = []
            for i, c in enumerate(code):
                if int(np.sum(counts[neigh[c]] == 1)) > 0:
                    keep.append(i)
            if len(keep) != len(code):
                keepset = set(keep)
                newcode = []
                for i, c in enumerate(code):
                    if i in keepset:
                        newcode.append(c)
                    else:
                        counts[neigh[c]] -= 1
                code = newcode
                changed = True
        return code

    best_code = None
    best_size = float('inf')
    seed = 0
    while time.time() < deadline:
        code = greedy(seed)
        code = local_search(code, seed)
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
            if best_size <= 186:
                break
        seed += 1

    return to_lists(best_code)
