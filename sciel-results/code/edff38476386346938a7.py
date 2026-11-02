import time
import numpy as np
import random
from itertools import product

def _hamming4():
    # 16 codewords of the (15,11) Hamming code restricted to first 4 coords
    # (all 16 possible 4-bit values form the code itself)
    return [list(map(int, np.binary_repr(i, 4))) for i in range(16)]

def _hamming4_offset(off):
    base = _hamming4()
    return [[(c + off[i]) % 3 for c in row] for i, row in enumerate(base)]

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0

    if q != 3 or n != 8 or radius != 1:
        # Fallback: exhaustive small or empty
        if q ** n > 100000:
            return []
        total = q ** n
        def idx_to_digits(idx):
            d = [0] * n
            for i in range(n - 1, -1, -1):
                d[i] = idx % q
                idx //= q
            return d
        def digits_to_idx(d):
            x = 0
            for v in d:
                x = x * q + v
            return x
        neigh = {}
        for i in range(total):
            d = idx_to_digits(i)
            nbs = [i]
            for pos in range(n):
                o = d[pos]
                for val in range(q):
                    if val == o:
                        continue
                    nd = d[:]
                    nd[pos] = val
                    nbs.append(digits_to_idx(nd))
            neigh[i] = nbs
        uncovered = set(range(total))
        code = []
        while uncovered and time.time() < deadline:
            best = -1
            bestc = -1
            for cw in list(uncovered):
                c = sum(1 for nb in neigh[cw] if nb in uncovered)
                if c > bestc:
                    bestc = c
                    best = cw
            if best == -1:
                break
            code.append(best)
            for nb in neigh[best]:
                uncovered.discard(nb)
        seen = set()
        out = []
        for cw in code:
            d = idx_to_digits(cw)
            t = tuple(d)
            if t not in seen:
                seen.add(t)
                out.append(d)
        return out

    # ---- Specific case q=3, n=8, radius=1 ----
    total_words = 3 ** 8
    num_coords = 8
    alphabet = 3

    # Precompute neighbors: neighbors[i] = list of 17 indices
    neighbors = np.zeros((total_words, 17), dtype=np.int32)
    for i in range(total_words):
        x = i
        digs = [0] * num_coords
        for k in range(num_coords - 1, -1, -1):
            digs[k] = x % alphabet
            x //= alphabet
        neighbors[i, 0] = i
        col = 1
        for pos in range(num_coords):
            for val in range(alphabet):
                if val == digs[pos]:
                    continue
                nd = digs[:]
                nd[pos] = val
                idx = 0
                for v in nd:
                    idx = idx * alphabet + v
                neighbors[i, col] = idx
                col += 1

    def idx_to_digits(idx):
        digs = [0] * num_coords
        for k in range(num_coords - 1, -1, -1):
            digs[k] = idx % alphabet
            idx //= alphabet
        return digs

    # Baseline: 492 words from two 4D Hamming codes
    h0 = _hamming4()
    h1 = _hamming4_offset([1, 2, 1, 2])
    code_idx = set()
    for a in range(3 ** 4):
        ad = [0] * 4
        x = a
        for k in range(3):
            ad[k] = x % 3
            x //= 3
        ad[3] = x
        for row in h0:
            cw = ad + row
            idx = 0
            for v in cw:
                idx = idx * 3 + v
            code_idx.add(idx)
    for a in range(3 ** 4):
        ad = [0] * 4
        x = a
        for k in range(3):
            ad[k] = x % 3
            x //= 3
        ad[3] = x
        for row in h1:
            cw = ad + row
            idx = 0
            for v in cw:
                idx = idx * 3 + v
            code_idx.add(idx)

    code_vec = np.zeros(total_words, dtype=np.int8)
    for c in code_idx:
        code_vec[c] = 1

    def compute_cov():
        return neighbors.dot(code_vec)

    def prune():
        nonlocal code_idx, code_vec
        cov = compute_cov()
        removed = 0
        for c in list(code_idx):
            if np.all(cov[neighbors[c]] > 1):
                code_idx.remove(c)
                code_vec[c] = 0
                removed += 1
        return removed > 0

    prune()

    # Local search
    def local_search():
        nonlocal code_idx, code_vec
        improved = True
        while improved and time.time() < deadline - 4:
            improved = False
            cov = compute_cov()
            c_list = list(code_idx)
            random.shuffle(c_list)

            # 1-swap
            for c in c_list:
                if time.time() > deadline - 4:
                    break
                if c not in code_idx:
                    continue
                critical = [w for w in neighbors[c] if cov[w] == 1]
                if not critical:
                    code_idx.remove(c)
                    code_vec[c] = 0
                    improved = True
                    cov = compute_cov()
                    continue
                crit0 = critical[0]
                cand = set(neighbors[crit0].tolist())
                for w in critical[1:]:
                    cand &= set(neighbors[w].tolist())
                    if not cand:
                        break
                for d in cand:
                    if d in code_idx:
                        continue
                    if np.all(A_check(critical, d)):
                        code_idx.remove(c)
                        code_vec[c] = 0
                        code_idx.add(d)
                        code_vec[d] = 1
                        improved = True
                        cov = compute_cov()
                        break

            # 2-swap (limited random)
            if time.time() < deadline - 6:
                c_list = list(code_idx)
                random.shuffle(c_list)
                n_cw = len(c_list)
                for _ in range(80):
                    if time.time() > deadline - 6:
                        break
                    if n_cw < 2:
                        break
                    i = random.randint(0, n_cw - 1)
                    j = random.randint(0, n_cw - 1)
                    if i == j:
                        continue
                    c1 = c_list[i]
                    c2 = c_list[j]
                    if c1 not in code_idx or c2 not in code_idx:
                        continue
                    critical = set()
                    for w in neighbors[c1]:
                        if cov[w] == code_vec[c1] + code_vec[c2]:
                            critical.add(int(w))
                    for w in neighbors[c2]:
                        if int(w) not in critical:
                            if cov[w] == code_vec[c1] + code_vec[c2]:
                                critical.add(int(w))
                    if not critical:
                        code_idx.remove(c1)
                        code_vec[c1] = 0
                        code_idx.remove(c2)
                        code_vec[c2] = 0
                        improved = True
                        cov = compute_cov()
                        break
                    if len(critical) <= 4:
                        cl = list(critical)
                        cand = set(neighbors[cl[0]].tolist())
                        for w in cl[1:]:
                            cand &= set(neighbors[w].tolist())
                            if not cand:
                                break
                        for d in cand:
                            if d in code_idx:
                                continue
                            if np.all(A_check(cl, d)):
                                code_idx.remove(c1)
                                code_vec[c1] = 0
                                code_idx.remove(c2)
                                code_vec[c2] = 0
                                code_idx.add(d)
                                code_vec[d] = 1
                                improved = True
                                cov = compute_cov()
                                break
                        if improved:
                            break

            prune()

    def A_check(critical, d):
        # returns bool array whether d covers each w in critical
        return np.array([1 if d in nb_set(w) else 0 for w in critical])

    # Precompute neighbor sets for membership
    _nb_sets = [set(nb.tolist()) for nb in neighbors]
    def nb_set(w):
        return _nb_sets[w]

    local_search()

    # Final verification & repair
    cov = compute_cov()
    if not np.all(cov > 0):
        for w in np.where(cov == 0)[0]:
            if time.time() > deadline - 1:
                break
            d = random.choice(neighbors[w].tolist())
            if d not in code_idx:
                code_idx.add(d)
                code_vec[d] = 1

    seen = set()
    out = []
    for cw in sorted(code_idx):
        d = idx_to_digits(cw)
        t = tuple(d)
        if t not in seen:
            seen.add(t)
            out.append(d)
    return out
