import time
import math
import numpy as np
from itertools import permutations

def construct(n):
    t0 = time.time()
    DEADLINE = 140.0  # safety margin under 150s wall-clock

    def classic(n):
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                w = tuple(word[i:i + m - 1])
                if len(set(w)) == m - 1 and w not in seen:
                    seen.add(w)
                    perms.append(w)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                o = 0
                for cand in range(min(len(new), len(block)), 0, -1):
                    if new[len(new) - cand:] == block[:cand]:
                        o = cand
                        break
                new += block[o:]
            word = new
        return word

    def verify(word, n):
        seen = set()
        for i in range(len(word) - n + 1):
            w = word[i:i + n]
            if len(set(w)) != n:
                return False
            seen.add(tuple(w))
        return len(seen) == math.factorial(n)

    if n <= 1:
        return "1"

    base = list(range(1, n))
    perms = [list(p) for p in permutations(base)]
    K = len(perms)
    L = n - 1

    # Precompute overlap table using numpy for speed
    # ov[i][j] = longest suffix of perms[i] == prefix of perms[j]
    ov = np.zeros((K, K), dtype=np.int8)
    for cand in range(L - 1, 0, -1):
        # For each pair, check if suffix of length cand of perms[i] == prefix of length cand of perms[j]
        # Extract suffixes and prefixes as integer keys
        # suffix of perms[i] of length cand: perms[i][L-cand:]
        # prefix of perms[j] of length cand: perms[j][:cand]
        # Convert to integer for fast comparison
        suffixes = np.array([sum((perms[i][L-cand+k] - 1) * (10 ** (cand-1-k)) for k in range(cand)) for i in range(K)], dtype=np.int64)
        prefixes = np.array([sum((perms[j][k] - 1) * (10 ** (cand-1-k)) for k in range(cand)) for j in range(K)], dtype=np.int64)
        # Find pairs where suffix[i] == prefix[j] and not already set
        for i in range(K):
            matches = np.where(prefixes == suffixes[i])[0]
            for j in matches:
                if i != j and ov[i, j] == 0:
                    ov[i, j] = cand
    ov = ov.tolist()

    # Nearest-neighbor from several deterministic starts
    best_order = None
    best_score = -1
    for s in range(min(K, 16)):
        order = [s]
        vis = bytearray(K)
        vis[s] = 1
        cur = s
        score = 0
        row = ov[cur]
        for _ in range(K - 1):
            bo = -1
            bo2 = -1
            for j in range(K):
                if not vis[j]:
                    o = row[j]
                    if o > bo2:
                        bo2 = o
                        bo = j
            order.append(bo)
            vis[bo] = 1
            score += bo2
            cur = bo
            row = ov[cur]
        if score > best_score:
            best_score = score
            best_order = order

    order = best_order
    cur_score = best_score

    # Local search with 2-opt and Or-opt
    def try_improve():
        nonlocal cur_score
        improved = False
        # 2-opt
        for i in range(1, K - 1):
            a = order[i - 1]
            for j in range(i + 1, K):
                old = ov[a][order[i]]
                if j + 1 < K:
                    old += ov[order[j]][order[j + 1]]
                new = ov[a][order[j]]
                if j + 1 < K:
                    new += ov[order[i]][order[j + 1]]
                if new > old:
                    order[i:j + 1] = order[i:j + 1][::-1]
                    cur_score += (new - old)
                    improved = True
                    break
            if improved:
                break
        if improved:
            return True
        # Or-opt: move a single element
        for i in range(1, K - 1):
            x = order[i]
            left = order[i - 1]
            right = order[i + 1]
            base_old = ov[left][x] + ov[x][right]
            base_new = ov[left][right]
            for j in range(0, K - 1):
                if j == i or j == i - 1:
                    continue
                ins_old = ov[order[j]][order[j + 1]]
                ins_new = ov[order[j]][x] + ov[x][order[j + 1]]
                if (base_new + ins_new) > (base_old + ins_old):
                    order.pop(i)
                    jj = j if j < i else j - 1
                    order.insert(jj + 1, x)
                    cur_score += (base_new + ins_new) - (base_old + ins_old)
                    return True
        return False

    # Run local search
    passes = 0
    while passes < 2000 and time.time() < DEADLINE - 30:
        passes += 1
        if not try_improve():
            # Try a random perturbation to escape local optimum
            if time.time() < DEADLINE - 35:
                # Random 2-opt perturbation
                for _ in range(5):
                    i = 1 + int(np.random.randint(0, K - 2))
                    j = i + 1 + int(np.random.randint(0, K - i - 1))
                    old = ov[order[i-1]][order[i]]
                    if j + 1 < K:
                        old += ov[order[j]][order[j+1]]
                    new = ov[order[i-1]][order[j]]
                    if j + 1 < K:
                        new += ov[order[i]][order[j+1]]
                    order[i:j+1] = order[i:j+1][::-1]
                    cur_score += (new - old)
            else:
                break
        if time.time() >= DEADLINE - 30:
            break

    # Build the word
    word = []
    prev = None
    for idx in order:
        p = perms[idx]
        block = p + [n] + p
        if prev is None:
            word += block
        else:
            word += block[ov[prev][idx]:]
        prev = idx

    if not verify(word, n):
        word = classic(n)

    return "".join(map(str, word))
