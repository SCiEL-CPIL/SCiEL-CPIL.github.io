import time
import math
import numpy as np

_START = time.time()
_DEADLINE = _START + 135.0   # safety margin under the 150s wall-clock kill

def _elapsed():
    return time.time() - _START

def _score_direct(U):
    """Vectorized score: 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)."""
    U = sorted(set(U))
    if not U or U[0] != 0:
        U = [0] + [x for x in U if x > 0]
    n = len(U)
    if n < 2:
        return 0.0
    M = U[-1]
    if M < 1:
        return 0.0
    U_arr = np.array(U, dtype=np.int64)
    sums = U_arr[:, None] + U_arr[None, :]
    diffs = U_arr[:, None] - U_arr[None, :]
    sumset = len(np.unique(sums))
    diffset = len(np.unique(diffs))
    if sumset == 0:
        return 0.0
    ratio = diffset / sumset
    if ratio <= 1.0:
        return 1.0
    base = 2 * M + 1
    if base <= 1:
        return 0.0
    return 1.0 + math.log(ratio) / math.log(base)

def _golomb_ruler(n):
    if n <= 1:
        return [0]
    known = {
        2: [0, 1],
        3: [0, 1, 3],
        4: [0, 1, 4, 6],
        5: [0, 1, 4, 10, 11],
        6: [0, 1, 4, 10, 12, 17],
        7: [0, 1, 4, 10, 12, 17, 22],
        8: [0, 1, 4, 10, 12, 17, 22, 29],
        9: [0, 1, 4, 10, 12, 17, 22, 29, 36],
        10: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44],
        11: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52],
        12: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60],
        13: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72],
        14: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 85],
        15: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 85, 99],
        16: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 85, 99, 115],
        17: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 85, 99, 115, 132],
        18: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 85, 99, 115, 132, 151],
        19: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 85, 99, 115, 132, 151, 171],
        20: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 85, 99, 115, 132, 151, 171, 193],
    }
    if n in known:
        return known[n]
    # Greedy construction fallback
    U = [0]
    for i in range(1, n):
        c = 1
        while True:
            U_test = sorted(U + [c])
            diffs = set()
            valid = True
            for j in range(len(U_test)):
                for k in range(j + 1, len(U_test)):
                    d = U_test[k] - U_test[j]
                    if d in diffs:
                        valid = False
                        break
                    diffs.add(d)
                if not valid:
                    break
            if valid:
                U.append(c)
                break
            c += 1
    return U

def _local_search(U, deadline):
    """Local search with add/remove/swap moves. Returns best set and score."""
    U = list(U)
    cur_score = _score_direct(U)
    improved = True
    while improved and _elapsed() < deadline:
        improved = False
        max_val = max(U) if U else 1

        # 1. Try adding an element (scan candidate range)
        add_lo = 1
        add_hi = max_val * 2 + 30
        for c in range(add_lo, add_hi + 1):
            if _elapsed() > deadline:
                break
            if c in U:
                continue
            trial = sorted(U + [c])
            s = _score_direct(trial)
            if s > cur_score + 1e-12:
                cur_score = s
                U = trial
                improved = True
                break
        if improved:
            continue

        # 2. Try removing an element
        if len(U) > 2:
            for idx in range(1, len(U)):
                if _elapsed() > deadline:
                    break
                trial = U[:idx] + U[idx + 1:]
                s = _score_direct(trial)
                if s > cur_score + 1e-12:
                    cur_score = s
                    U = trial
                    improved = True
                    break
        if improved:
            continue

        # 3. Try swapping an element
        if len(U) > 2:
            max_val = max(U) if U else 1
            for idx in range(1, len(U)):
                if _elapsed() > deadline:
                    break
                old_val = U[idx]
                for new_val in range(1, max_val * 2 + 15):
                    if new_val == old_val or new_val in U:
                        continue
                    trial = sorted(U[:idx] + [new_val] + U[idx + 1:])
                    s = _score_direct(trial)
                    if s > cur_score + 1e-12:
                        cur_score = s
                        U = trial
                        improved = True
                        break
                if improved:
                    break

    return U, cur_score

def construct():
    deadline = _START + 130.0  # leave ~20s buffer before the 150s kill

    best_U = [0, 1, 4, 10, 11]
    best_score = _score_direct(best_U)

    # Build a rich pool of seed candidates.
    candidates = []
    for n in range(3, 21):
        candidates.append(_golomb_ruler(n))
    for k in range(3, 11):
        candidates.append([0] + [i ** 2 for i in range(1, k)])
    for k in range(3, 8):
        candidates.append([0] + [i ** 3 for i in range(1, k)])

    rng = np.random.RandomState(42)
    for _ in range(30):
        size = int(rng.randint(3, 9))
        max_val = int(rng.randint(10, 60))
        cands = rng.choice(range(1, max_val + 1), size=size, replace=False)
        candidates.append(sorted([0] + list(cands)))

    # Evaluate the seed pool.
    for U in candidates:
        if _elapsed() > deadline - 20:
            break
        s = _score_direct(U)
        if s > best_score:
            best_score = s
            best_U = list(U)

    # Local search from the best seed.
    U, score = _local_search(best_U, deadline - 10)
    if score > best_score:
        best_score = score
        best_U = U

    # A few perturbation + re-search rounds to escape local optima.
    for _ in range(4):
        if _elapsed() > deadline - 5:
            break
        U = list(best_U)
        if len(U) > 2:
            idx = int(rng.randint(1, len(U) - 1))
            U.pop(idx)
        max_u = max(U) if U else 0
        new_val = int(rng.randint(1, max_u + max(10, max_u // 3)))
        if new_val not in U:
            U.append(new_val)
        U, score = _local_search(U, deadline - 3)
        if score > best_score:
            best_score = score
            best_U = U

    # Final high-precision polish: one more exhaustive single-move pass.
    if _elapsed() < deadline:
        U, score = _local_search(best_U, deadline)
        if score > best_score:
            best_score = score
            best_U = U

    if 0 not in best_U:
        best_U.append(0)
    return sorted(best_U)
