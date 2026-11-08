import time
import random
import math
import numpy as np

def compute_score(U):
    U = sorted(set(U))
    if 0 not in U:
        U = sorted(set(U) | {0})
    if len(U) < 2:
        return 0.0
    M = max(U)
    if M == 0:
        return 0.0
    U_arr = np.array(U)
    sums = U_arr[:, None] + U_arr[None, :]
    diffs = U_arr[:, None] - U_arr[None, :]
    sumset_size = len(np.unique(sums))
    diffset_size = len(np.unique(diffs))
    ratio = diffset_size / sumset_size
    if ratio <= 1:
        return 1.0
    return 1.0 + math.log(ratio) / math.log(2 * M + 1)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    random.seed(42)
    initial_sets = [
        list(range(10)),
        [0] + [2**i for i in range(1, 12)],
        [0, 1, 3, 7, 15, 31],
        [0] + list(range(1, 15)),
        [0, 1, 2, 5, 7, 12, 14, 19],
    ]
    best_set, best_score = None, 0.0
    for init in initial_sets:
        if time.time() > deadline - 5:
            break
        s = compute_score(init)
        if s > best_score:
            best_score, best_set = s, list(init)
    if best_set is None:
        best_set, best_score = [0, 1, 3], compute_score([0, 1, 3])
    cur_set, cur_score = list(best_set), best_score
    T, cooling = 0.5, 0.9995
    while time.time() < deadline:
        mt = random.random()
        ns = list(cur_set)
        if mt < 0.25:
            mv = max(cur_set)
            nv = random.randint(1, mv + 5)
            if nv not in ns:
                ns.append(nv)
        elif mt < 0.45:
            rm = [x for x in ns if x != 0]
            if rm and len(ns) > 2:
                ns.remove(random.choice(rm))
        elif mt < 0.75:
            if len(ns) > 1:
                idx = random.randint(1, len(ns) - 1)
                mv = max(ns)
                ns[idx] = random.randint(1, mv + 5)
        else:
            if len(ns) > 2:
                i, j = random.sample(range(1, len(ns)), 2)
                ns[i], ns[j] = ns[j], ns[i]
        if len(ns) < 2:
            continue
        ns_score = compute_score(ns)
        if ns_score >= cur_score or random.random() < math.exp((ns_score - cur_score) / T):
            cur_set, cur_score = ns, ns_score
            if cur_score > best_score:
                best_score, best_set = cur_score, list(cur_set)
        T *= cooling
        if T < 1e-10:
            T = 1e-10
    return best_set
