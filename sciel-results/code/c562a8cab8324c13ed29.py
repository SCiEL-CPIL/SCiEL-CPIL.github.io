import time
import numpy as np

def _score_arr(U):
    if len(U) < 2:
        return 0.0
    mx = int(U[-1])
    if mx <= 0:
        return 0.0
    s = np.unique(U[:, None] + U[None, :])
    d = np.unique(U[:, None] - U[None, :])
    ratio = len(d) / len(s)
    if ratio <= 1.0:
        return 1.0
    return float(1.0 + np.log2(ratio) / np.log2(2*mx + 1))

def _score_set(Uset):
    U = np.array(sorted(Uset), dtype=np.int64)
    return _score_arr(U)

def construct():
    deadline = time.time() + 110
    rng = np.random.default_rng(12345)
    best = [0, 3, 9, 27]
    best_score = _score_set(set(best))
    M_max = 200

    def try_update(trial):
        nonlocal best, best_score
        if 0 not in trial or len(trial) < 2:
            return
        sc = _score_set(trial)
        if sc > best_score:
            best_score = sc
            best = list(trial)

    def greedy(cur, cur_score, budget):
        nonlocal best, best_score
        improved = True
        while improved and time.time() < budget:
            improved = False
            for x in range(0, M_max + 1):
                if time.time() > budget:
                    break
                if x in cur:
                    trial = cur - {x}
                    if not (0 in trial and len(trial) >= 2):
                        continue
                else:
                    trial = cur | {x}
                sc = _score_set(trial)
                if sc > cur_score + 1e-12:
                    cur = trial
                    cur_score = sc
                    improved = True
                    if sc > best_score:
                        best_score = sc
                        best = list(trial)
        return cur, cur_score

    seeds = [
        {0, 3, 9, 27},
        {0, 1, 3, 9, 27},
        {0, 2, 4, 8, 16, 32},
        {0, 3, 6, 9, 12, 15, 18},
        {0, 5, 10, 15, 20, 25},
        {0, 3, 9, 27, 81},
    ]
    for seed in seeds:
        if time.time() > deadline - 45:
            break
        cur = set(seed)
        cur_score = _score_set(cur)
        if cur_score > best_score:
            best_score = cur_score
            best = list(cur)
        cur, cur_score = greedy(cur, cur_score, deadline - 35)

    # Simulated annealing to escape local optima
    if time.time() < deadline - 5:
        cur = set(best)
        cur_score = _score_set(cur)
        T = 0.08
        T_min = 1e-6
        cooling = 0.997
        while T > T_min and time.time() < deadline - 2:
            x = int(rng.integers(0, M_max + 1))
            if x in cur:
                trial = cur - {x}
                if not (0 in trial and len(trial) >= 2):
                    T *= cooling
                    continue
            else:
                trial = cur | {x}
            sc = _score_set(trial)
            delta = sc - cur_score
            if delta > 0 or (rng.random() < np.exp(delta / T)):
                cur = trial
                cur_score = sc
                if sc > best_score:
                    best_score = sc
                    best = list(trial)
            T *= cooling

    # Final greedy polish
    if time.time() < deadline - 1:
        cur = set(best)
        cur_score = _score_set(cur)
        greedy(cur, cur_score, deadline - 0.5)

    return best
