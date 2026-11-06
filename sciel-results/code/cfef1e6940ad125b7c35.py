import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _precompute_aps():
    """
    Precompute all 3-term arithmetic progressions in 1..512.
    """
    ap_by_elem = [[] for _ in range(513)]
    for d in range(1, 256):
        max_a = 512 - 2 * d
        for a in range(1, max_a + 1):
            b = a + d
            c = a + 2 * d
            ap_by_elem[a].append((d, b, c))
            ap_by_elem[b].append((d, a, c))
            ap_by_elem[c].append((d, a, b))
    return ap_by_elem

AP_BY_ELEM = _precompute_aps()

def _init_state(S):
    """
    Initialize state. Returns (S_set, mask, d_count, score).
    """
    S_set = set(S)
    mask = np.zeros(513, dtype=np.uint8)
    for x in S:
        mask[x] = 1
    d_count = np.zeros(256, dtype=np.int32)
    for d in range(1, 256):
        end = 513 - 2 * d
        if end <= 0:
            break
        d_count[d] = int(np.sum(mask[0:end] & mask[d:d + end] & mask[2 * d:2 * d + end]))
    score = int(np.sum(d_count > 0))
    return S_set, mask, d_count, score

def _apply_add(S_set, mask, d_count, score, x):
    for d, e1, e2 in AP_BY_ELEM[x]:
        if mask[e1] and mask[e2]:
            if d_count[d] == 0:
                score += 1
            d_count[d] += 1
    mask[x] = 1
    S_set.add(x)
    return score

def _apply_remove(S_set, mask, d_count, score, x):
    for d, e1, e2 in AP_BY_ELEM[x]:
        if mask[e1] and mask[e2]:
            d_count[d] -= 1
            if d_count[d] == 0:
                score -= 1
    mask[x] = 0
    S_set.discard(x)
    return score

def _simulated_annealing(S_init, time_budget, rng):
    S = list(S_init)
    S_set, mask, d_count, current_score = _init_state(S)
    best_S = list(S)
    best_score = current_score
    start_time = time.time()
    T = 5.0
    cooling = 0.9995
    while time.time() - start_time < time_budget:
        T *= cooling
        if T < 0.005:
            break
        i = rng.randint(0, 31)
        old_elem = S[i]
        new_elem = rng.randint(1, 512)
        if new_elem == old_elem or new_elem in S_set:
            continue
        _apply_remove(S_set, mask, d_count, current_score, old_elem)
        new_score = _apply_add(S_set, mask, d_count, current_score, new_elem)
        delta = new_score - current_score
        if delta > 0 or (delta == 0 and rng.random() < 0.5) or rng.random() < np.exp(delta / T):
            current_score = new_score
            S[i] = new_elem
            if current_score > best_score:
                best_score = current_score
                best_S = list(S)
        else:
            _apply_remove(S_set, mask, d_count, current_score, new_elem)
            _apply_add(S_set, mask, d_count, current_score, old_elem)
    return best_S, best_score

def _greedy_polish(S_init, time_limit=5.0):
    S = list(S_init)
    S_set, mask, d_count, current_score = _init_state(S)
    start_time = time.time()
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        for i in range(32):
            old_val = S[i]
            best_cand = None
            best_new_score = current_score
            _apply_remove(S_set, mask, d_count, current_score, old_val)
            score_without_old = current_score
            for cand in range(1, 513):
                if cand in S_set:
                    continue
                new_score = _apply_add(S_set, mask, d_count, score_without_old, cand)
                if new_score > best_new_score:
                    best_new_score = new_score
                    best_cand = cand
                    _apply_remove(S_set, mask, d_count, new_score, cand)
                else:
                    _apply_remove(S_set, mask, d_count, new_score, cand)
            current_score = _apply_add(S_set, mask, d_count, score_without_old, old_val)
            if best_cand is not None:
                _apply_remove(S_set, mask, d_count, current_score, old_val)
                current_score = _apply_add(S_set, mask, d_count, current_score, best_cand)
                S[i] = best_cand
                improved = True
    return S, current_score

def _structured_init(rng):
    r = rng.random()
    if r < 0.3:
        start = rng.randint(1, 480)
        S = sorted(set(range(start, start + 32)))
        while len(S) < 32:
            S.append(rng.randint(1, 512))
        return sorted(set(S))[:32]
    elif r < 0.6:
        step = 512 // 32
        S = [rng.randint(1, step * (i + 1)) for i in range(32)]
        S = sorted(set(S))
        while len(S) < 32:
            S.append(rng.randint(1, 512))
        return sorted(set(S))[:32]
    else:
        return sorted(rng.sample(range(1, 513), 32))

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    rng = random.Random(123)
    best_S = list(range(1, 33))
    best_score = _init_state(best_S)[3]
    num_restarts = 15
    per_restart_time = 7.0
    for restart in range(num_restarts):
        if time.time() > deadline - 8.0:
            break
        if restart == 0:
            S_init = list(range(1, 33))
        elif restart == 1:
            S_init = _structured_init(rng)
        elif restart == 2:
            S_init = sorted(set([max(1, min(512, int(512 * (i + 1) / 33))) for i in range(32)]))
            while len(S_init) < 32:
                S_init.append(rng.randint(1, 512))
            S_init = sorted(set(S_init))[:32]
        else:
            S_init = sorted(rng.sample(range(1, 513), 32))
        remaining = min(per_restart_time, deadline - time.time() - 8.0)
        if remaining < 2.0:
            break
        S, score = _simulated_annealing(S_init, remaining, rng)
        if time.time() < deadline - 5.0:
            S, score = _greedy_polish(S, time_limit=3.0)
        if score > best_score:
            best_score = score
            best_S = list(S)
    if time.time() < deadline - 2.0:
        best_S, best_score = _greedy_polish(best_S, time_limit=8.0)
    return sorted(best_S)

# EVOLVE-BLOCK-END
