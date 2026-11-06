"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def _is_sidon_with_diagonals(subset):
    """Check if subset has all unordered pair sums unique including diagonals (i.e. i<=j)."""
    n = len(subset)
    s = set()
    for i in range(n):
        for j in range(i, n):
            v = subset[i] + subset[j]
            if v in s:
                return False
            s.add(v)
    return True

def _fast_is_sidon(lst):
    """Fast Sidon check for sorted list."""
    n = len(lst)
    sums = set()
    for i in range(n):
        a = lst[i]
        for j in range(i, n):
            s = a + lst[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _max_sidon_size(ambient):
    """Compute max Sidon subset size for sorted 12-element list."""
    n = len(ambient)
    for k in range(min(n, 6), 1, -1):
        for combo in itertools.combinations(ambient, k):
            if _fast_is_sidon(list(combo)):
                return k
    return 2

def _random_ambient(rng):
    return sorted(rng.sample(range(129), 12))

def _local_search(ambient, rng, deadline, current_best_k):
    """Local search: swap one element at a time."""
    best = list(ambient)
    best_k = current_best_k
    no_improve = 0
    max_iters = 200
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        if best_k <= 3:
            break
        idx = rng.randint(0, 11)
        new_val = rng.randint(0, 128)
        if new_val in best:
            no_improve += 1
            continue
        trial = best[:]
        trial[idx] = new_val
        trial.sort()
        k = _max_sidon_size(trial)
        if k < best_k:
            best = trial
            best_k = k
            no_improve = 0
        else:
            no_improve += 1
            if k == best_k and rng.random() < 0.1:
                best = trial
                no_improve = 0
        if no_improve >= 80:
            break
    
    return best, best_k

def _candidates():
    cands = []
    cands.append(list(range(12)))
    cands.append(list(range(0, 24, 2)))
    cands.append(list(range(0, 36, 3)))
    cands.append(list(range(0, 48, 4)))
    cands.append(list(range(0, 60, 5)))
    cands.append(list(range(0, 72, 6)))
    cands.append(list(range(0, 96, 8)))
    cands.append(list(range(0, 120, 10)))
    cands.append(list(range(0, 132, 11)))
    cands.append([0,1,2,3,4,5,6,7,8,9,10,12])
    cands.append(list(range(1, 13)))
    cands.append([0,1,2,3,4,5,6,7,8,9,11,13])
    return cands

def _select_best_ambient():
    deadline = time.time() + 80
    best = None
    best_k = None
    
    # Test hand-picked candidates
    for cand in _candidates():
        if len(cand) != 12:
            continue
        if len(set(cand)) != 12 or any(x < 0 or x > 128 for x in cand):
            continue
        k = _max_sidon_size(cand)
        if best_k is None or k < best_k:
            best = cand
            best_k = k
            if best_k <= 3:
                break
        if time.time() > deadline:
            break
    
    # Random search with local improvement
    rng = random.Random(42)
    for trial in range(500):
        if time.time() > deadline - 2:
            break
        if best_k is not None and best_k <= 3:
            break
        cand = _random_ambient(rng)
        k = _max_sidon_size(cand)
        if best_k is None or k < best_k:
            best = cand
            best_k = k
            if best_k <= 3:
                break
        if time.time() < deadline - 10:
            improved, ik = _local_search(cand, rng, deadline - 5, best_k if best_k is not None else 12)
            if best_k is None or ik < best_k:
                best = improved
                best_k = ik
                if best_k <= 3:
                    break
    
    if best is None:
        return list(range(12))
    return best

def construct():
    return _select_best_ambient()
# EVOLVE-BLOCK-END
