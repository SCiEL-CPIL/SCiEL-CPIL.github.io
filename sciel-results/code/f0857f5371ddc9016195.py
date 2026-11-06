"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def _fast_is_sidon(lst):
    """Fast Sidon check for sorted list with diagonal sums (i<=j)."""
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

def _random_ambient(rng, center=64, spread=20):
    """Generate random ambient set clustered around center."""
    vals = set()
    while len(vals) < 12:
        v = rng.randint(max(0, center - spread), min(128, center + spread))
        vals.add(v)
    return sorted(vals)

def _local_search(ambient, rng, deadline, current_best_k):
    """Local search: swap one element at a time to minimize max Sidon size."""
    best = list(ambient)
    best_k = current_best_k
    no_improve = 0
    max_iters = 300
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        if best_k <= 3:
            break
        idx = rng.randint(0, 11)
        # Try to move element closer to others to increase collisions
        if best_k > 3:
            # Bias toward values close to existing elements
            if rng.random() < 0.7:
                neighbor = best[rng.randint(0, 11)]
                offset = rng.choice([-3, -2, -1, 1, 2, 3, 4, 5])
                new_val = neighbor + offset
            else:
                new_val = rng.randint(0, 128)
        else:
            new_val = rng.randint(0, 128)
        
        if new_val < 0 or new_val > 128 or new_val in best:
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
            if k == best_k and rng.random() < 0.05:
                best = trial
                no_improve = 0
        if no_improve >= 100:
            break
    
    return best, best_k

def _candidates():
    """Generate diverse candidate ambient sets."""
    cands = []
    # Consecutive integers - densest possible
    for start in range(0, 117):
        cands.append(list(range(start, start + 12)))
    # Arithmetic progressions with small steps
    for step in [2, 3, 4]:
        for start in range(0, 129 - 11 * step):
            cands.append(list(range(start, start + 12 * step, step)))
    # Concentrated sets: most elements in small range
    for base in range(0, 117):
        # 8 elements in [base, base+7], 4 elements spread out
        cands.append(sorted(list(range(base, base + 8)) + [base + 8, base + 10, base + 12, base + 14]))
    # Mix of close and far elements
    for base in range(0, 110):
        cands.append(sorted(list(range(base, base + 6)) + [base + 6, base + 7, base + 9, base + 11, base + 13, base + 15]))
    return cands

def _select_best_ambient():
    deadline = time.time() + 85  # Safety margin before 90s kill
    best = None
    best_k = None
    
    # Test hand-picked candidates
    for cand in _candidates():
        if time.time() > deadline:
            break
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
    
    # Random search with local improvement
    rng = random.Random(42)
    for trial in range(1000):
        if time.time() > deadline - 2:
            break
        if best_k is not None and best_k <= 3:
            break
        
        # Generate diverse random candidates
        strategy = rng.random()
        if strategy < 0.3:
            cand = _random_ambient(rng, center=rng.randint(0, 100), spread=rng.randint(5, 30))
        elif strategy < 0.6:
            # Clustered: 10 elements close, 2 far
            center = rng.randint(10, 100)
            close = set()
            while len(close) < 10:
                close.add(rng.randint(max(0, center - 8), min(128, center + 8)))
            far1 = rng.randint(0, 128)
            far2 = rng.randint(0, 128)
            cand = sorted(list(close) + [far1, far2])
            if len(set(cand)) != 12:
                continue
        else:
            # Random from full range
            cand = sorted(rng.sample(range(129), 12))
        
        k = _max_sidon_size(cand)
        if best_k is None or k < best_k:
            best = cand
            best_k = k
            if best_k <= 3:
                break
        
        if time.time() < deadline - 15:
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
