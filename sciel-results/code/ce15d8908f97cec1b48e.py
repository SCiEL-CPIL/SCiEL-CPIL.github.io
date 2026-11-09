N = 18
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _bitonic_merge(n, comps, start, k, reverse):
    if k == 1:
        return
    _bitonic_merge(n, comps, start, k // 2, reverse)
    _bitonic_merge(n, comps, start + k // 2, k // 2, reverse)
    d = k // 2
    for i in range(d):
        a = start + i
        b = start + i + d
        if reverse:
            a, b = b, a
        comps.append((a, b))

def _bitonic_sort(n, comps):
    k = 2
    while k <= n:
        for i in range(0, n, k):
            _bitonic_merge(n, comps, i, k, False)
        k *= 2

def _build_bitonic():
    comps = []
    _bitonic_sort(N, comps)
    return comps

def _precompute_targets():
    targets = np.zeros((N, 1 << N), dtype=np.uint8)
    popcount = np.zeros(1 << N, dtype=np.int32)
    for m in range(1, 1 << N):
        popcount[m] = popcount[m >> 1] + (m & 1)
    for i in range(N):
        threshold = N - i
        targets[i] = (popcount >= threshold).astype(np.uint8)
    return targets

def _precompute_initials():
    initials = np.zeros((N, 1 << N), dtype=np.uint8)
    for i in range(N):
        for m in range(1 << N):
            if (m >> i) & 1:
                initials[i, m] = 1
    return initials

def _verify_bitparallel(comps, targets, initials):
    states = initials.copy()
    for i, j in comps:
        si = states[i]
        sj = states[j]
        states[i] = np.minimum(si, sj)
        states[j] = np.maximum(si, sj)
    for i in range(N):
        if not np.array_equal(states[i], targets[i]):
            return False
    return True

def _verify_with_mismatch(comps, targets, initials):
    states = initials.copy()
    for i, j in comps:
        si = states[i]
        sj = states[j]
        states[i] = np.minimum(si, sj)
        states[j] = np.maximum(si, sj)
    total = 0
    for i in range(N):
        total += int(np.sum(states[i] != targets[i]))
    return total

def _prune(comps, targets, initials, deadline):
    comps = list(comps)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(len(comps) - 1, -1, -1):
            if time.time() > deadline:
                return comps
            candidate = comps[:idx] + comps[idx + 1:]
            if _verify_bitparallel(candidate, targets, initials):
                comps = candidate
                improved = True
                break
        if improved:
            # Continue from beginning after a deletion
            pass
    return comps

def construct():
    deadline = time.time() + 80  # 10s safety margin
    rng = random.Random(12345)
    
    targets = _precompute_targets()
    initials = _precompute_initials()
    
    # Start with bitonic sort
    best = _build_bitonic()
    best_size = len(best)
    assert _verify_bitparallel(best, targets, initials), "Bitonic sort must be valid"
    
    # Initial pruning
    best = _prune(best, targets, initials, deadline)
    best_size = len(best)
    
    # Simulated annealing
    temp = 50.0
    cooling = 0.9995
    min_temp = 0.1
    
    while time.time() < deadline and temp > min_temp:
        # Generate a random move
        move_type = rng.random()
        candidate = list(best)
        
        if move_type < 0.35:
            # Delete a random comparator
            if len(candidate) > 1:
                idx = rng.randint(0, len(candidate) - 1)
                candidate.pop(idx)
        elif move_type < 0.55:
            # Insert a random comparator at random position
            i = rng.randint(0, N - 2)
            j = rng.randint(i + 1, N - 1)
            pos = rng.randint(0, len(candidate))
            candidate.insert(pos, (i, j))
        elif move_type < 0.75:
            # Swap two comparators
            if len(candidate) > 1:
                i, j = rng.sample(range(len(candidate)), 2)
                candidate[i], candidate[j] = candidate[j], candidate[i]
        elif move_type < 0.9:
            # Move a comparator to a new position
            if len(candidate) > 1:
                idx = rng.randint(0, len(candidate) - 1)
                comp = candidate.pop(idx)
                new_pos = rng.randint(0, len(candidate))
                candidate.insert(new_pos, comp)
        else:
            # Delete and insert (replacement)
            if len(candidate) > 1:
                idx = rng.randint(0, len(candidate) - 1)
                candidate.pop(idx)
                i = rng.randint(0, N - 2)
                j = rng.randint(i + 1, N - 1)
                pos = rng.randint(0, len(candidate))
                candidate.insert(pos, (i, j))
        
        # Evaluate candidate
        if len(candidate) < best_size:
            # Fewer comparators - must be valid
            if _verify_bitparallel(candidate, targets, initials):
                best = candidate
                best_size = len(best)
                # Re-prune after improvement
                if time.time() < deadline:
                    best = _prune(best, targets, initials, deadline)
                    best_size = len(best)
        else:
            # Same or more comparators - check validity
            if _verify_bitparallel(candidate, targets, initials):
                # Valid but not smaller; accept with some probability
                # (helps escape local minima)
                if rng.random() < 0.1:
                    best = candidate
        
        temp *= cooling
    
    # Final pruning
    if time.time() < deadline:
        best = _prune(best, targets, initials, deadline)
    
    return best

# EVOLVE-BLOCK-END
