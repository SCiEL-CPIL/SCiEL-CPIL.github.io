# EVOLVE-BLOCK-START
import time
import random

def _max_discrepancy(signs):
    n = len(signs)
    max_abs = 0
    for d in range(1, n):
        for r in range(d):
            if r >= n:
                break
            s = 0
            idx = r
            while idx < n:
                s += signs[idx]
                a = s if s >= 0 else -s
                if a > max_abs:
                    max_abs = a
                idx += d
    return max_abs

def _find_worst_ap(signs):
    n = len(signs)
    worst_abs = 0
    worst_sum = 0
    worst_indices = []
    for d in range(1, n):
        for r in range(d):
            if r >= n:
                break
            s = 0
            idx = r
            indices = []
            while idx < n:
                s += signs[idx]
                indices.append(idx)
                idx += d
            a = s if s >= 0 else -s
            if a > worst_abs:
                worst_abs = a
                worst_sum = s
                worst_indices = indices
    return worst_abs, worst_sum, worst_indices

def _local_search(signs, max_time, rng):
    n = len(signs)
    signs = list(signs)
    best_val = _max_discrepancy(signs)
    deadline = time.time() + max_time
    stagnant = 0
    
    while time.time() < deadline:
        worst_abs, worst_sum, worst_indices = _find_worst_ap(signs)
        
        if worst_abs <= 1:
            break
        
        if worst_sum > 0:
            candidates = [i for i in worst_indices if signs[i] == 1]
        else:
            candidates = [i for i in worst_indices if signs[i] == -1]
        
        if not candidates:
            flip_idx = rng.randrange(n)
            signs[flip_idx] = -signs[flip_idx]
            stagnant += 1
            if stagnant > 200:
                break
            continue
        
        best_flip = None
        best_new_abs = worst_abs
        for i in candidates:
            new_sum = worst_sum - 2 * signs[i]
            new_abs = new_sum if new_sum >= 0 else -new_sum
            if new_abs < best_new_abs:
                best_new_abs = new_abs
                best_flip = i
        
        if best_flip is not None:
            signs[best_flip] = -signs[best_flip]
            stagnant = 0
        else:
            found = False
            for i in range(len(candidates)):
                for j in range(i+1, len(candidates)):
                    new_sum = worst_sum - 2*signs[candidates[i]] - 2*signs[candidates[j]]
                    new_abs = new_sum if new_sum >= 0 else -new_sum
                    if new_abs < best_new_abs:
                        signs[candidates[i]] = -signs[candidates[i]]
                        signs[candidates[j]] = -signs[candidates[j]]
                        found = True
                        break
                if found:
                    break
            if found:
                stagnant = 0
            else:
                flip_idx = rng.randrange(n)
                signs[flip_idx] = -signs[flip_idx]
                stagnant += 1
                if stagnant > 200:
                    break
        
        if stagnant == 0:
            val = _max_discrepancy(signs)
            if val < best_val:
                best_val = val
            stagnant = 0
        else:
            stagnant += 1
    
    return signs, _max_discrepancy(signs)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    rng = random.Random(171)
    n = 256
    
    best_signs = None
    best_val = float('inf')
    
    num_starts = 0
    while time.time() < deadline - 20.0:
        signs = [rng.choice((-1, 1)) for _ in range(n)]
        val = _max_discrepancy(signs)
        if val < best_val:
            best_val = val
            best_signs = signs
        num_starts += 1
        if num_starts >= 3:
            break
    
    if time.time() < deadline - 2.0:
        remaining = deadline - 2.0 - time.time()
        best_signs, best_val = _local_search(best_signs, remaining, rng)
    
    return best_signs
# EVOLVE-BLOCK-END
