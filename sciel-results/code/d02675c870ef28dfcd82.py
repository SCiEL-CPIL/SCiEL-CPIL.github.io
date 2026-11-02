import time
import random
import math
import numpy as np

def _precompute_neighbors(q, n, radius):
    total = q ** n
    neigh = [[] for _ in range(total)]
    for idx in range(total):
        neigh[idx].append(idx)
        for i in range(n):
            pow_q_i = q ** i
            digit = (idx // pow_q_i) % q
            for new_digit in range(q):
                if new_digit != digit:
                    new_idx = idx + (new_digit - digit) * pow_q_i
                    neigh[idx].append(new_idx)
    return neigh, total

def construct(q, n, radius):
    deadline = time.monotonic() + 110.0
    total = q ** n
    rng = random.Random(42)
    
    neigh, _ = _precompute_neighbors(q, n, radius)
    
    # Build ball matrix: ball_matrix[c, u] = 1 if u is in ball(c)
    ball_matrix = np.zeros((total, total), dtype=np.uint8)
    for c in range(total):
        for u in neigh[c]:
            ball_matrix[c, u] = 1
    
    def greedy_init(seed):
        """Vectorized greedy set cover."""
        r = random.Random(seed)
        counts = np.zeros(total, dtype=np.int32)
        code = []
        
        while True:
            if time.monotonic() > deadline:
                break
            remaining_mask = (counts == 0).astype(np.uint8)
            if remaining_mask.sum() == 0:
                break
            
            # Vectorized gain computation
            gains = ball_matrix @ remaining_mask
            best_gain = int(gains.max())
            if best_gain <= 0:
                break
            
            # Get all candidates with max gain
            candidates = np.where(gains == best_gain)[0]
            
            # Random tie-break among max-gain candidates
            if len(candidates) > 1:
                best_c = int(candidates[r.randint(0, len(candidates) - 1)])
            else:
                best_c = int(candidates[0])
            
            code.append(best_c)
            counts += ball_matrix[best_c]
        
        # Fallback if not fully covered
        remaining_mask = (counts == 0)
        if remaining_mask.any():
            code_set = set(code)
            while remaining_mask.any():
                uncovered = np.where(remaining_mask)[0]
                u = int(uncovered[0])
                # Find a codeword that covers u
                covering = np.where(ball_matrix[:, u] == 1)[0]
                for nc in covering:
                    nc = int(nc)
                    if nc not in code_set:
                        code.append(nc)
                        code_set.add(nc)
                        counts += ball_matrix[nc]
                        remaining_mask = (counts == 0)
                        break
        
        return code, counts

    def polish(code, counts, time_budget):
        """Local search to minimize code size using numpy."""
        code_set = set(code)
        best_code = code[:]
        best_size = len(code)
        end_time = time.monotonic() + time_budget
        
        def is_removable(c):
            # Check if any word in ball(c) has count == 1
            ball_words = neigh[c]
            return all(counts[u] > 1 for u in ball_words)
        
        # First pass: remove all removable codewords
        changed = True
        while changed and time.monotonic() < end_time:
            changed = False
            for i in range(len(code) - 1, -1, -1):
                c = code[i]
                if is_removable(c):
                    code.pop(i)
                    code_set.discard(c)
                    for u in neigh[c]:
                        counts[u] -= 1
                    changed = True
            if len(code) < best_size:
                best_size = len(code)
                best_code = code[:]
        
        # SA phase
        temp = 3.0
        cool_rate = 0.999
        min_temp = 0.05
        
        while time.monotonic() < end_time:
            if temp < min_temp:
                temp = min_temp
            
            move_type = rng.random()
            
            if move_type < 0.35:
                # Try Remove
                if len(code) > 0:
                    idx = rng.randint(0, len(code) - 1)
                    c = code[idx]
                    if is_removable(c):
                        code.pop(idx)
                        code_set.discard(c)
                        for u in neigh[c]:
                            counts[u] -= 1
                        if len(code) < best_size:
                            best_size = len(code)
                            best_code = code[:]
                        
            elif move_type < 0.85:
                # Try Swap
                if len(code) > 0:
                    idx = rng.randint(0, len(code) - 1)
                    old_c = code[idx]
                    if rng.random() < 0.7:
                        new_c = neigh[old_c][rng.randint(1, len(neigh[old_c])-1)]
                    else:
                        new_c = rng.randint(0, total - 1)
                    
                    if new_c == old_c or new_c in code_set:
                        continue
                    
                    # Check validity: no word in ball(old_c) with count==1 that's not in ball(new_c)
                    old_ball = set(neigh[old_c])
                    new_ball = set(neigh[new_c])
                    uncovered = 0
                    for u in old_ball:
                        if counts[u] == 1 and u not in new_ball:
                            uncovered += 1
                    
                    if uncovered == 0:
                        for u in neigh[old_c]:
                            counts[u] -= 1
                        for u in neigh[new_c]:
                            counts[u] += 1
                        code[idx] = new_c
                        code_set.discard(old_c)
                        code_set.add(new_c)
                        if len(code) < best_size:
                            best_size = len(code)
                            best_code = code[:]
                    else:
                        if uncovered <= 2 and rng.random() < math.exp(-uncovered / temp):
                            for u in neigh[old_c]:
                                counts[u] -= 1
                            for u in neigh[new_c]:
                                counts[u] += 1
                            code[idx] = new_c
                            code_set.discard(old_c)
                            code_set.add(new_c)
                            # Fix uncovered words
                            for u in neigh[old_c]:
                                if counts[u] == 0:
                                    covering = [nc for nc in neigh[u] if nc not in code_set]
                                    if covering:
                                        nc = covering[rng.randint(0, len(covering)-1)]
                                        code.append(nc)
                                        code_set.add(nc)
                                        for v in neigh[nc]:
                                            counts[v] += 1
                            
            else:
                # Try Add (perturbation)
                new_c = rng.randint(0, total - 1)
                if new_c not in code_set:
                    if rng.random() < math.exp(-1.0 / temp):
                        code.append(new_c)
                        code_set.add(new_c)
                        for u in neigh[new_c]:
                            counts[u] += 1
            
            temp *= cool_rate
        
        # Final removal pass
        changed = True
        while changed and time.monotonic() < end_time:
            changed = False
            for i in range(len(code) - 1, -1, -1):
                c = code[i]
                if is_removable(c):
                    code.pop(i)
                    code_set.discard(c)
                    for u in neigh[c]:
                        counts[u] -= 1
                    changed = True
            if len(code) < best_size:
                best_size = len(code)
                best_code = code[:]
        
        return best_code, best_size

    # Multi-restart strategy with more restarts due to faster greedy
    overall_best_code = None
    overall_best_size = total + 1
    
    num_restarts = 20
    time_per_restart = (deadline - time.monotonic()) / num_restarts
    
    for restart in range(num_restarts):
        if time.monotonic() > deadline - 5:
            break
        
        # Greedy initialization with different seeds
        code, counts = greedy_init(42 + restart * 7)
        
        if not code:
            continue
        
        # Polish
        polished_code, polished_size = polish(code, counts, time_per_restart - 0.3)
        
        if polished_size < overall_best_size:
            overall_best_size = polished_size
            overall_best_code = polished_code[:]
    
    # If we haven't found anything, do one more greedy
    if overall_best_code is None:
        code, counts = greedy_init(42)
        overall_best_code = code
        overall_best_size = len(code)
    
    # Convert to words
    result = []
    for idx in overall_best_code:
        word = [0] * n
        temp_idx = idx
        for i in range(n):
            word[i] = temp_idx % q
            temp_idx //= q
        result.append(word)
    
    return result

# Precompute neighbor sets at module level for reuse across calls
_neigh_sets_cache = {}

def _get_neigh_sets(q, n, radius):
    key = (q, n, radius)
    if key not in _neigh_sets_cache:
        total = q ** n
        neigh, _ = _precompute_neighbors(q, n, radius)
        _neigh_sets_cache[key] = [set(neigh[i]) for i in range(total)]
    return _neigh_sets_cache[key]
