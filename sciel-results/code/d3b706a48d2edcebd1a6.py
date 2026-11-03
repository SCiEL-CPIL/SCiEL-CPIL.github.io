import time
import random
import math

def construct(q, n, radius):
    start_time = time.monotonic()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    total = q ** n
    rng = random.Random(42)
    
    # Precompute neighbors (words within Hamming distance radius)
    # For radius 1, this is 1 + n*(q-1)
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
    
    # Convert neighbors to sets for O(1) membership check if needed, 
    # but for counting, lists are faster. We'll use lists for iteration.
    
    def greedy_init(seed):
        """Fast greedy set cover initialization."""
        r = random.Random(seed)
        counts = [0] * total
        code = []
        code_set = set()
        remaining = [True] * total
        rem_count = total
        
        while rem_count > 0:
            if time.monotonic() > deadline - 5.0:
                break
            
            best_c = -1
            best_gain = -1
            
            # Find candidate with max new coverage
            for c in range(total):
                if c in code_set:
                    continue
                gain = 0
                for u in neigh[c]:
                    if remaining[u]:
                        gain += 1
                        if gain > best_gain:
                            # Early exit if we can't beat best? 
                            # Max possible gain is len(neigh[c])
                            # But we need to check all if we don't know max.
                            # Actually, just compute full gain for simplicity in Python?
                            # No, this is O(N * K). N=729, K=16. 729*16*729 ~ 8M ops.
                            # This is acceptable for one pass.
                            pass
                
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            
            if best_c == -1 or best_gain == 0:
                # Fallback: pick any uncovered word's neighbor
                # This shouldn't happen if rem_count > 0 and we have a valid cover
                # But just in case
                for u in range(total):
                    if remaining[u]:
                        for nc in neigh[u]:
                            if nc not in code_set:
                                best_c = nc
                                break
                        if best_c != -1:
                            break
                if best_c == -1:
                    break
            
            code.append(best_c)
            code_set.add(best_c)
            
            for u in neigh[best_c]:
                counts[u] += 1
                if counts[u] == 1:
                    remaining[u] = False
                    rem_count -= 1
        
        return code, counts

    def polish(code, counts, time_budget):
        """Local search to minimize code size."""
        code_set = set(code)
        best_code = code[:]
        best_size = len(code)
        end_time = time.monotonic() + time_budget
        
        def is_removable(c):
            for u in neigh[c]:
                if counts[u] == 1:
                    return False
            return True
        
        # First pass: remove all removable codewords
        changed = True
        while changed and time.monotonic() < end_time:
            changed = False
            for i in range(len(code) - 1, -1, -1):
                if is_removable(code[i]):
                    c = code.pop(i)
                    code_set.discard(c)
                    for u in neigh[c]:
                        counts[u] -= 1
                    changed = True
            if len(code) < best_size:
                best_size = len(code)
                best_code = code[:]
        
        # Simulated Annealing
        temp = 2.0
        cool_rate = 0.9995
        min_temp = 0.01
        
        # Precompute neighbor sets for faster swap validation?
        # neigh_sets = [set(neigh[i]) for i in range(total)]
        # This might be memory heavy but 729 * 16 is small.
        neigh_sets = [set(neigh[i]) for i in range(total)]
        
        while time.monotonic() < end_time:
            if temp < min_temp:
                temp = min_temp
            
            move_type = rng.random()
            
            if move_type < 0.4:
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
                            
            elif move_type < 0.9:
                # Try Swap
                if len(code) > 0:
                    idx = rng.randint(0, len(code) - 1)
                    old_c = code[idx]
                    # Choose new_c from neighbors of old_c to maintain locality
                    if rng.random() < 0.8:
                        new_c = neigh[old_c][rng.randint(1, len(neigh[old_c])-1)]
                    else:
                        new_c = rng.randint(0, total - 1)
                    
                    if new_c == old_c or new_c in code_set:
                        continue
                    
                    # Check validity: Does removing old_c and adding new_c leave any word uncovered?
                    # A word u is uncovered if counts[u] == 1 (only covered by old_c) and u not in neigh[new_c]
                    valid = True
                    new_c_set = neigh_sets[new_c]
                    for u in neigh[old_c]:
                        if counts[u] == 1 and u not in new_c_set:
                            valid = False
                            break
                    
                    if valid:
                        # Accept
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
                        # SA acceptance for invalid swap
                        # Count uncovered
                        uncovered = 0
                        new_c_set = neigh_sets[new_c]
                        for u in neigh[old_c]:
                            if counts[u] == 1 and u not in new_c_set:
                                uncovered += 1
                        if uncovered <= 3 and rng.random() < math.exp(-uncovered / temp):
                            # Accept perturbation
                            for u in neigh[old_c]:
                                counts[u] -= 1
                            for u in neigh[new_c]:
                                counts[u] += 1
                            code[idx] = new_c
                            code_set.discard(old_c)
                            code_set.add(new_c)
                            
                            # Try to fix uncovered words immediately
                            for u in neigh[old_c]:
                                if counts[u] == 0:
                                    # Find a codeword that covers u
                                    # Prefer one that covers many other uncovered words?
                                    # Just pick the first available neighbor
                                    for nc in neigh[u]:
                                        if nc not in code_set:
                                            code.append(nc)
                                            code_set.add(nc)
                                            for v in neigh[nc]:
                                                counts[v] += 1
                                            break
                                    
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
                if is_removable(code[i]):
                    c = code.pop(i)
                    code_set.discard(c)
                    for u in neigh[c]:
                        counts[u] -= 1
                    changed = True
            if len(code) < best_size:
                best_size = len(code)
                best_code = code[:]
        
        return best_code, best_size

    # Multi-restart strategy
    overall_best_code = None
    overall_best_size = total + 1
    
    # Determine number of restarts
    # Each restart should take ~5-10s
    num_restarts = 10
    
    for restart in range(num_restarts):
        if time.monotonic() > deadline - 10.0:
            break
        
        # Use different seeds
        seed = 42 + restart * 13
        code, counts = greedy_init(seed)
        
        if not code:
            continue
        
        # Polish
        # Give more time to earlier restarts? Or equal?
        # Let's give equal time, but ensure we don't exceed deadline
        remaining_time = deadline - time.monotonic()
        if remaining_time < 10.0:
            break
            
        time_per_polish = min(10.0, (remaining_time - 2.0) / (num_restarts - restart))
        
        polished_code, polished_size = polish(code, counts, time_per_polish)
        
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
