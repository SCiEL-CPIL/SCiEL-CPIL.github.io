import time
import random
import math
import itertools

def _precompute_neighbors(q, n, radius):
    total = q ** n
    neigh = [[] for _ in range(total)]
    neigh_set = [set() for _ in range(total)]
    for idx in range(total):
        neigh[idx].append(idx)
        neigh_set[idx].add(idx)
        if radius >= 1:
            for i in range(n):
                pow_q_i = q ** i
                digit = (idx // pow_q_i) % q
                for new_digit in range(q):
                    if new_digit != digit:
                        new_idx = idx + (new_digit - digit) * pow_q_i
                        neigh[idx].append(new_idx)
                        neigh_set[idx].add(new_idx)
    return neigh, neigh_set, total

def construct(q, n, radius):
    deadline = time.monotonic() + 115.0
    total = q ** n
    rng = random.Random(42)
    
    # 1. Precompute neighbors
    neigh, neigh_set, _ = _precompute_neighbors(q, n, radius)
    
    # 2. Greedy Set Cover Initialization
    uncovered = [True] * total
    rem_count = total
    code = []
    code_set = set()
    
    while rem_count > 0:
        if time.monotonic() > deadline - 60.0:
            break
        best_c = -1
        best_gain = 0
        for c in range(total):
            if c in code_set:
                continue
            gain = 0
            for u in neigh[c]:
                if uncovered[u]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_c = c
        if best_c == -1 or best_gain == 0:
            break
        code.append(best_c)
        code_set.add(best_c)
        for u in neigh[best_c]:
            if uncovered[u]:
                uncovered[u] = False
                rem_count -= 1
                
    # Ensure we have a valid cover
    if rem_count > 0:
        for u in range(total):
            if uncovered[u]:
                for c in neigh[u]:
                    if c not in code_set:
                        code.append(c)
                        code_set.add(c)
                        for u2 in neigh[c]:
                            if uncovered[u2]:
                                uncovered[u2] = False
                                rem_count -= 1
                        break
                if rem_count == 0:
                    break

    best_code = code[:]
    best_size = len(code)
    
    # 3. Simulated Annealing
    counts = [0] * total
    for c in code:
        for u in neigh[c]:
            counts[u] += 1
            
    temp = 1.0
    cool_rate = 0.9995
    min_temp = 0.01
    start_time = time.monotonic()
    
    def is_removable(c):
        for u in neigh[c]:
            if counts[u] == 1:
                return False
        return True

    def is_valid_swap(old_c, new_c):
        for u in neigh[old_c]:
            if counts[u] == 1 and u not in neigh_set[new_c]:
                return False
        return True

    while time.monotonic() < deadline:
        if temp < min_temp:
            temp = min_temp
            
        move_type = rng.random()
        
        if move_type < 0.5:
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
                        
        elif move_type < 0.8:
            # Try Swap
            if len(code) > 0:
                idx = rng.randint(0, len(code) - 1)
                old_c = code[idx]
                new_c = rng.randint(0, total - 1)
                
                if new_c == old_c or new_c in code_set:
                    continue
                    
                if is_valid_swap(old_c, new_c):
                    for u in neigh[old_c]:
                        counts[u] -= 1
                    for u in neigh[new_c]:
                        counts[u] += 1
                    code[idx] = new_c
                    code_set.discard(old_c)
                    code_set.add(new_c)
                    
        else:
            # Try Add (Perturbation)
            new_c = rng.randint(0, total - 1)
            if new_c not in code_set:
                if rng.random() < 0.1:
                    code.append(new_c)
                    code_set.add(new_c)
                    for u in neigh[new_c]:
                        counts[u] += 1
                    
        temp *= cool_rate

    # Final cleanup: try to remove redundant words from best_code
    best_set = set(best_code)
    best_counts = [0] * total
    for c in best_code:
        for u in neigh[c]:
            best_counts[u] += 1
            
    final_code = []
    for c in best_code:
        removable = True
        for u in neigh[c]:
            if best_counts[u] == 1:
                removable = False
                break
        if not removable:
            final_code.append(c)
            for u in neigh[c]:
                best_counts[u] -= 1
        else:
            pass
            
    if len(final_code) < len(best_code):
        best_code = final_code

    # Convert best_code indices to words
    result = []
    for idx in best_code:
        word = [0] * n
        temp_idx = idx
        for i in range(n):
            word[i] = temp_idx % q
            temp_idx //= q
        result.append(word)
        
    return result
