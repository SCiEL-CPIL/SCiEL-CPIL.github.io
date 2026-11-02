import time
import random
import math

def _precompute_neighbors(q, n, radius):
    total = q ** n
    # For radius 1, neighbors are the word itself and words differing in 1 position
    neigh = [[] for _ in range(total)]
    neigh_set = [set() for _ in range(total)]
    for idx in range(total):
        neigh[idx].append(idx)
        neigh_set[idx].add(idx)
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
    
    # 2. Build distance-2 graph for independent set approach
    # Two codewords at Hamming distance 2 can be "merged" (replaced by a word
    # that covers both if possible), so we want an independent set in this graph
    dist2_adj = [set() for _ in range(total)]
    for idx in range(total):
        # Find all words at Hamming distance 2 from idx
        for i in range(n):
            pow_q_i = q ** i
            digit_i = (idx // pow_q_i) % q
            for j in range(i + 1, n):
                pow_q_j = q ** j
                digit_j = (idx // pow_q_j) % q
                for new_i in range(q):
                    if new_i == digit_i:
                        continue
                    for new_j in range(q):
                        if new_j == digit_j:
                            continue
                        new_idx = idx + (new_i - digit_i) * pow_q_i + (new_j - digit_j) * pow_q_j
                        if new_idx != idx:
                            dist2_adj[idx].add(new_idx)
                            dist2_adj[new_idx].add(idx)
    
    # 3. Greedy independent set (maximal, not maximum) for initial cover
    # This gives a starting point that avoids distance-2 pairs
    in_set = [False] * total
    available = [True] * total
    code = []
    
    # Greedy: repeatedly pick vertex with minimum degree in remaining graph
    remaining = set(range(total))
    while remaining:
        if time.monotonic() > deadline - 30.0:
            break
        # Pick vertex with minimum degree among remaining
        best_v = min(remaining, key=lambda v: len(dist2_adj[v] & remaining))
        code.append(best_v)
        in_set[best_v] = True
        available[best_v] = False
        remaining.discard(best_v)
        # Remove neighbors from remaining
        for u in dist2_adj[best_v]:
            if u in remaining:
                remaining.discard(u)
    
    # Check if this is a valid cover (it should be, but verify)
    counts = [0] * total
    remaining_covered = [True] * total
    rem_count = total
    
    for c in code:
        for u in neigh[c]:
            counts[u] += 1
            if counts[u] == 1:
                remaining_covered[u] = False
                rem_count -= 1
    
    # If not fully covered, add words greedily
    if rem_count > 0:
        while rem_count > 0:
            if time.monotonic() > deadline - 10.0:
                break
            best_c = -1
            best_gain = -1
            for c in range(total):
                if c in code:
                    continue
                gain = 0
                for u in neigh[c]:
                    if remaining_covered[u]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            if best_c == -1:
                break
            code.append(best_c)
            for u in neigh[best_c]:
                counts[u] += 1
                if counts[u] == 1:
                    remaining_covered[u] = False
                    rem_count -= 1
    
    if rem_count > 0:
        # Fallback
        while rem_count > 0:
            c = rng.randint(0, total - 1)
            if c not in code:
                code.append(c)
                for u in neigh[c]:
                    counts[u] += 1
                    if counts[u] == 1:
                        remaining_covered[u] = False
                        rem_count -= 1
            else:
                for u in range(total):
                    if remaining_covered[u]:
                        nc = neigh[u][0]
                        if nc not in code:
                            code.append(nc)
                            for u2 in neigh[nc]:
                                counts[u2] += 1
                                if counts[u2] == 1:
                                    remaining_covered[u2] = False
                                    rem_count -= 1
                            break
    
    best_code = code[:]
    best_size = len(code)
    
    # 4. Local search: remove redundant words, swap to improve
    temp = 1.0
    cool_rate = 0.9995
    min_temp = 0.02
    
    def is_removable(c):
        for u in neigh[c]:
            if counts[u] == 1:
                return False
        return True

    def is_valid_swap(c, new_c):
        for u in neigh[c]:
            if counts[u] == 1 and u not in neigh_set[new_c]:
                return False
        return True

    while time.monotonic() < deadline:
        if temp < min_temp:
            temp = min_temp
            
        move_type = rng.random()
        
        if move_type < 0.6:
            # Try Remove
            if len(code) > 0:
                idx = rng.randint(0, len(code) - 1)
                c = code[idx]
                if is_removable(c):
                    code.pop(idx)
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
                new_c = rng.randint(0, total - 1)
                
                if new_c == old_c or new_c in code:
                    continue
                    
                if is_valid_swap(old_c, new_c):
                    for u in neigh[old_c]:
                        counts[u] -= 1
                    for u in neigh[new_c]:
                        counts[u] += 1
                    code[idx] = new_c
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = code[:]

        else:
            # Try Add (Perturbation)
            new_c = rng.randint(0, total - 1)
            if new_c not in code:
                if rng.random() < math.exp(-1.0 / temp):
                    code.append(new_c)
                    for u in neigh[new_c]:
                        counts[u] += 1
                    
        temp *= cool_rate

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
