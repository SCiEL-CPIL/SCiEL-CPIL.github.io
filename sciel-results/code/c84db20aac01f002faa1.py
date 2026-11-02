import time
import random
import math

def _precompute_neighbors(q, n, radius):
    total = q ** n
    # For radius 1, neighbors are the word itself and words differing in 1 position
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
    
    # 1. Precompute neighbors
    neigh, _ = _precompute_neighbors(q, n, radius)
    neigh_sets = _get_neigh_sets(q, n, radius)
    
    # 2. Greedy Initialization
    # We want to start with a reasonably good cover.
    # Greedy: pick word covering most uncovered words.
    counts = [0] * total
    code = []
    remaining = [True] * total
    rem_count = total
    
    def add_to_code(c):
        nonlocal rem_count
        code.append(c)
        for u in neigh[c]:
            counts[u] += 1
            if counts[u] == 1:
                remaining[u] = False
                rem_count -= 1

    def remove_from_code(c):
        nonlocal rem_count
        code.remove(c)
        for u in neigh[c]:
            counts[u] -= 1
            if counts[u] == 0:
                remaining[u] = True
                rem_count += 1

    # Greedy selection
    while rem_count > 0:
        if time.monotonic() > deadline:
            break
        best_c = -1
        best_gain = -1
        for c in range(total):
            gain = 0
            for u in neigh[c]:
                if remaining[u]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_c = c
        if best_c == -1:
            break
        add_to_code(best_c)
        
    if rem_count > 0:
        # Fallback
        while rem_count > 0:
            c = rng.randint(0, total - 1)
            if c not in code:
                add_to_code(c)
            else:
                for u in range(total):
                    if remaining[u]:
                        nc = neigh[u][0]
                        if nc not in code:
                            add_to_code(nc)
                            break

    best_code = code[:]
    best_size = len(code)
    
    # 3. Simulated Annealing / Local Search
    temp = 5.0
    cool_rate = 0.9995
    min_temp = 0.01
    
    # Helper to check if removing c is valid
    def is_removable(c):
        for u in neigh[c]:
            if counts[u] == 1:
                return False
        return True

    # Helper to check if swapping c with new_c is valid
    def is_valid_swap(c, new_c):
        # Words at risk: covered by c but not new_c, with count == 1
        new_c_set = neigh_sets[new_c]
        for u in neigh[c]:
            if counts[u] == 1 and u not in new_c_set:
                return False
        return True

    # Convert code to a set for fast membership in swap check
    code_set = set(code)
    
    while time.monotonic() < deadline:
        if temp < min_temp:
            temp = min_temp
            
        move_type = rng.random()
        
        if move_type < 0.45:
            # Try Remove
            if len(code) > 0:
                idx = rng.randint(0, len(code) - 1)
                c = code[idx]
                if is_removable(c):
                    # Accept removal
                    code.pop(idx)
                    code_set.discard(c)
                    for u in neigh[c]:
                        counts[u] -= 1
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = code[:]
                else:
                    pass
                    
        elif move_type < 0.90:
            # Try Swap
            if len(code) > 0:
                idx = rng.randint(0, len(code) - 1)
                old_c = code[idx]
                new_c = rng.randint(0, total - 1)
                
                if new_c == old_c or new_c in code_set:
                    continue
                    
                if is_valid_swap(old_c, new_c):
                    # Accept swap
                    for u in neigh[old_c]:
                        counts[u] -= 1
                    for u in neigh[new_c]:
                        counts[u] += 1
                    code[idx] = new_c
                    code_set.discard(old_c)
                    code_set.add(new_c)
                else:
                    pass

        else:
            # Try Add (Perturbation to escape local minima)
            new_c = rng.randint(0, total - 1)
            if new_c not in code_set:
                if rng.random() < math.exp(-1.0 / temp):
                    code.append(new_c)
                    code_set.add(new_c)
                    for u in neigh[new_c]:
                        counts[u] += 1
                    
        temp *= cool_rate
    
    # Final removal pass: try to remove any removable codewords
    changed = True
    while changed and time.monotonic() < deadline:
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

# Precompute neighbor sets at module level for reuse across calls
_neigh_sets_cache = {}

def _get_neigh_sets(q, n, radius):
    key = (q, n, radius)
    if key not in _neigh_sets_cache:
        total = q ** n
        neigh, _ = _precompute_neighbors(q, n, radius)
        _neigh_sets_cache[key] = [set(neigh[i]) for i in range(total)]
    return _neigh_sets_cache[key]
