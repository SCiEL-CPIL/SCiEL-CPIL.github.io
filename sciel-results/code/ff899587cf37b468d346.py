import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    start_time = time.time()
    deadline = start_time + 590
    
    # 1. Compute forbidden differences
    forbidden = set()
    for x in range(1, m):
        r = (x * x) % m
        if r != 0:
            forbidden.add(r)
    
    # 2. Precompute conflict masks
    masks = [0] * m
    for u in range(m):
        mask_val = 0
        for v in range(m):
            if u == v:
                continue
            d1 = (u - v) % m
            d2 = (v - u) % m
            if d1 in forbidden or d2 in forbidden:
                mask_val |= (1 << v)
        masks[u] = mask_val
        
    def popcount(x):
        return bin(x).count('1')
        
    def try_add(current, u):
        if current & masks[u]:
            return False
        return True
        
    def greedy_init(rng):
        current = 0
        nodes = list(range(m))
        rng.shuffle(nodes)
        for u in nodes:
            if try_add(current, u):
                current |= (1 << u)
        return current
        
    def local_search(current, max_time):
        improved = True
        while improved and time.time() < max_time:
            improved = False
            current_set = current
            
            # 1. Try adding any node
            for u in range(m):
                if not (current_set >> u) & 1:
                    if try_add(current_set, u):
                        current_set |= (1 << u)
                        improved = True
            if improved:
                current = current_set
                continue
                
            # 2. Try 1-swap: remove u, add v
            for u in range(m):
                if not (current_set >> u) & 1:
                    continue
                temp_set = current_set & ~(1 << u)
                for v in range(m):
                    if (current_set >> v) & 1:
                        continue
                    if try_add(temp_set, v):
                        current_set = temp_set | (1 << v)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                current = current_set
                continue

            # 3. Try 2-swap: remove u, add v, w
            # To avoid O(N^3) explosion, we limit the search for v, w
            # We only try if the set is "dense" enough or randomly
            if random.random() < 0.5:
                for u in range(m):
                    if not (current_set >> u) & 1:
                        continue
                    temp_set = current_set & ~(1 << u)
                    # Find candidates v that can be added to temp_set
                    # We want to find two such candidates that are compatible with each other
                    # Let C be the set of candidates
                    # We iterate v in C, then w in C (w > v) and check compatibility
                    # To keep it fast, we only check a subset of C if it's large
                    candidates = []
                    for v in range(m):
                        if (current_set >> v) & 1:
                            continue
                        if try_add(temp_set, v):
                            candidates.append(v)
                            
                    if len(candidates) < 2:
                        continue
                    
                    # If candidates is large, sample it
                    if len(candidates) > 30:
                        candidates = random.sample(candidates, 30)
                        
                    found = False
                    for i in range(len(candidates)):
                        v = candidates[i]
                        for j in range(i+1, len(candidates)):
                            w = candidates[j]
                            # Check if v and w are compatible
                            if try_add(temp_set | (1 << v), w):
                                current_set = temp_set | (1 << v) | (1 << w)
                                improved = True
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
            if improved:
                current = current_set
                continue
                
            # 4. Random kick: remove 2 random, re-greedy
            if random.random() < 0.1:
                # Remove 2 random elements
                to_remove = random.sample([i for i in range(m) if (current_set >> i) & 1], min(2, popcount(current_set)))
                for u in to_remove:
                    current_set &= ~(1 << u)
                # Re-greedy
                nodes = list(range(m))
                random.shuffle(nodes)
                for u in nodes:
                    if try_add(current_set, u):
                        current_set |= (1 << u)
                if popcount(current_set) > popcount(current):
                    current = current_set
                    improved = True
        return current

    best = 0
    best_size = 0
    
    # Multiple random restarts
    for i in range(200):
        if time.time() > deadline - 10:
            break
        rng = random.Random(i)
        curr = greedy_init(rng)
        curr = local_search(curr, deadline - 5)
        sz = popcount(curr)
        if sz > best_size:
            best_size = sz
            best = curr
            
    # Extract result
    result = []
    for i in range(m):
        if (best >> i) & 1:
            result.append(i)
            
    # Final verification
    valid = True
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            d1 = (result[i] - result[j]) % m
            d2 = (result[j] - result[i]) % m
            if d1 in forbidden or d2 in forbidden:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        result = [0]
        
    return np.array(result, dtype=np.int64)
