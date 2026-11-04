"""
Optimized ternary covering code construction using Greedy + Local Search.
Targets K_3(6,1).
"""
import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin before 120s limit
    
    if q != 3 or n != 6 or radius != 1:
        return _greedy_construct(q, n, radius, deadline - (time.time() - start_time), start_time)
    
    return _construct_k3_6_1(deadline - (time.time() - start_time), start_time)

def _greedy_construct(q, n, radius, time_limit, start_time):
    total_words = q ** n
    if total_words > 100000:
        return []
        
    neighbors = []
    for i in range(total_words):
        w = np.array([(i // (q ** (n - 1 - pos))) % q for pos in range(n)])
        nbrs = [i]
        if radius == 1:
            for pos in range(n):
                for new_val in range(q):
                    if new_val != w[pos]:
                        new_w = w.copy()
                        new_w[pos] = new_val
                        idx = 0
                        for j in range(n):
                            idx = idx * q + new_w[j]
                        nbrs.append(idx)
        neighbors.append(set(nbrs))
        
    covered = np.zeros(total_words, dtype=bool)
    code = []
    
    while time.time() < start_time + time_limit:
        gains = np.zeros(total_words)
        for i in range(total_words):
            gains[i] = np.sum(~covered[list(neighbors[i])])
        
        best_idx = np.argmax(gains)
        if gains[best_idx] == 0:
            break
            
        code.append(best_idx)
        covered[list(neighbors[best_idx])] = True
        
    result = []
    for idx in code:
        w = [(idx // (q ** (n - 1 - pos))) % q for pos in range(n)]
        result.append(w)
    return result

def _construct_k3_6_1(time_limit, start_time):
    n = 6
    q = 3
    total_words = 729
    
    words = np.array([[ (i // (3 ** (5 - pos))) % 3 for pos in range(6) ] for i in range(total_words)])
    
    neighbor_lists = []
    for i in range(total_words):
        w = words[i]
        nbrs = [i]
        for pos in range(6):
            orig_val = w[pos]
            for new_val in range(3):
                if new_val != orig_val:
                    new_w = w.copy()
                    new_w[pos] = new_val
                    idx = 0
                    for j in range(6):
                        idx = idx * 3 + new_w[j]
                    nbrs.append(idx)
        neighbor_lists.append(nbrs)
        
    sphere = np.zeros((total_words, total_words), dtype=bool)
    for i in range(total_words):
        sphere[i, neighbor_lists[i]] = True
        
    # Precompute coverers for faster swap search
    # coverers[j] is a list of indices i such that j is in sphere[i]
    # This is equivalent to sphere.T
    # We can just use sphere.T directly in numpy operations.
    
    covered = np.zeros(total_words, dtype=bool)
    code_indices = []
    
    uncovered_count = total_words
    greedy_deadline = start_time + time_limit * 0.2 
    while uncovered_count > 0 and time.time() < greedy_deadline:
        gains = (sphere & (~covered)[np.newaxis, :]).sum(axis=1)
        best_idx = np.argmax(gains)
        best_gain = gains[best_idx]
        if best_gain == 0:
            break
        code_indices.append(best_idx)
        covered[sphere[best_idx]] = True
        uncovered_count -= best_gain
        
    current_code = code_indices
    best_code = current_code.copy()
    best_size = len(best_code)
    
    count = np.zeros(total_words, dtype=np.int32)
    for c in current_code:
        count[sphere[c]] += 1
        
    search_deadline = start_time + time_limit
    iterations = 0
    max_iterations = 100000 # Safety cap
    
    while time.time() < search_deadline and iterations < max_iterations:
        iterations += 1
        modified = False
        
        # Phase 1: Remove redundant nodes
        random.shuffle(current_code)
        i = 0
        while i < len(current_code):
            c = current_code[i]
            min_cnt = 999
            for j in neighbor_lists[c]:
                if count[j] < min_cnt:
                    min_cnt = count[j]
                    if min_cnt == 1:
                        break
            
            if min_cnt > 1:
                count[sphere[c]] -= 1
                current_code.pop(i)
                modified = True
            else:
                i += 1
                
        if len(current_code) < best_size:
            best_size = len(current_code)
            best_code = current_code.copy()
            
        # Phase 2: Swap to escape local minima
        if not modified:
            swap_found = False
            code_set = set(current_code)
            
            for c in current_code:
                W_crit = [j for j in neighbor_lists[c] if count[j] == 1]
                
                if not W_crit:
                    count[sphere[c]] -= 1
                    current_code.remove(c)
                    code_set.remove(c)
                    modified = True
                    swap_found = True
                    break
                    
                if W_crit:
                    w0 = W_crit[0]
                    # Get candidates covering w0
                    candidates = np.where(sphere.T[w0])[0]
                    
                    for c_prime in candidates:
                        if c_prime in code_set:
                            continue
                        if np.all(sphere[c_prime, W_crit]):
                            # Perform swap
                            count[sphere[c]] -= 1
                            current_code.remove(c)
                            code_set.remove(c)
                            
                            count[sphere[c_prime]] += 1
                            current_code.append(c_prime)
                            code_set.add(c_prime)
                            
                            modified = True
                            swap_found = True
                            break
                            
                    if swap_found:
                        break
                        
            if not swap_found and not modified:
                # Perturb: Add a random node
                rand_idx = random.randint(0, total_words - 1)
                if rand_idx not in code_set:
                    count[sphere[rand_idx]] += 1
                    current_code.append(rand_idx)
                    code_set.add(rand_idx)
                else:
                    # If stuck, maybe try a different random node or break
                    # To avoid infinite loop of adding same node, we rely on time limit
                    pass

    result = []
    for idx in best_code:
        w = words[idx].tolist()
        result.append(w)
    return result
