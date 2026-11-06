import numpy as np
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 55.0

    max_val = 4096
    max_sum = 3 * max_val + 1
    
    # Precompute triu indices for sizes up to 40
    triu_cache = {}
    for k in range(41):
        if k > 1:
            triu_cache[k] = np.triu_indices(k)
        else:
            triu_cache[k] = (np.array([], dtype=np.int32), np.array([], dtype=np.int32))

    def is_valid_candidate(x, S, used_sums):
        k = len(S)
        if k == 0:
            return True
        
        # 1. 3x
        if used_sums[3 * x]:
            return False
        
        # 2. S + 2x
        sums_2 = S + 2 * x
        if np.any(used_sums[sums_2]):
            return False
        
        # 3. x + S[i] + S[j]
        if k > 1:
            i_idx, j_idx = triu_cache[k]
            sums_3 = x + S[i_idx] + S[j_idx]
            if np.any(used_sums[sums_3]):
                return False
        else:
            # k=1
            if used_sums[x + 2 * S[0]]:
                return False

        return True

    def add_element(x, S, used_sums):
        k = len(S)
        if k == 0:
            S_new = np.array([x], dtype=np.int32)
            used_sums[3 * x] = True
            return S_new
        
        used_sums[3 * x] = True
        sums_2 = S + 2 * x
        used_sums[sums_2] = True
        
        if k > 1:
            i_idx, j_idx = triu_cache[k]
            sums_3 = x + S[i_idx] + S[j_idx]
            used_sums[sums_3] = True
        else:
            used_sums[x + 2 * S[0]] = True
            
        S_new = np.append(S, x)
        return S_new

    def remove_element(x, S, used_sums):
        k = len(S)
        idx = np.where(S == x)[0][0]
        S_old = np.delete(S, idx)
        
        used_sums[3 * x] = False
        if len(S_old) > 0:
            sums_2 = S_old + 2 * x
            used_sums[sums_2] = False
            
            if len(S_old) > 1:
                i_idx, j_idx = triu_cache[len(S_old)]
                sums_3 = x + S_old[i_idx] + S_old[j_idx]
                used_sums[sums_3] = False
            else:
                used_sums[x + 2 * S_old[0]] = False
        return S_old

    best_set = []
    best_len = 0

    # Phase 1: Greedy
    for seed in range(10):
        if time.time() - start_time > DEADLINE:
            break
            
        random.seed(seed + 100)
        candidates = list(range(1, max_val + 1))
        random.shuffle(candidates)
        
        used_sums_init = np.zeros(max_sum, dtype=bool)
        S_init = np.array([], dtype=np.int32)
        
        for x in candidates:
            if is_valid_candidate(x, S_init, used_sums_init):
                S_init = add_element(x, S_init, used_sums_init)
                
        if len(S_init) > best_len:
            best_len = len(S_init)
            best_set = S_init.tolist()
            
        if best_len >= 41:
            return best_set

    # Phase 2: Local Search
    S_ls = np.array(best_set, dtype=np.int32)
    used_sums_ls = np.zeros(max_sum, dtype=bool)
    S_rebuild = np.array([], dtype=np.int32)
    for x in S_ls:
        S_rebuild = add_element(x, S_rebuild, used_sums_ls)
    S_ls = S_rebuild
    
    current_len = len(S_ls)
    no_improve_count = 0
    
    while time.time() - start_time < DEADLINE:
        # Try Add
        x_new = random.randint(1, max_val)
        if x_new not in S_ls:
            if is_valid_candidate(x_new, S_ls, used_sums_ls):
                S_ls = add_element(x_new, S_ls, used_sums_ls)
                current_len += 1
                no_improve_count = 0
                if current_len > best_len:
                    best_len = current_len
                    best_set = S_ls.tolist()
                    if best_len >= 41:
                        return best_set
                continue
        
        # Try Swap
        if len(S_ls) > 0:
            u = random.choice(S_ls.tolist())
            v = random.randint(1, max_val)
            if v != u and v not in S_ls:
                S_temp = remove_element(u, S_ls, used_sums_ls)
                if is_valid_candidate(v, S_temp, used_sums_ls):
                    S_ls = add_element(v, S_temp, used_sums_ls)
                    no_improve_count = 0
                    continue
        
        # Try 2-Swap
        if len(S_ls) >= 2:
            u1, u2 = random.sample(S_ls.tolist(), 2)
            # Get v1, v2
            while True:
                v1 = random.randint(1, max_val)
                v2 = random.randint(1, max_val)
                if v1 != v2 and v1 not in S_ls and v2 not in S_ls:
                    break
            
            S_temp = remove_element(u1, S_ls, used_sums_ls)
            S_temp = remove_element(u2, S_temp, used_sums_ls)
            
            if is_valid_candidate(v1, S_temp, used_sums_ls):
                S_temp = add_element(v1, S_temp, used_sums_ls)
                if is_valid_candidate(v2, S_temp, used_sums_ls):
                    S_ls = add_element(v2, S_temp, used_sums_ls)
                    no_improve_count = 0
                    continue
        
        no_improve_count += 1
        if no_improve_count > 5000:
            no_improve_count = 0

    return best_set
