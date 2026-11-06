import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 4096
    MAX_SUM = 3 * N
    best_size = 0
    best_set = []
    
    triu_indices_cache = {}
    def get_triu_indices(k):
        if k not in triu_indices_cache:
            triu_indices_cache[k] = np.triu_indices(k)
        return triu_indices_cache[k]

    def try_add(S_np, x, sum_flag):
        """
        S_np: numpy array of current set
        x: candidate
        sum_flag: boolean numpy array
        Returns True if x can be added, and updates sum_flag.
        """
        k = len(S_np)
        
        # 3x
        s3 = 3 * x
        if s3 > MAX_SUM or sum_flag[s3]:
            return False
            
        if k > 0:
            # 2x + S
            sums_2x_s = 2 * x + S_np
            if np.any(sum_flag[sums_2x_s]):
                return False
                
            # x + S_i + S_j
            if k > 1:
                idx_i, idx_j = get_triu_indices(k)
                pair_sums = S_np[idx_i] + S_np[idx_j]
                sums_x_s_s = x + pair_sums
                if np.any(sum_flag[sums_x_s_s]):
                    return False
        
        # Valid
        sum_flag[s3] = True
        if k > 0:
            sum_flag[sums_2x_s] = True
            if k > 1:
                sum_flag[sums_x_s_s] = True
                
        return True

    for restart in range(2000):
        if time.time() > deadline:
            break
        
        r = restart % 3
        if r == 0:
            candidates = list(range(1, N + 1))
        elif r == 1:
            candidates = list(range(N, 0, -1))
        else:
            candidates = list(range(1, N + 1))
            random.shuffle(candidates)
            
        S = []
        S_np = np.empty(0, dtype=np.int32)
        sum_flag = np.zeros(MAX_SUM + 1, dtype=bool)
        
        for x in candidates:
            if time.time() > deadline:
                break
            
            if try_add(S_np, x, sum_flag):
                S.append(x)
                S_np = np.append(S_np, x)
                
                if len(S) > best_size:
                    best_size = len(S)
                    best_set = S.copy()
                    if best_size == 80:
                        return best_set
                        
    return best_set
