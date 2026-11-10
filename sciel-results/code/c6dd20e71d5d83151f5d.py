# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def kissing_number11() -> np.ndarray:
    """
    Returns a 2-D array of shape (k, 11) with integer entries (int64) such that:
      - second dimension is 11
      - k >= 2
      - no point is the zero vector
      - min squared pairwise distance >= max squared norm
    
    Strategy:
    1. Generate pools of weight-4 and weight-5 vectors.
    2. Use minimum-degree greedy with degree updates:
       - At each step, pick the available vector with the fewest conflicts
       - Update degrees of neighbors after each selection
       - Add small random noise to break ties across restarts
    3. Try multiple restarts and select the best result.
    """
    start_time = time.time()
    deadline = start_time + 570
    d = 11
    
    def generate_pool(w):
        pool = []
        for idx in itertools.combinations(range(d), w):
            for signs in itertools.product([1, -1], repeat=w):
                v = np.zeros(d, dtype=np.int64)
                for i, s in zip(idx, signs):
                    v[i] = s
                pool.append(v)
        return np.array(pool)
    
    def greedy_min_degree(pool, time_limit, n_restarts=20):
        n = len(pool)
        if n == 0:
            return np.empty((0, d), dtype=np.int64)
        
        norms2 = np.sum(pool * pool, axis=1)
        D = pool @ pool.T
        
        min_norms = np.minimum.outer(norms2, norms2)
        max_dot_allowed = min_norms // 2
        conflict_matrix = D > max_dot_allowed
        
        best_set = []
        rng = np.random.RandomState(42)
        
        for restart in range(n_restarts):
            if time.time() > time_limit:
                break
            
            degree = np.sum(conflict_matrix, axis=1).astype(np.float64)
            degree += rng.uniform(0, 0.1, size=n)
            
            available = np.ones(n, dtype=bool)
            selected = []
            
            while True:
                if time.time() > time_limit:
                    break
                if not np.any(available):
                    break
                
                deg_avail = np.where(available, degree, np.finfo(np.float64).max)
                idx = int(np.argmin(deg_avail))
                
                if not available[idx]:
                    break
                
                selected.append(idx)
                
                conflicts = conflict_matrix[idx]
                avail_conflicts = np.where(available & conflicts)[0]
                degree[avail_conflicts] -= 1.0
                available[conflicts] = False
                available[idx] = False
            
            if len(selected) > len(best_set):
                best_set = selected
        
        if len(best_set) == 0:
            return np.empty((0, d), dtype=np.int64)
        
        return pool[best_set]
    
    pool4 = generate_pool(4)
    pool5 = generate_pool(5)
    
    results = []
    
    t_limit4 = deadline - 60
    res4 = greedy_min_degree(pool4, t_limit4, n_restarts=30)
    results.append((len(res4), res4, "w4"))
    
    if time.time() < deadline - 30:
        res5 = greedy_min_degree(pool5, deadline - 10, n_restarts=15)
        results.append((len(res5), res5, "w5"))
    
    if len(results) == 0:
        v1 = np.zeros(11, dtype=np.int64)
        v1[0] = 1
        v1[1] = 1
        v2 = np.zeros(11, dtype=np.int64)
        v2[0] = 1
        v2[1] = -1
        return np.vstack([v1, v2])
    
    results.sort(key=lambda x: -x[0])
    best_k, best_arr, best_name = results[0]
    
    best_arr = np.around(best_arr).astype(np.int64)
    
    if len(best_arr) < 2:
        v1 = np.zeros(11, dtype=np.int64)
        v1[0] = 1
        v1[1] = 1
        v2 = np.zeros(11, dtype=np.int64)
        v2[0] = 1
        v2[1] = -1
        best_arr = np.vstack([v1, v2])
    
    return best_arr

# EVOLVE-BLOCK-END
