import numpy as np
import itertools
import time

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    
    def get_220():
        points = []
        for i in range(d):
            for j in range(i + 1, d):
                for s1 in [1, -1]:
                    for s2 in [1, -1]:
                        v = np.zeros(d, dtype=np.int64)
                        v[i] = s1
                        v[j] = s2
                        points.append(v)
        return np.array(points)
        
    best_points = get_220()
    best_k = len(best_points)
    
    def try_greedy(R2, candidates):
        # candidates is a list of numpy arrays
        # We want to select a subset such that u.v <= R2/2 for all pairs
        # Since u, v are integers, u.v is integer.
        # If R2 is even, u.v <= R2/2.
        # If R2 is odd, u.v <= (R2-1)/2.
        max_dot = R2 // 2
        
        # Shuffle to find a good local optimum
        np.random.shuffle(candidates)
        
        selected = []
        for v in candidates:
            conflict = False
            for u in selected:
                if np.dot(u, v) > max_dot:
                    conflict = True
                    break
            if not conflict:
                selected.append(v)
            
            if time.time() - start_time > 590:
                break
                
        if len(selected) > best_k:
            return np.array(selected)
        return None

    # Try R2 = 3
    # Generate candidates
    cand_3 = []
    for i in range(d):
        for j in range(i+1, d):
            for k in range(j+1, d):
                for s1 in [1, -1]:
                    for s2 in [1, -1]:
                        for s3 in [1, -1]:
                            v = np.zeros(d, dtype=np.int64)
                            v[i] = s1
                            v[j] = s2
                            v[k] = s3
                            cand_3.append(v)
                            
    # Run greedy a few times
    for _ in range(10):
        res = try_greedy(3, cand_3)
        if res is not None:
            best_points = res
            best_k = len(best_points)
            if time.time() - start_time > 590:
                break

    # Try R2 = 4
    # Candidates: 2e_i and e_i+e_j+e_k+e_l
    cand_4 = []
    for i in range(d):
        v = np.zeros(d, dtype=np.int64)
        v[i] = 2
        cand_4.append(v)
        v[i] = -2
        cand_4.append(v)
        
    for i in range(d):
        for j in range(i+1, d):
            for k in range(j+1, d):
                for l in range(k+1, d):
                    for s1 in [1, -1]:
                        for s2 in [1, -1]:
                            for s3 in [1, -1]:
                                for s4 in [1, -1]:
                                    v = np.zeros(d, dtype=np.int64)
                                    v[i] = s1
                                    v[j] = s2
                                    v[k] = s3
                                    v[l] = s4
                                    cand_4.append(v)
                                    
    for _ in range(5):
        res = try_greedy(4, cand_4)
        if res is not None:
            best_points = res
            best_k = len(best_points)
            if time.time() - start_time > 590:
                break

    return best_points
