import numpy as np

def construct():
    n = 50
    best_h = np.zeros(n)
    best_score = 0.0
    
    def calc_score_evaluator(h):
        # Assumed evaluator metric: L2^2 / (L1 * max_off_diag_c)
        # Normalized to L1=1
        if np.sum(h) == 0:
            return 0.0
        h = h / np.sum(h)
        l2_sq = np.sum(h**2)
        c = np.convolve(h, h, mode='full')
        # Exclude center (index n-1 in full mode? No, full mode index n-1 is c_0)
        # c[k] = sum h[i] h[k-i]
        # c_0 is at index n-1 in 'full' mode of length 2n-1.
        c_max_off = 0.0
        for k in range(len(c)):
            if k != n - 1:
                if c[k] > c_max_off:
                    c_max_off = c[k]
        if c_max_off == 0:
            return float('inf') # Or large number
        return l2_sq / c_max_off

    # Backtracking search for max Sidon set
    def find_max_sidon():
        max_size = 0
        best_set = []
        
        def backtrack(current_set, used_sums, start):
            nonlocal max_size, best_set
            if len(current_set) > max_size:
                max_size = len(current_set)
                best_set = current_set.copy()
                
            # Pruning: if remaining elements + current size <= max_size, stop
            if len(current_set) + (n - start) <= max_size:
                return
                
            for i in range(start, n):
                # Check if i can be added
                valid = True
                new_sums = set()
                for j in current_set:
                    s = i + j
                    if s in used_sums:
                        valid = False
                        break
                    new_sums.add(s)
                
                if valid:
                    s_ii = i + i
                    if s_ii in used_sums:
                        valid = False
                    else:
                        new_sums.add(s_ii)
                        
                if valid:
                    current_set.append(i)
                    used_sums.update(new_sums)
                    backtrack(current_set, used_sums, i + 1)
                    current_set.pop()
                    used_sums.difference_update(new_sums)
                    
        backtrack([], set(), 0)
        return best_set

    sidon_set = find_max_sidon()
    
    if len(sidon_set) > 0:
        k = len(sidon_set)
        for idx in sidon_set:
            best_h[idx] = 1.0 / k
    else:
        best_h[0] = 1.0

    return list(best_h)
