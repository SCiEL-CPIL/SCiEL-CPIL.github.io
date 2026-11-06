import time
import random

def construct():
    deadline = time.time() + 85
    n = 256
    
    # Precompute for each position, the list of (start, step) APs containing it
    ap_containing = [[] for _ in range(n)]
    for pos in range(n):
        for d in range(1, n + 1):
            s = pos
            while s >= 0:
                ap_containing[pos].append((s, d))
                s -= d
    
    def compute_all_sums(signs):
        sums = {}
        for d in range(1, n + 1):
            for start in range(n):
                s = 0
                idx = start
                while idx < n:
                    s += signs[idx]
                    idx += d
                sums[(start, d)] = s
        return sums
    
    best_signs = None
    best_score = float('inf')
    
    rng = random.Random(42)
    
    for restart in range(200):
        if time.time() > deadline - 10:
            break
        
        signs = [rng.choice((-1, 1)) for _ in range(n)]
        sums = compute_all_sums(signs)
        
        hist = [0] * (n + 1)
        for v in sums.values():
            hist[abs(v)] += 1
        
        max_val = n
        while max_val > 0 and hist[max_val] == 0:
            max_val -= 1
        
        # Greedy pass
        for _ in range(10):
            improved = False
            for pos in range(n):
                old_sign = signs[pos]
                delta = 2 * old_sign
                
                changed = []
                for (start, d) in ap_containing[pos]:
                    key = (start, d)
                    old_sum = sums[key]
                    new_sum = old_sum + delta
                    hist[abs(old_sum)] -= 1
                    hist[abs(new_sum)] += 1
                    sums[key] = new_sum
                    changed.append((key, old_sum, new_sum))
                
                new_max = max_val
                if hist[new_max] == 0:
                    while new_max > 0 and hist[new_max] == 0:
                        new_max -= 1
                
                if new_max < max_val:
                    max_val = new_max
                    signs[pos] = -old_sign
                    improved = True
                else:
                    signs[pos] = old_sign
                    for (key, old_sum, new_sum) in changed:
                        hist[abs(new_sum)] -= 1
                        hist[abs(old_sum)] += 1
                        sums[key] = old_sum
            if not improved:
                break
        
        if max_val < best_score:
            best_score = max_val
            best_signs = signs[:]
        
        if best_score <= 2:
            break
    
    return best_signs
