import time
import itertools
import random
import sys

def _deadline_hit(deadline, margin=0.0):
    return time.time() >= deadline - margin

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin

    n = 24
    L = 5354228880
    w = [L // i for i in range(1, n + 1)]

    # Precompute subset sums
    # Using a list of size 2^n is memory intensive but fast enough for 24
    # 2^24 = 16,777,216
    # List of ints ~ 134 MB. Acceptable.
    
    # To save memory and time, we can just use a dictionary for sums that match targets?
    # No, we need to find all masks that sum to w[i].
    # We can iterate masks and check if sum is in targets.
    # But calculating sum is the bottleneck.
    
    # DP Sum Calculation
    # sum_w[m] = sum_w[m & (m-1)] + w[lsb]
    # We only care about sums that are in {w[0], ..., w[23]}
    # But we don't know which sums appear without calculating them.
    # However, we can store the sum in a dictionary only if it's a target?
    # No, because intermediate sums might not be targets, but we need the final sum.
    # Actually, we can just store the sum for all masks.
    
    # Optimization: Use array module or list. List is fine.
    sum_w = [0] * (1 << n)
    for m in range(1, 1 << n):
        lsb = m & -m
        idx = lsb.bit_length() - 1
        sum_w[m] = sum_w[m ^ lsb] + w[idx]
        
    if _deadline_hit(deadline, 10):
        return list(range(13, 25))

    # Group masks by sum
    # We only care about sums equal to w[i]
    targets = set(w)
    sum_to_masks = {}
    for m in range(1, 1 << n):
        s = sum_w[m]
        if s in targets:
            if s in sum_to_masks:
                sum_to_masks[s].append(m)
            else:
                sum_to_masks[s] = [m]

    # Find minimal forbidden masks
    forbidden = set()
    for x in range(1, n + 1):
        target = w[x - 1]
        if target in sum_to_masks:
            for m in sum_to_masks[target]:
                if (m & (1 << (x - 1))) == 0:
                    fm = m | (1 << (x - 1))
                    forbidden.add(fm)

    # Remove supersets
    forbidden_list = sorted(forbidden, key=lambda x: bin(x).count('1'))
    minimal_forbidden = []
    for fm in forbidden_list:
        is_superset = False
        for mf in minimal_forbidden:
            if (fm & mf) == mf:
                is_superset = True
                break
        if not is_superset:
            minimal_forbidden.append(fm)
            
    # Sort minimal_forbidden by size for faster pruning in backtracking
    minimal_forbidden.sort(key=lambda x: bin(x).count('1'))
    
    # Backtracking Search for Maximum Valid Set
    # We want to maximize the number of elements in the set.
    # We try to include elements 0..23.
    # Pruning: If current_size + (n - idx) <= best_size, prune.
    
    best_size = 0
    best_mask = 0
    
    # Order elements by "conflict degree" to prune earlier?
    # Or just 0..23.
    # Let's try 0..23 first.
    
    def dfs(idx, current_mask, current_size):
        nonlocal best_size, best_mask
        
        if _deadline_hit(deadline, 5):
            return
            
        # Pruning
        if current_size + (n - idx) <= best_size:
            return
            
        if idx == n:
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask
            return
        
        # Option 1: Exclude idx
        dfs(idx + 1, current_mask, current_size)
        
        # Option 2: Include idx
        new_mask = current_mask | (1 << idx)
        
        # Check if new_mask contains any forbidden mask
        valid = True
        for fm in minimal_forbidden:
            if (new_mask & fm) == fm:
                valid = False
                break
        
        if valid:
            dfs(idx + 1, new_mask, current_size + 1)
            
    dfs(0, 0, 0)
    
    # If backtracking didn't find anything or timed out, use a greedy fallback
    if best_size == 0:
        # Greedy: add elements if they don't create violations
        current_mask = 0
        for i in range(n):
            new_mask = current_mask | (1 << i)
            valid = True
            for fm in minimal_forbidden:
                if (new_mask & fm) == fm:
                    valid = False
                    break
            if valid:
                current_mask = new_mask
        best_mask = current_mask
        best_size = bin(best_mask).count('1')
        
    # Local Search Polish (WalkSAT) to potentially find a larger set if backtracking was pruned early
    # Or to fix if backtracking failed due to time limit
    if _deadline_hit(deadline, 10):
        pass # Just return what we have
    
    current_mask = best_mask
    current_size = best_size
    
    # Simple Local Search: Try to add elements that were excluded
    # This is useful if the backtracking got stuck in a local optimum or pruned too aggressively
    # But backtracking is exact if it finishes.
    # If it didn't finish, we might have a suboptimal best_size.
    
    # Let's try a few random restarts of greedy or local search if we have time
    if not _deadline_hit(deadline, 10):
        random.seed(12345)
        for _ in range(50): # 50 restarts
            if _deadline_hit(deadline, 5):
                break
            # Greedy with random order
            order = list(range(n))
            random.shuffle(order)
            c_mask = 0
            c_size = 0
            for i in order:
                new_mask = c_mask | (1 << i)
                valid = True
                for fm in minimal_forbidden:
                    if (new_mask & fm) == fm:
                        valid = False
                        break
                if valid:
                    c_mask = new_mask
                    c_size += 1
            
            if c_size > best_size:
                best_size = c_size
                best_mask = c_mask

    if best_mask:
        result = [i + 1 for i in range(n) if (best_mask & (1 << i))]
        if result:
            return result

    return [1]
