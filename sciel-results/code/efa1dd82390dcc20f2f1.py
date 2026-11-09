import time
import random
from math import lcm
from functools import reduce
from itertools import combinations

UNIVERSE = list(range(1, 25))
L = reduce(lcm, UNIVERSE)
WEIGHTS = [L // x for x in UNIVERSE]

def is_valid(S):
    """Check if set S is reciprocal-subsum-free."""
    if not S:
        return True
    S_list = list(S)
    n = len(S_list)
    if n == 0:
        return True

    weights = [L // x for x in S_list]

    for i, x in enumerate(S_list):
        target = weights[i]
        others = weights[:i] + weights[i + 1:]
        if not others:
            continue
        
        # Quick check: if any single weight equals target
        if target in others:
            return False
            
        # For small sets, use direct enumeration
        m = len(others)
        if m <= 15:
            # Check all subsets directly
            for r in range(2, m + 1):
                for combo in combinations(others, r):
                    if sum(combo) == target:
                        return False
        else:
            # Meet-in-the-middle for larger sets
            mid = m // 2
            left = others[:mid]
            right = others[mid:]

            left_sums = {0}
            for w in left:
                left_sums = left_sums | {s + w for s in left_sums}

            right_sums = {0}
            found = False
            for w in right:
                new_sums = {s + w for s in right_sums}
                for s in new_sums:
                    if (target - s) in left_sums:
                        found = True
                        break
                if found:
                    break
                right_sums |= new_sums

            if found:
                return False

    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    def now():
        return time.time()

    def in_time():
        return now() < deadline

    # Initial greedy to get a good lower bound
    random.seed(2024)
    best_size = 0
    best_mask = 0
    
    # Try a few greedy orders
    orders = [
        list(range(24)), # 1 to 24
        list(range(23, -1, -1)), # 24 to 1
    ]
    for _ in range(10):
        perm = list(range(24))
        random.shuffle(perm)
        orders.append(perm)
        
    for order in orders:
        mask = 0
        size = 0
        for idx in order:
            # Check if adding idx is valid
            # Extract current weights
            curr_weights = [WEIGHTS[i] for i in range(24) if (mask >> i) & 1]
            new_weight = WEIGHTS[idx]
            
            # Check 1: Is new_weight a subset sum of curr_weights?
            # Check 2: Does new_weight complete a sum for any curr_weight?
            
            # Compute subset sums of curr_weights
            sums = {0}
            for w in curr_weights:
                sums = sums | {s + w for s in sums}
                
            valid = True
            if new_weight in sums:
                valid = False
            else:
                for w in curr_weights:
                    if w - new_weight in sums:
                        valid = False
                        break
                        
            if valid:
                mask |= (1 << idx)
                size += 1
                
        if size > best_size:
            best_size = size
            best_mask = mask
            
    if not in_time():
        return [i+1 for i in range(24) if (best_mask >> i) & 1]

    # B&B Search
    # Order indices by descending weight (most restrictive first) to find conflicts early?
    # Or ascending?
    # Let's sort indices by descending WEIGHTS to put "big" reciprocals first.
    indices = sorted(range(24), key=lambda i: WEIGHTS[i], reverse=True)
    
    # Map sorted index to original index
    # Actually, let's just iterate through the sorted list.
    
    global_best_size = best_size
    global_best_mask = best_mask
    
    def dfs(pos, mask, size):
        nonlocal global_best_size, global_best_mask
        
        if not in_time():
            return
            
        if size > global_best_size:
            global_best_size = size
            global_best_mask = mask
            
        if pos == 24:
            return
            
        # Pruning
        if size + (24 - pos) <= global_best_size:
            return
            
        idx = indices[pos]
        
        # Option 1: Include idx
        curr_weights = [WEIGHTS[i] for i in range(24) if (mask >> i) & 1]
        new_weight = WEIGHTS[idx]
        
        sums = {0}
        for w in curr_weights:
            sums = sums | {s + w for s in sums}
            
        valid = True
        if new_weight in sums:
            valid = False
        else:
            for w in curr_weights:
                if w - new_weight in sums:
                    valid = False
                    break
                    
        if valid:
            dfs(pos + 1, mask | (1 << idx), size + 1)
            
        # Option 2: Exclude idx
        dfs(pos + 1, mask, size)
        
    dfs(0, best_mask, best_size)
    
    best_list = [i+1 for i in range(24) if (global_best_mask >> i) & 1]
    if not best_list:
        best_list = [24]
        
    # Final validation
    if not is_valid(best_list):
        # Fallback to greedy result if B&B somehow broke it (shouldn't happen)
        best_list = [i+1 for i in range(24) if (best_mask >> i) & 1]
        if not best_list:
            best_list = [24]

    return best_list
