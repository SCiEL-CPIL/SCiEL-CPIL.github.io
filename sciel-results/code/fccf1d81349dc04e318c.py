import time
import random
import sys

N = 256

def _calc_score(S):
    """Calculate sum of reciprocals."""
    return sum(1.0 / x for x in S)

def _is_valid_add_bitmask(mask, y):
    """Check if y can be added to the set represented by bitmask without forming a 3-AP.
    Bit i represents element i+1 (so bit 0 = element 1, bit 255 = element 256).
    """
    if mask & (1 << (y - 1)):
        return False
    
    # For each a in S, check if adding y creates a 3-AP
    # We need to check three cases:
    # 1. y is midpoint: a, y, b => b = 2y - a
    # 2. y is left endpoint: y, b, c => b = y + d, c = y + 2d for some d
    #    Equivalently: for each b in S where b > y, check if c = 2b - y is in S
    # 3. y is right endpoint: a, b, y => b = (a+y)/2 if a+y even
    
    # More efficient: iterate through all possible 3-APs involving y
    # A 3-AP (a, b, c) has b = (a+c)/2, so a+c = 2b
    # If y is in the AP, either:
    # - y = a: need to check if there exist b, c in S with b-c = b-y and c-b = b-y, i.e., b-y = c-b
    #   This means for each b in S with b > y, check if c = 2b - y is in S and c > b
    # - y = b: need to check if there exist a, c in S with a+c = 2y
    #   For each a in S with a < y, check if c = 2y - a is in S and c > y
    # - y = c: need to check if there exist a, b in S with a+2(b-a) = y, i.e., b = (a+y)/2
    #   For each a in S with a < y and (a+y) even, check if b = (a+y)//2 is in S and a < b < y
    
    # Let's use a different approach: for each pair (a, c) in S, check if midpoint is y
    # Or: for each d >= 1, check if y-d and y+d are both in S (y is midpoint)
    #      or if y and y+d and y+2d are in S (y is left endpoint)
    #      or if y-2d and y-d and y are in S (y is right endpoint)
    
    # Case 1: y is midpoint. Check if there exists d >= 1 such that y-d >= 1 and y+d <= N and both in S
    d = 1
    while y - d >= 1 and y + d <= N:
        if (mask & (1 << (y - d - 1))) and (mask & (1 << (y + d - 1))):
            return False
        d += 1
    
    # Case 2: y is left endpoint. Check if there exists d >= 1 such that y+d <= N and y+2d <= N and both in S
    d = 1
    while y + 2 * d <= N:
        if (mask & (1 << (y + d - 1))) and (mask & (1 << (y + 2 * d - 1))):
            return False
        d += 1
    
    # Case 3: y is right endpoint. Check if there exists d >= 1 such that y-2d >= 1 and y-d >= 1 and both in S
    d = 1
    while y - 2 * d >= 1:
        if (mask & (1 << (y - 2 * d - 1))) and (mask & (1 << (y - d - 1))):
            return False
        d += 1
    
    return True

def _greedy_init():
    """Greedy initialization: add elements in order of decreasing reciprocal weight (i.e., increasing value)."""
    mask = 0
    S = set()
    # Try to add elements from 1 to 256 (smaller numbers have higher reciprocal weight)
    for x in range(1, N + 1):
        if _is_valid_add_bitmask(mask, x):
            mask |= (1 << (x - 1))
            S.add(x)
    return S, mask

def _local_search(S, mask, deadline):
    """Perform local search to improve the set."""
    best_S = set(S)
    best_mask = mask
    best_score = _calc_score(best_S)
    
    # Precompute reciprocal weights
    recip = [0.0] + [1.0 / i for i in range(1, N + 1)]
    
    # Simulated annealing
    T = 5.0
    cooling_rate = 0.9995
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < deadline and T > 0.01:
        if no_improve_count > max_no_improve:
            # Reset temperature occasionally
            T = max(T, 2.0)
            no_improve_count = 0
        
        # Generate a move
        move_type = random.random()
        
        if move_type < 0.4:
            # Try to add an element
            # Prefer smaller numbers (higher reciprocal weight)
            candidates = [x for x in range(1, N + 1) if not (mask & (1 << (x - 1)))]
            if not candidates:
                continue
            # Weight towards smaller numbers
            weights = [recip[x] for x in candidates]
            total_w = sum(weights)
            if total_w == 0:
                y = random.choice(candidates)
            else:
                r = random.random() * total_w
                cum = 0
                y = candidates[-1]
                for x, w in zip(candidates, weights):
                    cum += w
                    if r <= cum:
                        y = x
                        break
            
            if _is_valid_add_bitmask(mask, y):
                new_mask = mask | (1 << (y - 1))
                new_S = S | {y}
                new_score = best_score + recip[y]
                delta = new_score - _calc_score(S)
                
                if delta > 0 or random.random() < pow(2.718281828, delta / T):
                    S = new_S
                    mask = new_mask
                    if new_score > best_score:
                        best_score = new_score
                        best_S = new_S
                        best_mask = new_mask
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
        
        elif move_type < 0.8:
            # Try to swap: remove x from S, add y not in S
            if not S:
                continue
            x = random.choice(list(S))
            candidates = [y for y in range(1, N + 1) if not (mask & (1 << (y - 1)))]
            if not candidates:
                continue
            
            # Prefer y with higher reciprocal weight than x
            # Weight towards smaller y
            weights = [max(recip[y] - recip[x], 0.01) for y in candidates]
            total_w = sum(weights)
            if total_w == 0:
                y = random.choice(candidates)
            else:
                r = random.random() * total_w
                cum = 0
                y = candidates[-1]
                for x_cand, w in zip(candidates, weights):
                    cum += w
                    if r <= cum:
                        y = x_cand
                        break
            
            # Check if swap is valid
            mask_without_x = mask & ~(1 << (x - 1))
            if _is_valid_add_bitmask(mask_without_x, y):
                new_mask = mask_without_x | (1 << (y - 1))
                new_S = (S - {x}) | {y}
                new_score = _calc_score(new_S)
                old_score = _calc_score(S)
                delta = new_score - old_score
                
                if delta > 0 or random.random() < pow(2.718281828, delta / T):
                    S = new_S
                    mask = new_mask
                    if new_score > best_score:
                        best_score = new_score
                        best_S = new_S
                        best_mask = new_mask
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
        else:
            # Try to remove an element (to free up space for better additions)
            if len(S) <= 1:
                continue
            # Prefer removing larger elements (lower reciprocal weight)
            x = max(S, key=lambda v: -recip[v])  # Remove the one with lowest reciprocal weight
            if random.random() < 0.5:
                # Actually remove it
                new_mask = mask & ~(1 << (x - 1))
                new_S = S - {x}
                new_score = _calc_score(new_S)
                old_score = _calc_score(S)
                delta = new_score - old_score
                
                if random.random() < pow(2.718281828, delta / T):
                    S = new_S
                    mask = new_mask
                    no_improve_count += 1
        
        T *= cooling_rate
    
    return best_S, best_mask, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Initialize with greedy
    S, mask = _greedy_init()
    
    # Run local search
    best_S, best_mask, best_score = _local_search(S, mask, deadline)
    
    return list(best_S)
