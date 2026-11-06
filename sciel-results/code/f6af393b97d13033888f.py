import time
from math import lcm

def construct():
    start_time = time.time()
    deadline = 50.0  # seconds
    
    # Domain for small numbers A
    smalls = list(range(2, 129))
    n_small = len(smalls)
    
    # Base set B
    B = list(range(129, 257))
    sum_B = sum(1.0 / y for y in B)
    
    # Precompute multiples in B for each x in smalls
    # mults[x] is list of y in B such that x divides y
    mults = {x: [] for x in smalls}
    for x in smalls:
        for y in B:
            if y % x == 0:
                mults[x].append(y)
                
    # Precompute penalty sum for each x alone
    # penalty[x] = sum(1/y for y in mults[x])
    penalty = {x: sum(1.0 / y for y in mults[x]) for x in smalls}
    
    # Precompute compatibility graph for smalls
    # compat[x] is a bitset or set of z in smalls such that lcm(x, z) > 256
    # We need to know incompatibility to prune.
    # incompatible[x] is set of z such that lcm(x, z) <= 256
    incompatible = {x: set() for x in smalls}
    for i, x in enumerate(smalls):
        for z in smalls[i+1:]:
            if lcm(x, z) <= 256:
                incompatible[x].add(z)
                incompatible[z].add(x)
                
    # B&B State
    best_score = 0.0
    best_set = []
    
    # We will use a recursive function
    # current_set: list of selected x
    # available: set of x that can still be added (compatible with current_set)
    # current_sum: sum of 1/x for x in current_set
    # current_penalty_sum: sum of 1/y for y in union of mults[x] for x in current_set
    
    # To make B&B efficient, we process x in order of descending potential gain?
    # Or just index order?
    # Let's sort smalls by 1/x descending (i.e., x ascending)
    # But we need to maintain compatibility.
    
    # Optimization: Represent available as a bitset for fast intersection?
    # n_small is 127, fits in 2-3 Python ints.
    
    # Let's use a list of indices for smalls
    # index 0 -> 2, index 1 -> 3, etc.
    
    # Initial available: all smalls
    initial_available = set(smalls)
    
    def solve(current_set, available, current_sum, current_penalty_mask):
        nonlocal best_score, best_set, deadline
        
        if time.time() > deadline:
            return
            
        # Update best
        # Total score = current_sum - sum(1/y for y in union of mults) + sum_B
        # We track current_penalty_mask as a set of y's?
        # Or just recalculate?
        # Recalculating penalty sum from mask is O(|mask|).
        # Let's pass current_penalty_set (set of y in B)
        
        # Actually, passing a set of y's is expensive.
        # Let's pass a bitmask of B? B has 128 elements.
        # Or just a frozenset?
        
        # For simplicity and speed in Python, let's pass a set of y's.
        # 128 elements is small.
        
        score = current_sum - sum(1.0 / y for y in current_penalty_set) + sum_B
        if score > best_score:
            best_score = score
            best_set = list(current_set)
            
        # Upper Bound
        # Max possible additional gain
        # For each x in available, max gain is 1/x.
        # Sum of 1/x for all available is a loose bound.
        # Tighter: sum of max(0, 1/x - penalty_contribution)
        # But penalty overlaps.
        # Let's use sum(1/x for x in available) as bound.
        bound = current_sum + sum(1.0 / x for x in available) + sum_B
        # We also need to subtract minimum possible penalty?
        # Minimum penalty is 0 (if we pick nothing more).
        # So bound is valid.
        if bound <= best_score:
            return
            
        # Choose next x to branch on
        # Pick x in available with highest 1/x (smallest x)
        if not available:
            return
            
        # Sort available by x ascending
        # To avoid sorting every time, we can pre-sort and filter?
        # But available changes.
        # Let's just pick min(available)
        x = min(available)
        
        # Branch 1: Include x
        new_available = available - {x}
        # Remove incompatible nodes
        for z in incompatible[x]:
            if z in new_available:
                new_available.remove(z)
                
        new_penalty_set = current_penalty_set.union(mults[x])
        solve(current_set + [x], new_available, current_sum + 1.0/x, new_penalty_set)
        
        if time.time() > deadline:
            return
            
        # Branch 2: Exclude x
        solve(current_set, available - {x}, current_sum, current_penalty_set)

    solve([], initial_available, 0.0, set())
    
    # Construct final set
    # best_set contains small numbers
    # We need to add B \ union(mults[x] for x in best_set)
    penalty_set = set()
    for x in best_set:
        penalty_set.update(mults[x])
        
    final_set = set(best_set)
    for y in B:
        if y not in penalty_set:
            final_set.add(y)
            
    return list(final_set)
