"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random
import time
from collections import defaultdict

def construct():
    n = 512
    
    # Precompute all 4-term arithmetic progressions
    # A 4-AP is (a, a+d, a+2d, a+3d)
    # For each element x, we store the APs it participates in
    # For each AP, we store its 4 elements
    
    ap_list = []
    num_to_aps = defaultdict(list)
    
    for d in range(1, (n - 3) // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            ap_idx = len(ap_list)
            ap_list.append((a, b, c, e))
            for x in (a, b, c, e):
                num_to_aps[x].append(ap_idx)
    
    # For fast validity checking, precompute for each x and each AP containing x,
    # the indices of the other 3 elements in that AP
    # Also precompute a bitmask for the "other three" for faster checking?
    # Actually, for N=512, we can use a boolean array or a set.
    # Let's use a set for the current solution S for O(1) membership.
    # But for speed, a boolean array in_set[1..n] is faster.
    
    # Precompute for each x, a list of (ap_idx, [other1, other2, other3])
    # Actually, just storing the other 3 elements is enough.
    x_to_conflicts = defaultdict(list)
    for ap_idx, (a, b, c, e) in enumerate(ap_list):
        for x in (a, b, c, e):
            others = [y for y in (a, b, c, e) if y != x]
            x_to_conflicts[x].append(others)
    
    def is_valid(in_set, x):
        """Check if x can be added to the set represented by in_set (boolean array)."""
        for others in x_to_conflicts[x]:
            if in_set[others[0]] and in_set[others[1]] and in_set[others[2]]:
                return False
        return True
    
    def try_add(in_set, S, x):
        """Try to add x to S. Returns True if successful."""
        if in_set[x]:
            return True
        if is_valid(in_set, x):
            in_set[x] = True
            S.add(x)
            return True
        return False
    
    def greedy_fill(in_set, S):
        """Try to add any valid element from 1 to n. Returns number of additions."""
        added = 0
        for x in range(1, n + 1):
            if not in_set[x]:
                if is_valid(in_set, x):
                    in_set[x] = True
                    S.add(x)
                    added += 1
        return added
    
    # Main search loop
    best_size = 0
    best_set = set()
    
    start_time = time.time()
    time_limit = 55.0
    
    # We'll do multiple restarts with different random orders
    # Each restart: randomized greedy init + local search (swaps + fills)
    
    while time.time() - start_time < time_limit:
        in_set = np.zeros(n + 1, dtype=bool)
        S = set()
        
        # Randomized greedy initialization
        order = list(range(1, n + 1))
        random.shuffle(order)
        
        for x in order:
            if is_valid(in_set, x):
                in_set[x] = True
                S.add(x)
        
        current_size = len(S)
        
        # Local search: try swaps
        # We'll do multiple passes of local search
        # In each pass, try random swaps until no improvement or limit reached
        
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            num_swaps = 0
            max_swaps = 5000  # Limit swaps per pass to avoid getting stuck
            
            while num_swaps < max_swaps and time.time() - start_time < time_limit:
                num_swaps += 1
                
                if current_size == 0:
                    break
                if current_size == n:
                    break
                
                # Pick random u in S
                u = random.choice(list(S))
                
                # Pick random v not in S
                v = random.randint(1, n)
                if in_set[v]:
                    continue
                
                # Try swap: remove u, add v
                # Check if v is valid in S - {u}
                # We need to check if adding v to (S without u) creates a 4-AP
                # This is equivalent to: for each AP containing v, check if the other 3
                # are all in S - {u}
                
                # Temporarily remove u
                in_set[u] = False
                S.remove(u)
                
                if is_valid(in_set, v):
                    # Swap accepted
                    in_set[v] = True
                    S.add(v)
                    current_size = len(S)
                    improved = True
                    
                    # Try to fill after swap
                    added = greedy_fill(in_set, S)
                    current_size = len(S)
                    
                    if current_size > best_size:
                        best_size = current_size
                        best_set = set(S)
                else:
                    # Swap rejected, restore u
                    in_set[u] = True
                    S.add(u)
                
                # Check time
                if time.time() - start_time > time_limit - 2.0:
                    break
        
        # If time is almost up, break
        if time.time() - start_time > time_limit - 5.0:
            break
    
    return list(best_set)

# EVOLVE-BLOCK-END
