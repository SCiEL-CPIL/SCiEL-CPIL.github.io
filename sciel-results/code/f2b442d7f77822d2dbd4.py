"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random
import time

def construct():
    n = 512
    
    # Precompute all 4-term arithmetic progressions
    # A 4-AP is (a, a+d, a+2d, a+3d)
    ap_list = []
    # Map each number to the indices of the APs it participates in
    num_to_aps = {i: [] for i in range(1, n + 1)}
    
    for d in range(1, (n - 3) // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            ap_idx = len(ap_list)
            ap_list.append((a, b, c, e))
            for x in (a, b, c, e):
                num_to_aps[x].append(ap_idx)
    
    # Helper to check if adding x to current set S (represented as boolean array or set) is valid
    # We'll use a set for S for O(1) lookup, and a boolean array for speed if needed, but set is fine for N=512
    # Actually, a boolean array `in_set` is faster for membership.
    
    def is_valid_addition(in_set, x):
        # Check all APs containing x
        for ap_idx in num_to_aps[x]:
            a, b, c, e = ap_list[ap_idx]
            # The other 3 elements
            others = [y for y in (a, b, c, e) if y != x]
            if in_set[others[0]] and in_set[others[1]] and in_set[others[2]]:
                return False
        return True

    def greedy_fill(in_set, S_set, candidates):
        # Try to add any valid candidate from the list
        # To make it faster, we can iterate through all 1..n not in S
        # But maintaining a candidate list is tricky if we reject some.
        # Simple approach: iterate through all x in 1..n
        added_any = False
        for x in range(1, n + 1):
            if not in_set[x]:
                if is_valid_addition(in_set, x):
                    in_set[x] = True
                    S_set.add(x)
                    added_any = True
        return added_any

    # Initial Greedy Solution with Random Order
    # We will run multiple iterations of (Greedy Init + Local Search)
    
    best_size = 0
    best_set = set()
    
    # Time limit management
    start_time = time.time()
    time_limit = 55.0 # Leave 5s buffer
    
    # Number of restarts? We can just loop until time runs out.
    # Each iteration:
    # 1. Randomized Greedy Initialization
    # 2. Local Search (Swap and Fill)
    
    while time.time() - start_time < time_limit:
        
        # 1. Initialization
        in_set = np.zeros(n + 1, dtype=bool)
        S_set = set()
        
        # Random order for greedy
        order = list(range(1, n + 1))
        random.shuffle(order)
        
        for x in order:
            if is_valid_addition(in_set, x):
                in_set[x] = True
                S_set.add(x)
        
        # 2. Local Search
        current_size = len(S_set)
        
        # First, try to fill any gaps from the initial greedy (in case random order left some easy picks)
        # Actually, the greedy above already checked all numbers in random order.
        # But a number rejected early might be valid later? No, adding numbers only creates more conflicts.
        # So if it was invalid then, it's still invalid now.
        # So no need to re-fill unless we do swaps.
        
        # Local Search Loop
        while time.time() - start_time < time_limit:
            # Try to add any new number
            # Since we did a full greedy pass, there are no numbers that can be added immediately.
            # So we must do a swap.
            
            # Pick random u in S, v not in S
            if current_size == 0:
                break
            if current_size == n:
                break # Perfect set found (unlikely)
            
            # To avoid picking v that is obviously invalid, we can just pick random
            u = random.choice(list(S_set))
            # Pick v from 1..n not in S
            # To be efficient, we can maintain a list of non-members, but S changes.
            # For N=512, picking random v and checking if in S is fast enough.
            v = random.randint(1, n)
            if v in S_set:
                continue
            
            # Try swap: remove u, add v
            # Check if S - {u} + {v} is valid
            # This is equivalent to:
            # 1. Is v valid in S - {u}?
            #    i.e., does v form a 4-AP with 3 elements from S - {u}?
            # 2. Does removing u break any 4-AP that was previously "blocked"? 
            #    No, removing an element never creates a new 4-AP. It only removes potential conflicts.
            #    So we only need to check if adding v to (S - {u}) creates a 4-AP.
            
            # Temporarily remove u
            in_set[u] = False
            S_set.remove(u)
            
            if is_valid_addition(in_set, v):
                # Swap accepted
                in_set[v] = True
                S_set.add(v)
                current_size = len(S_set)
                
                # Try to fill again after swap
                # This might allow adding more numbers
                # We'll do a single pass of greedy fill
                # Note: is_valid_addition is expensive if called for all N.
                # But N=512 is small.
                for x in range(1, n + 1):
                    if not in_set[x]:
                        if is_valid_addition(in_set, x):
                            in_set[x] = True
                            S_set.add(x)
                            current_size = len(S_set)
                
                if current_size > best_size:
                    best_size = current_size
                    best_set = set(S_set)
                    
                # If we filled a lot, we might have made it harder to swap further.
                # But the loop continues.
            else:
                # Swap rejected, restore u
                in_set[u] = True
                S_set.add(u)
                # current_size unchanged
                
            # If we have reached a local maximum and time is running out, break
            if time.time() - start_time > time_limit - 1.0:
                break
        
        # If time is almost up, break outer loop
        if time.time() - start_time > time_limit - 5.0:
            break
            
    # Convert best_set to list
    return list(best_set)

# EVOLVE-BLOCK-END
