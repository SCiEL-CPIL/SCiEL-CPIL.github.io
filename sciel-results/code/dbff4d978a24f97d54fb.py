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
    
    # Precompute "other 3" for each AP and each position to speed up updates
    # ap_info[ap_idx] = [(x1, [x2,x3,x4]), (x2, [x1,x3,x4]), ...]
    # Actually, we just need to know, for a given x and ap_idx, what are the other 3.
    # We can store ap_list as tuples and just filter.
    
    # We will maintain:
    # in_set: boolean array
    # block_count: array of integers. block_count[x] > 0 means x cannot be added.
    
    def update_block_counts_add(in_set, block_count, x):
        """Update block_counts when adding x to the set."""
        for ap_idx in num_to_aps[x]:
            a, b, c, e = ap_list[ap_idx]
            others = [y for y in (a, b, c, e) if y != x]
            # If 2 of the others are in set, the 3rd is blocked
            # We need to check all pairs of others
            # others[0], others[1] in set -> others[2] blocked
            if in_set[others[0]] and in_set[others[1]]:
                block_count[others[2]] += 1
            if in_set[others[0]] and in_set[others[2]]:
                block_count[others[1]] += 1
            if in_set[others[1]] and in_set[others[2]]:
                block_count[others[0]] += 1

    def update_block_counts_remove(in_set, block_count, x):
        """Update block_counts when removing x from the set."""
        for ap_idx in num_to_aps[x]:
            a, b, c, e = ap_list[ap_idx]
            others = [y for y in (a, b, c, e) if y != x]
            if in_set[others[0]] and in_set[others[1]]:
                block_count[others[2]] -= 1
            if in_set[others[0]] and in_set[others[2]]:
                block_count[others[1]] -= 1
            if in_set[others[1]] and in_set[others[2]]:
                block_count[others[0]] -= 1

    best_size = 0
    best_set = set()
    
    start_time = time.time()
    time_limit = 55.0
    
    while time.time() - start_time < time_limit:
        
        in_set = np.zeros(n + 1, dtype=bool)
        block_count = np.zeros(n + 1, dtype=int)
        S_set = set()
        
        # Greedy Initialization
        # Use a random order, but we can bias towards numbers with lower degree
        # or just random.
        order = list(range(1, n + 1))
        random.shuffle(order)
        
        for x in order:
            if block_count[x] == 0:
                in_set[x] = True
                S_set.add(x)
                update_block_counts_add(in_set, block_count, x)
        
        current_size = len(S_set)
        
        # Local Search
        while time.time() - start_time < time_limit:
            
            # 1. Try to fill any unblocked numbers
            # We can iterate through all numbers not in S
            # To be faster, we could maintain a list of candidates, but N=512 is small
            # We'll just do a pass.
            any_added = False
            for x in range(1, n + 1):
                if not in_set[x] and block_count[x] == 0:
                    in_set[x] = True
                    S_set.add(x)
                    update_block_counts_add(in_set, block_count, x)
                    current_size += 1
                    any_added = True
                    
            if current_size > best_size:
                best_size = current_size
                best_set = set(S_set)
                
            if not any_added:
                # No more fills possible, try swaps
                if current_size == 0 or current_size == n:
                    break
                
                # Pick random u in S
                u = random.choice(list(S_set))
                # Pick random v not in S
                v = random.randint(1, n)
                if v in S_set:
                    continue
                
                # Try swap: remove u, add v
                # Remove u
                in_set[u] = False
                S_set.remove(u)
                update_block_counts_remove(in_set, block_count, u)
                
                # Check if v can be added
                if block_count[v] == 0:
                    # Add v
                    in_set[v] = True
                    S_set.add(v)
                    update_block_counts_add(in_set, block_count, v)
                    current_size += 1 # Net change 0, but we added v and removed u
                    # Actually size is same. 
                    # But we might enable new fills.
                    
                    # Try to fill again
                    # We don't need to check if size increased here immediately, 
                    # the fill loop above will handle it next iteration.
                    # But we can do a quick fill check here to see if we improved.
                    # For efficiency, just let the main loop handle it.
                else:
                    # Swap rejected, restore u
                    in_set[u] = True
                    S_set.add(u)
                    update_block_counts_add(in_set, block_count, u)
                    
            if time.time() - start_time > time_limit - 1.0:
                break
                
        if time.time() - start_time > time_limit - 5.0:
            break
            
    return list(best_set)

# EVOLVE-BLOCK-END
