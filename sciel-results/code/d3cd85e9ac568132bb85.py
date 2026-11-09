"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random
import time

def construct():
    n = 512
    
    # Precompute trios for fast validity check
    # trios_for_x[x] contains tuples (y, z, w) such that if y, z, w are in S, 
    # then x cannot be added (because {x, y, z, w} is a 4-AP)
    trios_for_x = {i: [] for i in range(1, n + 1)}
    
    for d in range(1, (n - 3) // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            
            # If we add a, we need b, c, e in S
            trios_for_x[a].append((b, c, e))
            # If we add b, we need a, c, e in S
            trios_for_x[b].append((a, c, e))
            # If we add c, we need a, b, e in S
            trios_for_x[c].append((a, b, e))
            # If we add e, we need a, b, c in S
            trios_for_x[e].append((a, b, c))
            
    def is_valid_addition(in_set, x):
        for y, z, w in trios_for_x[x]:
            if in_set[y] and in_set[z] and in_set[w]:
                return False
        return True

    best_size = 0
    best_set = set()
    
    start_time = time.time()
    time_limit = 55.0 # Leave 5s buffer
    
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
        
        # Local Search Loop
        while time.time() - start_time < time_limit:
            
            if current_size == 0 or current_size == n:
                break
            
            # Pick random u in S, v not in S
            u = random.choice(list(S_set))
            v = random.randint(1, n)
            if v in S_set:
                continue
            
            # Try swap: remove u, add v
            in_set[u] = False
            S_set.remove(u)
            
            if is_valid_addition(in_set, v):
                # Swap accepted
                in_set[v] = True
                S_set.add(v)
                current_size = len(S_set)
                
                # Try to fill again after swap
                for x in range(1, n + 1):
                    if not in_set[x]:
                        if is_valid_addition(in_set, x):
                            in_set[x] = True
                            S_set.add(x)
                            current_size = len(S_set)
                
                if current_size > best_size:
                    best_size = current_size
                    best_set = set(S_set)
            else:
                # Swap rejected, restore u
                in_set[u] = True
                S_set.add(u)
                
            if time.time() - start_time > time_limit - 1.0:
                break
        
        if time.time() - start_time > time_limit - 5.0:
            break
            
    return list(best_set)

# EVOLVE-BLOCK-END
