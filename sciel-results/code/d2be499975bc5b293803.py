import time
import random
from math import gcd

def construct():
    N = 128
    elements = list(range(1, N + 1))
    
    # Precompute gcd matrix
    gcds = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcds[i][j] = g
            gcds[j][i] = g
            
    # Precompute conflict masks
    # conflicts[x] is a list of masks. Each mask has bits y and z set
    # if (x, y, z) is a bad triple.
    conflicts = [[] for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            gab = gcds[a][b]
            for c in range(b + 1, N + 1):
                gbc = gcds[b][c]
                if gab == gbc:
                    gca = gcds[c][a]
                    if gca == gab:
                        mask_bc = (1 << b) | (1 << c)
                        mask_ac = (1 << a) | (1 << c)
                        mask_ab = (1 << a) | (1 << b)
                        
                        conflicts[a].append(mask_bc)
                        conflicts[b].append(mask_ac)
                        conflicts[c].append(mask_ab)

    start_time = time.time()
    deadline = start_time + 55
    random.seed(42)
    
    def can_add(S_mask, x):
        for mask in conflicts[x]:
            if (S_mask & mask) == mask:
                return False
        return True

    def greedy_build(order):
        S_mask = 0
        for x in order:
            if can_add(S_mask, x):
                S_mask |= (1 << x)
        return S_mask

    def local_search(S_mask):
        current_size = bin(S_mask).count("1")
        
        while time.time() < deadline:
            improved = False
            
            # Phase 1: Try to add any element
            not_in_S = [x for x in range(1, N + 1) if not (S_mask & (1 << x))]
            random.shuffle(not_in_S)
            
            for x in not_in_S:
                if can_add(S_mask, x):
                    S_mask |= (1 << x)
                    current_size += 1
                    improved = True
                    break
            
            if improved:
                continue

            # Phase 2: Smart Swap
            # Identify elements in S that are "blocking" many elements outside S
            # For each y in S, count how many x in not_in_S are blocked by y?
            # This is expensive. 
            # Alternative: Random swap with bias towards "high degree" nodes in conflict graph?
            
            # Let's try a simple random swap but with more iterations
            elems_in_S = [x for x in range(1, N + 1) if (S_mask & (1 << x))]
            if not elems_in_S:
                break
                
            y = random.choice(elems_in_S)
            S_temp = S_mask & ~(1 << y)
            
            # Try to add elements to S_temp
            # Prioritize elements that were in conflicts with y?
            # We don't track that easily. Just shuffle not_in_S.
            not_in_S_temp = [x for x in range(1, N + 1) if not (S_temp & (1 << x))]
            random.shuffle(not_in_S_temp)
            
            added = False
            for x in not_in_S_temp:
                if can_add(S_temp, x):
                    S_temp |= (1 << x)
                    added = True
                    break
            
            if added:
                S_mask = S_temp
                current_size = bin(S_mask).count("1")
            else:
                # If no swap worked, we might be stuck in a local optimum.
                # Try a "kick": remove 2 random elements, add back greedily?
                # Or just restart local search with a random perturbation.
                if random.random() < 0.1:
                    # Perturb: remove 2 random elements
                    if len(elems_in_S) > 2:
                        y1, y2 = random.sample(elems_in_S, 2)
                        S_mask = S_mask & ~(1 << y1) & ~(1 << y2)
                        current_size -= 2
                    else:
                        break

        return S_mask, current_size

    best_set_mask = 0
    best_size = 0
    
    # Multiple restarts with different heuristics
    # Heuristic 1: Random order
    # Heuristic 2: Order by number of conflicts (fewest first)
    # Heuristic 3: Order by value
    
    conflict_counts = [len(conflicts[i]) for i in range(N + 1)]
    
    orders = []
    orders.append(list(range(1, N + 1)))
    random.shuffle(orders[0])
    
    orders.append(sorted(range(1, N + 1), key=lambda x: conflict_counts[x]))
    orders.append(sorted(range(1, N + 1), key=lambda x: -conflict_counts[x]))
    orders.append(list(range(1, N + 1)))
    
    for run in range(50):
        if time.time() > deadline - 5:
            break
            
        if run < len(orders):
            order = orders[run]
        else:
            order = list(range(1, N + 1))
            random.shuffle(order)
            
        S_mask = greedy_build(order)
        S_mask, size = local_search(S_mask)
        
        if size > best_size:
            best_size = size
            best_set_mask = S_mask

    result = [x for x in range(1, N + 1) if (best_set_mask & (1 << x))]
    
    if not result:
        return [1]
        
    return result
