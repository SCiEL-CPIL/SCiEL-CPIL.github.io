"""Optimized C4-free edge set construction for Q7."""
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    d = 7
    n = 2 ** d
    m = n * d // 2  # 448 edges
    
    # Precompute all edges and their properties
    all_edges = []
    edge_to_idx = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(all_edges)
                all_edges.append((x, y))
                edge_to_idx[(x, y)] = idx
    
    # Precompute forbidden triples for each edge
    # For edge idx with direction i, the C4 faces are formed with each j != i
    # In face (i,j), the other 3 edges form a triple that conflicts with idx
    forbidden_triples = [[] for _ in range(m)]
    for idx in range(m):
        x, y = all_edges[idx]
        diff = x ^ y
        i = diff.bit_length() - 1
        for j in range(d):
            if j == i:
                continue
            c = x ^ (1 << j)
            e = y ^ (1 << j)
            e1 = (min(x, c), max(x, c))
            e2 = (min(y, e), max(y, e))
            e3 = (min(c, e), max(c, e))
            idx1 = edge_to_idx.get(e1)
            idx2 = edge_to_idx.get(e2)
            idx3 = edge_to_idx.get(e3)
            if idx1 is not None and idx2 is not None and idx3 is not None:
                forbidden_triples[idx].append((idx1, idx2, idx3))
    
    # Track which edges are in each triple for fast conflict checking
    # conflict_count[idx] = number of triples where all other 3 edges are selected
    conflict_count = np.zeros(m, dtype=np.int32)
    
    def can_add(idx, chosen):
        """Check if edge idx can be added without creating a C4."""
        for (i1, i2, i3) in forbidden_triples[idx]:
            if chosen[i1] and chosen[i2] and chosen[i3]:
                return False
        return True
    
    def greedy_polish(chosen, count):
        """Try to add all possible unselected edges."""
        for idx in range(m):
            if not chosen[idx]:
                if can_add(idx, chosen):
                    chosen[idx] = True
                    count += 1
        return count
    
    # Initial construction using a structured approach
    # Use a greedy algorithm that processes edges in a smart order
    chosen = np.zeros(m, dtype=bool)
    count = 0
    
    # Order edges by their direction and position for better initial packing
    edge_order = list(range(m))
    random.shuffle(edge_order)
    
    for idx in edge_order:
        if can_add(idx, chosen):
            chosen[idx] = True
            count += 1
    
    best_count = count
    best_chosen = chosen.copy()
    
    # Greedy polish
    count = greedy_polish(chosen, count)
    if count > best_count:
        best_count = count
        best_chosen = chosen.copy()
    
    # Local search with simulated annealing
    T = 2.0
    T_min = 0.01
    cooling = 0.99999
    no_improve = 0
    max_no_improve = 30000
    
    while time.time() < deadline and T > T_min:
        idx = random.randint(0, m - 1)
        
        if not chosen[idx]:
            # Try adding
            if can_add(idx, chosen):
                chosen[idx] = True
                count += 1
                if count > best_count:
                    best_count = count
                    best_chosen = chosen.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Try a swap: find a conflicting edge to remove
                conflicts = []
                for (i1, i2, i3) in forbidden_triples[idx]:
                    if chosen[i1] and chosen[i2] and chosen[i3]:
                        conflicts.extend([i1, i2, i3])
                
                if conflicts:
                    remove_idx = random.choice(conflicts)
                    if not chosen[remove_idx]:
                        no_improve += 1
                        continue
                    
                    # Try the swap
                    chosen[remove_idx] = False
                    count -= 1
                    if can_add(idx, chosen):
                        chosen[idx] = True
                        count += 1
                        # Swap maintains count, so check if it enables further improvements
                        # Do a quick greedy polish
                        old_count = count
                        count = greedy_polish(chosen, count)
                        if count > best_count:
                            best_count = count
                            best_chosen = chosen.copy()
                            no_improve = 0
                        else:
                            no_improve += 1
                    else:
                        # Revert
                        chosen[idx] = False
                        chosen[remove_idx] = True
                        count += 1
                        no_improve += 1
                else:
                    no_improve += 1
        else:
            # Try removing and finding better edges
            chosen[idx] = False
            count -= 1
            old_count = count
            count = greedy_polish(chosen, count)
            
            # Accept with simulated annealing probability
            delta = count - old_count  # net change
            if delta > 0 or random.random() < np.exp(delta / T):
                if count > best_count:
                    best_count = count
                    best_chosen = chosen.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Revert
                chosen[idx] = True
                count = old_count
                no_improve += 1
        
        T *= cooling
        
        if no_improve > max_no_improve:
            # Reset to best and restart with higher temperature
            chosen = best_chosen.copy()
            count = best_count
            T = max(T, 1.0)
            no_improve = 0
    
    # Final greedy polish
    count = greedy_polish(best_chosen, best_count)
    
    # Convert to output
    out = []
    for idx in range(m):
        if best_chosen[idx]:
            x, y = all_edges[idx]
            out.append([x, y])
    
    return out
