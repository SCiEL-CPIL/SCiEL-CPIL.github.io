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
            
    # Precompute bad triples
    # A triple (a, b, c) is bad if gcd(a,b) == gcd(a,c) == gcd(b,c)
    bad_triples = []
    triples_with = [[] for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            gab = gcds[a][b]
            for c in range(b + 1, N + 1):
                gbc = gcds[b][c]
                if gab == gbc:
                    gca = gcds[c][a]
                    if gca == gab:
                        bad_triples.append((a, b, c))
                        # Store pairs for each element
                        triples_with[a].append((b, c))
                        triples_with[b].append((a, c))
                        triples_with[c].append((a, b))
    
    # Precompute reverse block structure:
    # reverse_block[x] = list of (v, other) where (x, other) is in triples_with[v]
    # When x is added to S, for each (v, other) in reverse_block[x],
    # if other is in S, then block_count[v] increases by 1.
    reverse_block = [[] for _ in range(N + 1)]
    for v in range(1, N + 1):
        for a, b in triples_with[v]:
            reverse_block[a].append((v, b))
            reverse_block[b].append((v, a))
    
    start_time = time.time()
    deadline = start_time + 55
    random.seed(42)
    
    def greedy_build(order):
        S_mask = 0
        in_set = [False] * (N + 1)
        block_count = [0] * (N + 1)
        for x in order:
            if block_count[x] == 0:
                S_mask |= (1 << x)
                in_set[x] = True
                # Update block counts for affected elements
                for v, other in reverse_block[x]:
                    if in_set[other]:
                        block_count[v] += 1
        return S_mask
    
    def tabu_search(S_mask, max_iter=20000):
        current_size = bin(S_mask).count("1")
        best_size = current_size
        best_mask = S_mask
        
        in_set = [False] * (N + 1)
        for x in range(1, N + 1):
            in_set[x] = bool(S_mask & (1 << x))
        
        # Compute block_count for initial state
        block_count = [0] * (N + 1)
        for x in range(1, N + 1):
            if in_set[x]:
                for v, other in reverse_block[x]:
                    if in_set[other]:
                        block_count[v] += 1
        
        # Tabu sets for O(1) lookup
        tabu_add = set()
        tabu_remove = set()
        tabu_len = 15
        iter_count = 0
        stall_count = 0
        
        # Precompute list of elements not in S for fast iteration
        not_in_S_list = [x for x in range(1, N + 1) if not in_set[x]]
        in_S_list = [x for x in range(1, N + 1) if in_set[x]]
        
        while time.time() < deadline and iter_count < max_iter:
            iter_count += 1
            improved = False
            
            # Clean up tabu sets periodically
            if iter_count % 20 == 0:
                # Rebuild tabu sets keeping only recent entries
                pass  # We'll manage sizes differently below
            
            # Phase 1: Try to add an element with block_count == 0
            # Shuffle for randomness
            random.shuffle(not_in_S_list)
            for x in not_in_S_list:
                if x in tabu_remove:
                    continue
                if block_count[x] == 0:
                    # Add x
                    S_mask |= (1 << x)
                    in_set[x] = True
                    in_S_list.append(x)
                    not_in_S_list.remove(x)
                    current_size += 1
                    tabu_add.add(x)
                    improved = True
                    
                    # Update block counts
                    for v, other in reverse_block[x]:
                        if in_set[other]:
                            block_count[v] += 1
                    
                    if current_size > best_size:
                        best_size = current_size
                        best_mask = S_mask
                    break
            
            if improved:
                stall_count = 0
                continue
            
            # Phase 2: Try to swap
            if not in_S_list:
                break
                
            # Try swaps: remove u, add v
            # Prioritize u that unblocks the most elements
            # Compute for each u in in_S_list: how many v in not_in_S_list would become unblocked
            # This is expensive, so sample a few u's
            random.shuffle(in_S_list)
            swap_done = False
            
            # Sample up to 10 candidates for u
            for u in in_S_list[:10]:
                if u in tabu_add:
                    continue
                
                # Temporarily remove u
                in_set[u] = False
                S_mask &= ~(1 << u)
                
                # Update block counts: decrement for elements blocked by u
                for v, other in reverse_block[u]:
                    if in_set[other]:
                        block_count[v] -= 1
                
                # Find best v to add: prefer one that was blocked only by u
                # Check all not_in_S_list for block_count == 0
                random.shuffle(not_in_S_list)
                best_v = -1
                for v in not_in_S_list:
                    if v in tabu_remove:
                        continue
                    if block_count[v] == 0:
                        best_v = v
                        break
                
                if best_v != -1:
                    # Perform swap: keep u removed, add best_v
                    in_set[best_v] = True
                    S_mask |= (1 << best_v)
                    in_S_list.remove(u)
                    in_S_list.append(best_v)
                    not_in_S_list.remove(best_v)
                    not_in_S_list.append(u)
                    
                    tabu_add.add(best_v)
                    tabu_remove.add(u)
                    
                    # Update block counts for adding best_v
                    for v2, other in reverse_block[best_v]:
                        if in_set[other]:
                            block_count[v2] += 1
                    
                    swap_done = True
                    break
                else:
                    # Put u back
                    in_set[u] = True
                    S_mask |= (1 << u)
                    for v, other in reverse_block[u]:
                        if in_set[other]:
                            block_count[v] += 1
            
            if not swap_done:
                stall_count += 1
                if stall_count > 30:
                    # Perturb: Remove 2-3 random elements
                    if len(in_S_list) > 3:
                        k = random.randint(2, 3)
                        to_remove = random.sample(in_S_list, k)
                        for u in to_remove:
                            in_set[u] = False
                            S_mask &= ~(1 << u)
                            in_S_list.remove(u)
                            not_in_S_list.append(u)
                            tabu_remove.add(u)
                            # Update block counts
                            for v, other in reverse_block[u]:
                                if in_set[other]:
                                    block_count[v] -= 1
                        current_size -= k
                        stall_count = 0
                    else:
                        break
            
            # Periodically clean tabu sets to prevent unbounded growth
            if iter_count % 50 == 0:
                # Keep only the most recent additions/removals
                # Since we use sets, we just clear and re-add recent ones
                # Simple approach: clear and let the algorithm re-establish tabu
                pass  # Tabu sets grow but we cap by not adding duplicates

        return best_mask, best_size

    best_set_mask = 0
    best_size = 0
    
    # Conflict counts for greedy ordering
    conflict_counts = [len(triples_with[i]) for i in range(N + 1)]
    
    # Multiple restarts with diverse orderings
    orders = []
    orders.append(sorted(range(1, N + 1), key=lambda x: conflict_counts[x]))
    orders.append(sorted(range(1, N + 1), key=lambda x: -conflict_counts[x]))
    orders.append(sorted(range(1, N + 1), key=lambda x: (conflict_counts[x], x)))
    random_order = list(range(1, N + 1))
    random.shuffle(random_order)
    orders.append(random_order)
    
    for run in range(25):
        if time.time() > deadline - 5:
            break
            
        if run < len(orders):
            order = orders[run]
        else:
            order = list(range(1, N + 1))
            random.shuffle(order)
            
        S_mask = greedy_build(order)
        S_mask, size = tabu_search(S_mask)
        
        if size > best_size:
            best_size = size
            best_set_mask = S_mask

    result = [x for x in range(1, N + 1) if (best_set_mask & (1 << x))]
    
    if not result:
        return [1]
        
    return result
