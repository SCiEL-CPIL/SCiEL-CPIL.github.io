import random
import time
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 40s margin for safety/overhead
    
    # Target n values to try, in descending order
    target_ns = [17, 16, 15, 14]
    
    # Fallback result: A known valid coloring for n=9 or similar?
    # Let's use a simple greedy or random for small n as fallback.
    # Actually, the task says "best known colouring has 17 vertices".
    # If we fail to find 17, we should return the best we found.
    # Let's initialize with a trivial valid coloring for n=1 (or just return empty if none found? No, must be valid).
    # A random coloring of K_9 is likely valid?
    # R_4(C6) > 9 is very likely.
    # Let's generate a random coloring for n=10 and verify.
    
    def verify(n, colors):
        # Fast verification for small n
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        edge_index = {e: i for i, e in enumerate(edges)}
        # Check all 6-cycles
        for v_set in itertools.combinations(range(n), 6):
            # Check if any permutation forms a mono cycle
            # Optimized: Just check if the induced subgraph on these 6 vertices contains a mono C6.
            # This is hard to do without enumerating cycles.
            # For n <= 15, we can just enumerate the 30 cycles per set.
            v0 = v_set[0]
            seen = set()
            for perm in itertools.permutations(v_set[1:]):
                if perm[0] > perm[4]: continue
                key = tuple(sorted([
                    edge_index[(min(v0, perm[0]), max(v0, perm[0]))],
                    edge_index[(min(perm[0], perm[1]), max(perm[0], perm[1]))],
                    edge_index[(min(perm[1], perm[2]), max(perm[1], perm[2]))],
                    edge_index[(min(perm[2], perm[3]), max(perm[2], perm[3]))],
                    edge_index[(min(perm[3], perm[4]), max(perm[3], perm[4]))],
                    edge_index[(min(perm[4], v0), max(perm[4], v0))]
                ]))
                if key in seen: continue
                seen.add(key)
                c = colors[key[0]]
                if all(colors[e] == c for e in key):
                    return False
        return True

    # Generate a fallback for n=10
    fallback_n = 10
    fallback_colors = None
    for _ in range(100):
        c = [random.randint(0, 3) for _ in range(fallback_n * (fallback_n - 1) // 2)]
        if verify(fallback_n, c):
            fallback_colors = c
            break
    if fallback_colors is None:
        # If even n=10 fails (unlikely), return n=1?
        # R_4(C6) is definitely > 1.
        fallback_colors = [0]
        fallback_n = 2 # K2 has no C6.
        
    best_result = {'n': fallback_n, 'colors': fallback_colors}
    
    for n in target_ns:
        if time.time() > deadline - 20: # Need time for search
            break
            
        # Precompute cycles for K_n
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        edge_index = {e: i for i, e in enumerate(edges)}
        num_edges = len(edges)
        
        cycles = []
        edge_to_cycles = [[] for _ in range(num_edges)]
        
        # Time guard for cycle generation
        gen_start = time.time()
        try:
            for v_set in itertools.combinations(range(n), 6):
                if time.time() - gen_start > 15:
                    raise TimeoutError("Cycle gen too slow")
                    
                v0 = v_set[0]
                # Generate 30 unique cycles for this set
                seen_keys = set()
                for perm in itertools.permutations(v_set[1:]):
                    if perm[0] > perm[4]:
                        continue
                    e1 = (min(v0, perm[0]), max(v0, perm[0]))
                    e2 = (min(perm[0], perm[1]), max(perm[0], perm[1]))
                    e3 = (min(perm[1], perm[2]), max(perm[1], perm[2]))
                    e4 = (min(perm[2], perm[3]), max(perm[2], perm[3]))
                    e5 = (min(perm[3], perm[4]), max(perm[3], perm[4]))
                    e6 = (min(perm[4], v0), max(perm[4], v0))
                    
                    idxs = [edge_index[e1], edge_index[e2], edge_index[e3], edge_index[e4], edge_index[e5], edge_index[e6]]
                    key = tuple(sorted(idxs))
                    
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    
                    cycles.append(idxs)
                    for idx in idxs:
                        edge_to_cycles[idx].append(len(cycles) - 1)
        except TimeoutError:
            continue # Skip this n, try smaller
        
        num_cycles = len(cycles)
        
        # Local Search: Simulated Annealing
        def try_solve(n):
            if time.time() > deadline - 5:
                return None
            
            # Initialize random
            colors = np.random.randint(0, 4, size=num_edges, dtype=np.int8)
            
            # Calculate initial violations
            # cycle_counts[c_idx][color]
            # Using a list of lists of ints is slow.
            # Let's use a 2D numpy array? 
            # num_cycles x 4.
            # For n=17, 370k x 4 = 1.5M ints. 6MB. Okay.
            cycle_counts = np.zeros((num_cycles, 4), dtype=np.int8)
            
            # Init counts
            for c_idx, c_edges in enumerate(cycles):
                for e_idx in c_edges:
                    cycle_counts[c_idx, colors[e_idx]] += 1
            
            violations = np.sum(cycle_counts[:, 0] == 6) + np.sum(cycle_counts[:, 1] == 6) + np.sum(cycle_counts[:, 2] == 6) + np.sum(cycle_counts[:, 3] == 6)
            
            if violations == 0:
                return colors.tolist()
            
            T = 10.0
            T_min = 0.01
            cooling = 0.9995
            max_iters = 50000
            iters = 0
            
            # Precompute list of edges for faster random access?
            # Just use random.randint
            
            while iters < max_iters:
                if time.time() > deadline - 5:
                    break
                iters += 1
                
                if violations == 0:
                    return colors.tolist()
                
                e_idx = random.randint(0, num_edges - 1)
                old_color = colors[e_idx]
                
                best_new_color = -1
                best_delta = 0
                current_v = violations
                
                c_indices = edge_to_cycles[e_idx]
                
                if not c_indices:
                    continue
                
                for new_color in range(4):
                    if new_color == old_color:
                        continue
                    
                    delta_v = 0
                    # Simulate
                    # We need to check which cycles change status
                    # A cycle is a violation if count[c] == 6.
                    # Changing edge e from old to new:
                    # For each cycle in c_indices:
                    #   old_count_old = cycle_counts[c_idx, old_color]
                    #   old_count_new = cycle_counts[c_idx, new_color]
                    #   
                    #   was_viol = (old_count_old == 6)
                    #   
                    #   new_count_old = old_count_old - 1
                    #   new_count_new = old_count_new + 1
                    #   
                    #   is_viol = (new_count_new == 6)
                    #   
                    #   if was_viol and not is_viol: delta_v -= 1
                    #   elif not was_viol and is_viol: delta_v += 1
                    
                    for c_idx in c_indices:
                        oc = cycle_counts[c_idx, old_color]
                        nc = cycle_counts[c_idx, new_color]
                        
                        was_viol = (oc == 6)
                        
                        n_oc = oc - 1
                        n_nc = nc + 1
                        is_viol = (n_nc == 6)
                        
                        if was_viol and not is_viol:
                            delta_v -= 1
                        elif not was_viol and is_viol:
                            delta_v += 1
                        
                    if delta_v < best_delta:
                        best_delta = delta_v
                        best_new_color = new_color
                
                # Acceptance
                if best_new_color != -1:
                    if best_delta < 0:
                        accept = True
                    else:
                        # Simulated Annealing acceptance
                        # P = exp(-delta / T)
                        if T > T_min:
                            r = random.random()
                            if r < np.exp(-best_delta / T):
                                accept = True
                            else:
                                accept = False
                        else:
                            accept = False
                    
                    if accept:
                        # Apply
                        colors[e_idx] = best_new_color
                        for c_idx in c_indices:
                            cycle_counts[c_idx, old_color] -= 1
                            cycle_counts[c_idx, best_new_color] += 1
                        violations += best_delta
                        T *= cooling
                else:
                    T *= cooling

            return None
        
        solution = try_solve(n)
        if solution:
            return {'n': n, 'colors': solution}
    
    return best_result
