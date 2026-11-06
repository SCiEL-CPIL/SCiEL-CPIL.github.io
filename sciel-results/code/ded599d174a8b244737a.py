"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s deadline, 5s safety margin for overhead/return
    
    N = 256
    
    # Build conflict sets
    squares = set()
    s = 1
    while s * s <= 2 * N:
        squares.add(s * s)
        s += 1
    
    # conflict[a] = set of b in [1,N] such that a+b is a perfect square
    conflict = [set() for _ in range(N + 1)]
    for a in range(1, N + 1):
        for sq in squares:
            b = sq - a
            if 1 <= b <= N:
                conflict[a].add(b)
    
    # Numbers where 2a is a perfect square are forbidden (self-conflict)
    forbidden = set()
    for a in range(1, N + 1):
        if 2 * a in squares:
            forbidden.add(a)
    
    valid_numbers = [x for x in range(1, N + 1) if x not in forbidden]
    
    # Use bitmask for speed
    M = len(valid_numbers)
    num_to_idx = {x: i for i, x in enumerate(valid_numbers)}
    idx_to_num = valid_numbers
    
    # Build adjacency as bitmasks
    adj = [0] * M
    for i, a in enumerate(valid_numbers):
        for b in conflict[a]:
            if b in num_to_idx:
                j = num_to_idx[b]
                if j != i:  # exclude self
                    adj[i] |= (1 << j)
    
    # Precompute popcount lookup for bytes to speed up counting
    popcount_table = [bin(i).count('1') for i in range(256)]
    
    def popcount(x):
        """Fast popcount using table lookup."""
        count = 0
        while x:
            count += popcount_table[x & 0xFF]
            x >>= 8
        return count
    
    # Compute degrees
    degrees = [popcount(adj[i]) for i in range(M)]
    
    def greedy_init(order):
        selected = 0
        for i in order:
            if (adj[i] & selected) == 0:
                selected |= (1 << i)
        return selected
    
    # Initial solution: Greedy with lowest degree first
    degree_order = sorted(range(M), key=lambda i: degrees[i])
    best_mask = greedy_init(degree_order)
    best_size = popcount(best_mask)
    
    # Simulated Annealing with faster operations
    def simulated_annealing(start_mask, start_size, steps, t_start, t_end):
        current_mask = start_mask
        current_size = start_size
        t = t_start
        cooling_rate = (t_end / t_start) ** (1.0 / max(steps, 1))
        
        # Precompute list of vertices sorted by degree for biased sampling
        # Lower degree vertices are more likely to be addable
        vert_list = list(range(M))
        
        for step in range(steps):
            if time.time() > deadline:
                break
            
            # Pick a random vertex - bias toward lower degree for adds
            v = random.randint(0, M - 1)
            
            if (current_mask >> v) & 1:
                # v is in set, try removing it
                trial_mask = current_mask & ~(1 << v)
                trial_size = current_size - 1
                delta_e = 1
            else:
                # v is not in set, try adding it
                neighbors_in_set = adj[v] & current_mask
                if neighbors_in_set == 0:
                    trial_mask = current_mask | (1 << v)
                    trial_size = current_size + 1
                    delta_e = -1
                else:
                    n_count = popcount(neighbors_in_set)
                    if n_count == 1:
                        # Single neighbor swap: remove neighbor, add v
                        # Find the single neighbor
                        nb_idx = (neighbors_in_set & -neighbors_in_set).bit_length() - 1
                        trial_mask = (current_mask & ~(1 << nb_idx)) | (1 << v)
                        trial_size = current_size
                        delta_e = 0
                    else:
                        # Remove all neighbors, add v
                        trial_mask = (current_mask & ~neighbors_in_set) | (1 << v)
                        trial_size = current_size - n_count + 1
                        delta_e = n_count - 1
            
            # Acceptance criterion
            if delta_e <= 0:
                current_mask = trial_mask
                current_size = trial_size
            else:
                # Accept with probability exp(-delta_e / t)
                if t > 0 and random.random() < math.exp(-delta_e / t):
                    current_mask = trial_mask
                    current_size = trial_size
            
            t *= cooling_rate
            
        return current_mask, current_size
    
    # Exact local search polish: try all single additions, removals, and swaps
    def local_polish(mask, size, max_iterations=10):
        """Iteratively try to improve by single vertex changes and 2-swaps."""
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            improved = False
            # Try adding each non-member vertex
            for v in range(M):
                if (mask >> v) & 1:
                    continue
                neighbors = adj[v] & mask
                if neighbors == 0:
                    # Can add v freely
                    mask |= (1 << v)
                    size += 1
                    improved = True
                else:
                    n_count = popcount(neighbors)
                    if n_count == 1:
                        # Swap: remove one neighbor, add v
                        nb_idx = (neighbors & -neighbors).bit_length() - 1
                        mask = (mask & ~(1 << nb_idx)) | (1 << v)
                        # size unchanged
                        improved = True
                    elif n_count == 2:
                        # 2-swap: remove two neighbors, add v
                        # Find the two neighbors
                        nb1 = (neighbors & -neighbors).bit_length() - 1
                        neighbors2 = neighbors & ~(1 << nb1)
                        nb2 = (neighbors2 & -neighbors2).bit_length() - 1
                        # Check if removing both neighbors allows adding v
                        # and doesn't create new conflicts
                        new_mask = (mask & ~(1 << nb1)) & ~(1 << nb2) | (1 << v)
                        # Verify no conflicts
                        if (adj[v] & new_mask) == 0:
                            mask = new_mask
                            size += 1 - 2  # add 1, remove 2
                            improved = True
            # Try 2-swap: remove one vertex, add two non-vertices
            for u in range(M):
                if time.time() > deadline:
                    break
                if not ((mask >> u) & 1):
                    continue
                # Try removing u
                mask_without_u = mask & ~(1 << u)
                # Find vertices that can be added after removing u
                for v in range(M):
                    if (mask_without_u >> v) & 1:
                        continue
                    neighbors_v = adj[v] & mask_without_u
                    if neighbors_v == 0:
                        # Can add v after removing u
                        mask = mask_without_u | (1 << v)
                        size = size - 1 + 1  # size unchanged
                        improved = True
                        break
                    elif popcount(neighbors_v) == 1:
                        # Check if the single neighbor is u (which we're removing)
                        if neighbors_v == (1 << u):
                            # This shouldn't happen since we already removed u
                            pass
                        else:
                            # The neighbor is some other vertex, so this doesn't help
                            pass
                if improved:
                    break
            if not improved:
                break
        return mask, size
    
    # Run multiple SA runs with different initializations and temperatures
    random.seed(42)
    
    # 1. SA from greedy lowest degree
    mask, size = simulated_annealing(best_mask, best_size, 40000, 50.0, 0.1)
    if size > best_size:
        best_mask = mask
        best_size = size

    # 2. SA from random greedy restarts
    for restart in range(20):
        if time.time() > deadline - 8:
            break
        random_order = list(range(M))
        random.shuffle(random_order)
        init_mask = greedy_init(random_order)
        init_size = popcount(init_mask)
        
        # Use higher temp for exploration
        mask, size = simulated_annealing(init_mask, init_size, 25000, 80.0, 0.1)
        if size > best_size:
            best_mask = mask
            best_size = size

    # 3. Final polish: SA with low temp from best found
    if time.time() < deadline - 5:
        mask, size = simulated_annealing(best_mask, best_size, 15000, 10.0, 0.01)
        if size > best_size:
            best_mask = mask
            best_size = size
    
    # 4. Exact local search polish
    if time.time() < deadline - 2:
        best_mask, best_size = local_polish(best_mask, best_size)
    
    # 5. One more SA pass after polish to escape any local optima
    if time.time() < deadline - 2:
        mask, size = simulated_annealing(best_mask, best_size, 10000, 5.0, 0.01)
        if size > best_size:
            best_mask = mask
            best_size = size
        # Polish again
        if time.time() < deadline - 1:
            best_mask, best_size = local_polish(best_mask, best_size)
    
    # 6. Final aggressive polish with more iterations
    if time.time() < deadline - 1:
        best_mask, best_size = local_polish(best_mask, best_size, max_iterations=20)

    # Convert back to actual numbers
    result = [idx_to_num[i] for i in range(M) if (best_mask >> i) & 1]
    
    if not result:
        result = [x for x in range(1, N + 1, 3) if x not in forbidden]
    
    return sorted(result)
# EVOLVE-BLOCK-END
