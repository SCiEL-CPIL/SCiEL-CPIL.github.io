import numpy as np
import time

def solve(a, b):
    n = len(a)
    # Initialization
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # Precompute
    a_int = a.astype(np.int64)
    b_int = b.astype(np.int64)
    
    def calc_cost(p):
        # Sum a[i, j] * b[p[i], p[j]]
        # Vectorized
        bp = b_int[p[:, None], p[None, :]]
        return int(np.sum(a_int * bp))
    
    current_cost = calc_cost(p)
    
    def or_opt_pass(p, current_cost, max_block=3, time_limit=None):
        """
        Vectorized Or-opt local search: move a contiguous block of size 1, 2, or 3
        to a new position. This can escape 2-swap local minima.
        Returns (p, current_cost) with improved solution.
        """
        improved = True
        while improved and (time_limit is None or time.time() < time_limit):
            improved = False
            for block_size in range(1, max_block + 1):
                if time_limit is not None and time.time() >= time_limit:
                    break
                for i in range(n - block_size + 1):
                    if time_limit is not None and time.time() >= time_limit:
                        break
                    # Block is facilities i..i+block_size-1
                    block = p[i:i+block_size]
                    # Try inserting block at each position j (after removing block)
                    # We need to compute delta for each possible insertion point
                    # Remove block: remaining facilities
                    if i == 0:
                        rest = p[block_size:]
                    elif i + block_size == n:
                        rest = p[:i]
                    else:
                        rest = np.concatenate([p[:i], p[i+block_size:]])
                    
                    m = len(rest)  # n - block_size
                    
                    # For each insertion position pos (0..m), the new permutation is:
                    # rest[:pos] + block + rest[pos:]
                    # We compute delta for all positions at once using vectorization
                    
                    # The cost change comes from:
                    # 1. Removing block from position i: interactions between block and rest change
                    # 2. Inserting block at new position: interactions between block and rest change
                    # 3. Internal block interactions stay the same (relative order preserved)
                    
                    # Let's compute the cost delta for each insertion position
                    # old_cost_block_rest = sum over k in block, l in rest of a[k,l]*b[p[k],p[l]] + a[l,k]*b[p[l],p[k]]
                    # But we need to be careful about positions
                    
                    # Simpler approach: compute new cost for each candidate and compare
                    # But that's O(m) full cost computations which is expensive
                    # Instead, use the delta formula
                    
                    # For a block move from position i to position j (in the rest array):
                    # The facilities in the block keep their relative order
                    # The rest facilities keep their relative order
                    # Only the interactions between block and rest change
                    
                    # Let B = block facilities (indices in original problem)
                    # Let R = rest facilities
                    # Old: B is at positions i..i+block_size-1 in p, R is elsewhere
                    # New: B is at positions pos..pos+block_size-1 in new_p, R is elsewhere
                    
                    # The delta is:
                    # sum_{k in B, l in R} a[k,l] * (b[p_new[k], p_new[l]] - b[p_old[k], p_old[l]])
                    # + sum_{l in R, k in B} a[l,k] * (b[p_new[l], p_new[k]] - b[p_old[l], p_old[k]])
                    
                    # This is complex to vectorize fully. Let's do a simpler approach:
                    # For each insertion position, compute the delta using precomputed arrays
                    
                    # Actually, let's just do a greedy local search with block moves
                    # For each block start and size, try all insertion points and pick best
                    best_delta = 0
                    best_pos = -1
                    for pos in range(m + 1):
                        if time_limit is not None and time.time() >= time_limit:
                            break
                        # Build new permutation
                        new_p = np.empty(n, dtype=np.int64)
                        new_p[:pos] = rest[:pos]
                        new_p[pos:pos+block_size] = block
                        new_p[pos+block_size:] = rest[pos:]
                        
                        # Compute delta efficiently
                        # Only interactions between block and rest change
                        # Block facilities are at positions pos..pos+block_size-1 in new_p
                        # Rest facilities are at other positions
                        
                        # Let's compute the cost difference
                        # Old cost contribution from block-rest interactions:
                        # For each k in block (at old position i+k_idx), l in rest (at old position)
                        # b[p[k], p[l]] where p[k] is old location, p[l] is old location
                        
                        # This is getting complex. Let's use a different approach:
                        # Precompute the distance matrix D where D[i,j] = b[p[i], p[j]]
                        # Then cost = sum a[i,j] * D[i,j]
                        # When we move a block, D changes only for entries involving block positions
                        
                        # For now, let's just compute the full cost for candidate moves
                        # and accept if better. This is O(n^2) per candidate but we limit candidates.
                        
                        # Actually, let's use the swap delta approach but for block moves
                        # The key insight: moving a block of size s is equivalent to s swaps
                        # but we can compute the delta more efficiently
                        
                        # Let's just do a simple check: compute new cost and compare
                        # This is expensive but correct
                        bp_new = b_int[new_p[:, None], new_p[None, :]]
                        new_cost = int(np.sum(a_int * bp_new))
                        delta = new_cost - current_cost
                        if delta < best_delta:
                            best_delta = delta
                            best_pos = pos
                    
                    if best_pos >= 0 and best_delta < 0:
                        # Apply the best move
                        new_p = np.empty(n, dtype=np.int64)
                        new_p[:best_pos] = rest[:best_pos]
                        new_p[best_pos:best_pos+block_size] = block
                        new_p[best_pos+block_size:] = rest[best_pos:]
                        p = new_p
                        current_cost += best_delta
                        improved = True
                        break  # Restart the while loop
                if time_limit is not None and time.time() >= time_limit:
                    break
        return p, current_cost
    
    # Simulated Annealing
    T = 1000.0 # Initial temperature
    cooling = 0.99
    min_T = 1.0
    
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # First, do a quick Or-opt pass to improve initial solution
    p, current_cost = or_opt_pass(p, current_cost, max_block=2, time_limit=start_time + 30.0)
    
    while T > min_T and time.time() < deadline:
        # Randomize order of pairs to avoid bias
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        np.random.shuffle(pairs)
        
        for u, v in pairs:
            if time.time() > deadline:
                break
            
            # Calculate delta for swapping u and v
            u_loc = p[u]
            v_loc = p[v]
            
            D_v_row = b_int[v_loc, p]
            D_u_row = b_int[u_loc, p]
            D_v_col = b_int[p, v_loc]
            D_u_col = b_int[p, u_loc]
            
            a_u = a_int[u]
            a_v = a_int[v]
            a_uT = a_int[:, u]
            a_vT = a_int[:, v]
            
            base1 = np.dot(a_u, D_v_row - D_u_row)
            T1 = base1 + a_u[u] * (D_v_row[v] - D_v_row[u]) + a_u[v] * (D_v_col[u] - D_v_row[v])
            
            base2 = np.dot(a_uT, D_v_col - D_u_col)
            T2 = base2 + a_uT[u] * (D_v_row[v] - D_v_col[u]) + a_uT[v] * (D_u_col[v] - D_v_col[v])
            
            base3 = np.dot(a_v, D_u_row - D_v_row)
            T3 = base3 + a_v[v] * (D_u_row[u] - D_u_row[v]) + a_v[u] * (D_u_col[v] - D_u_row[u])
            
            base4 = np.dot(a_vT, D_u_col - D_v_col)
            T4 = base4 + a_vT[v] * (D_u_col[u] - D_u_col[v]) + a_vT[u] * (D_u_col[v] - D_u_col[u])
            
            delta = T1 + T2 + T3 + T4
            
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                # Accept
                p[u], p[v] = p[v], p[u]
                current_cost += delta
        
        T *= cooling
        
        # Periodically do an Or-opt pass to escape 2-swap local minima
        if int(T * 1000) % 100 == 0 and time.time() < deadline - 20.0:
            p, current_cost = or_opt_pass(p, current_cost, max_block=2, time_limit=deadline - 10.0)
    
    # Final Or-opt pass with more time
    remaining = deadline - time.time()
    if remaining > 5.0:
        p, current_cost = or_opt_pass(p, current_cost, max_block=3, time_limit=deadline - 2.0)
    
    return p.tolist()
