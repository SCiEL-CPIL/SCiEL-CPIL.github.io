"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115
    
    # Dimensions of U, V, W
    dim_U = n * m  # 9
    dim_V = m * p  # 15
    dim_W = p * n  # 15
    
    # Build the target tensor T (scaled by 8 to allow half-integer solutions as integers)
    # T[a, b, c] = 8 if (a,b,c) corresponds to (i,j,k) else 0
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    T = np.zeros((dim_U, dim_V, dim_W), dtype=np.int32)
    idx_ijk = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 8
                idx_ijk.append((a, b, c))
                
    # Precompute indices for fast error calculation
    # We need to sum over all a, b, c.
    # Total size 9 * 15 * 15 = 2025.
    
    def compute_error(U, V, W):
        # U: (dim_U, r), V: (dim_V, r), W: (dim_W, r)
        # R[a, b, c] = sum_d U[a, d] V[b, d] W[c, d]
        # Efficiently: R = np.einsum('ad,bd,cd->abc', U, V, W)
        # This might be slow if called too often.
        # For local search, we can update error incrementally.
        # But for simplicity and correctness first, let's use einsum and see speed.
        # 2025 entries * r multiplications. If r=30, 60k ops. Very fast.
        R = np.einsum('ad,bd,cd->abc', U, V, W)
        diff = R - T
        return np.sum(diff * diff)

    def check_exact(U, V, W):
        R = np.einsum('ad,bd,cd->abc', U, V, W)
        return np.array_equal(R, T)

    # Schoolbook fallback (scaled)
    rank_sb = n * m * p
    U_sb = np.zeros((dim_U, rank_sb), dtype=np.int32)
    V_sb = np.zeros((dim_V, rank_sb), dtype=np.int32)
    W_sb = np.zeros((dim_W, rank_sb), dtype=np.int32)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_sb[a, col] = 2
                V_sb[b, col] = 2
                W_sb[c, col] = 2
                col += 1
                
    best_rank = rank_sb
    best_result = {
        "U": [[x / 2.0 for x in row] for row in U_sb],
        "V": [[x / 2.0 for x in row] for row in V_sb],
        "W": [[x / 2.0 for x in row] for row in W_sb]
    }

    # Local Search
    def local_search(r, seed, max_iters=5000):
        if time.time() > deadline - 5:
            return None
        random.seed(seed)
        np.random.seed(seed)
        
        # Initialize with small random integers
        # Range [-2, 2]
        U = np.random.randint(-2, 3, size=(dim_U, r), dtype=np.int32)
        V = np.random.randint(-2, 3, size=(dim_V, r), dtype=np.int32)
        W = np.random.randint(-2, 3, size=(dim_W, r), dtype=np.int32)
        
        # Optionally start with schoolbook-like structure for some columns
        # But random is fine for local search if we have enough iterations.
        
        current_error = compute_error(U, V, W)
        
        if current_error == 0:
            return (r, U, V, W)

        for it in range(max_iters):
            if time.time() > deadline - 5:
                return None
            
            # Pick a random matrix and entry
            mat_idx = random.randint(0, 2) # 0:U, 1:V, 2:W
            if mat_idx == 0:
                mat = U
                dim = dim_U
            elif mat_idx == 1:
                mat = V
                dim = dim_V
            else:
                mat = W
                dim = dim_W
                
            row = random.randint(0, dim - 1)
            col = random.randint(0, r - 1)
            
            old_val = mat[row, col]
            
            # Try changing to a nearby value
            # Delta in {-2, -1, 1, 2}
            delta = random.choice([-2, -1, 1, 2])
            new_val = old_val + delta
            if new_val < -4 or new_val > 4: # Bound the search space
                continue
                
            # Calculate error change
            # R[a,b,c] = sum_d ...
            # Changing mat[row, col] by delta changes R by delta * (other two factors)
            
            if mat_idx == 0:
                # U changed
                # R[row, b, c] += delta * V[b, col] * W[c, col]
                # For all b, c
                # V[:, col] is (dim_V,), W[:, col] is (dim_W,)
                # Change in R is delta * np.outer(V[:, col], W[:, col])
                # But only for a=row
                # So we update R[row, :, :]
                
                # We need to compute the change in error for R[row, :, :]
                # R_old[row, b, c] -> R_new[row, b, c] = R_old + delta * V[b, col] * W[c, col]
                
                # To avoid full recomputation, we can just recompute the error for the affected slice
                # But compute_error is fast enough?
                # Let's just recompute full error for simplicity first, if too slow, optimize.
                # 2025 entries * 30 dims = 60k mults. 5000 iters * 60k = 300M ops.
                # Python/numpy is fast, but 300M might take seconds.
                # 5000 iters * 0.01s = 50s. Might be tight.
                
                # Let's try to optimize error calculation.
                # Store R? No, R is 2025 floats/ints.
                # We can maintain R explicitly.
                
                pass # Placeholder for optimization logic
                
            # For this iteration, let's just use the simple recompute if it's fast enough
            # But let's implement the incremental update properly.
            
            # We will maintain R as a numpy array.
            # This requires passing R into the loop or making it non-local.
            # Let's refactor local_search to maintain R.
            
            break # Break out to implement proper version

        return None

    # Refactored Local Search with incremental updates
    def local_search_fast(r, seed, max_iters=20000):
        if time.time() > deadline - 5:
            return None
        random.seed(seed)
        np.random.seed(seed)
        
        U = np.random.randint(-2, 3, size=(dim_U, r), dtype=np.int32)
        V = np.random.randint(-2, 3, size=(dim_V, r), dtype=np.int32)
        W = np.random.randint(-2, 3, size=(dim_W, r), dtype=np.int32)
        
        # Compute initial R and Error
        R = np.einsum('ad,bd,cd->abc', U, V, W)
        diff = R - T
        current_error = np.sum(diff * diff)
        
        if current_error == 0:
            return (r, U, V, W)

        # Precompute column vectors for fast updates
        # When U[row, col] changes by delta:
        # R[row, :, :] += delta * np.outer(V[:, col], W[:, col])
        # Error change:
        # Let S = np.outer(V[:, col], W[:, col])
        # R_new = R + delta * S (only at row)
        # E_new = E - sum((R-T)^2) + sum((R + delta*S - T)^2)
        # Let D = R - T
        # E_new = E - sum(D[row,:]^2) + sum((D[row,:] + delta*S)^2)
        #       = E - sum(D_row^2) + sum(D_row^2 + 2*delta*D_row*S + delta^2*S^2)
        #       = E + 2*delta*sum(D_row*S) + delta^2*sum(S^2)
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                return None
            
            mat_idx = random.randint(0, 2)
            if mat_idx == 0:
                mat = U
                dim = dim_U
                # Affected R slice: R[row, :, :]
                # S = outer(V[:, col], W[:, col])
            elif mat_idx == 1:
                mat = V
                dim = dim_V
                # Affected R slice: R[:, row, :]
                # S = outer(U[:, col], W[:, col]) -> R[:, row, :] += delta * U[:, col] * W[:, col]?
                # R[a, row, c] = sum U[a,d] V[row,d] W[c,d]
                # Change V[row, col] by delta.
                # R[a, row, c] += delta * U[a, col] * W[c, col]
                # S[a, c] = U[a, col] * W[c, col]
            else:
                mat = W
                dim = dim_W
                # Affected R slice: R[:, :, row]
                # S[a, b] = U[a, col] * V[b, col]
                
            row = random.randint(0, dim - 1)
            col = random.randint(0, r - 1)
            
            delta = random.choice([-2, -1, 1, 2])
            new_val = mat[row, col] + delta
            if new_val < -4 or new_val > 4:
                continue
                
            # Calculate S
            if mat_idx == 0:
                # S is (dim_V, dim_W)
                S = np.outer(V[:, col], W[:, col])
                D_slice = R[row, :, :] - T[row, :, :]
            elif mat_idx == 1:
                # S is (dim_U, dim_W)
                S = np.outer(U[:, col], W[:, col])
                D_slice = R[:, row, :] - T[:, row, :]
            else:
                # S is (dim_U, dim_V)
                S = np.outer(U[:, col], V[:, col])
                D_slice = R[:, :, row] - T[:, :, row]
                
            # Compute error change
            # Delta_E = 2 * delta * sum(D_slice * S) + delta^2 * sum(S^2)
            sum_DS = np.sum(D_slice * S)
            sum_S2 = np.sum(S * S)
            delta_E = 2 * delta * sum_DS + delta * delta * sum_S2
            
            if delta_E < 0 or (delta_E == 0 and random.random() < 0.1):
                # Accept
                mat[row, col] = new_val
                
                # Update R
                if mat_idx == 0:
                    R[row, :, :] += delta * S
                elif mat_idx == 1:
                    R[:, row, :] += delta * S
                else:
                    R[:, :, row] += delta * S
                    
                current_error += delta_E
                
                if current_error == 0:
                    return (r, U, V, W)

        return None

    # Try ranks
    # Lower bound is ~9. Let's try from 20 upwards.
    # If we find one, we can try to go lower?
    # The prompt asks for "rank-minimal".
    # We should try to find the smallest r.
    # But local search might get stuck in local minima.
    # We'll try a range of ranks and keep the best (lowest) found.
    
    # To save time, we can start with a higher rank where solution is likely,
    # then try to reduce rank by re-optimizing with lower r?
    # Or just scan r from 20 to 44.
    
    # Given time constraints, scanning all r might be too slow.
    # Let's try a few specific ranks: 25, 28, 30, 32, 35, 38, 40, 42, 44.
    # And for each, run a few seeds.
    
    ranks_to_try = [25, 28, 30, 32, 35, 38, 40, 42, 44]
    # Also include schoolbook rank just in case
    # If we find a solution at r, we can break? No, we want minimal.
    # So we should try lower ranks first.
    
    # But if r is too low, no solution exists.
    # Let's try from 30 down to 20?
    # Or from 20 up to 44?
    # If we find a solution at r=30, we should try r=29, 28...
    
    # Strategy:
    # 1. Try r=30, 32, 35... to find *a* solution.
    # 2. Once a solution is found at r_best, try r < r_best.
    
    found_rank = None
    found_sol = None
    
    # First pass: find any solution
    for r in [35, 32, 30, 28, 25]:
        if time.time() > deadline - 10:
            break
        for seed in [42, 0, 1, 2, 3, 4, 5, 6, 7, 8]:
            if time.time() > deadline - 10:
                break
            res = local_search_fast(r, seed, max_iters=5000)
            if res is not None:
                r_found, U_f, V_f, W_f = res
                if found_rank is None or r_found < found_rank:
                    found_rank = r_found
                    found_sol = (U_f, V_f, W_f)
                break # Found one for this rank, try lower rank
    
    # Second pass: try to improve
    if found_rank is not None:
        # Try ranks lower than found_rank
        for r in range(found_rank - 1, 15, -1):
            if time.time() > deadline - 10:
                break
            for seed in [42, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                if time.time() > deadline - 10:
                    break
                res = local_search_fast(r, seed, max_iters=5000)
                if res is not None:
                    r_found, U_f, V_f, W_f = res
                    if r_found < found_rank:
                        found_rank = r_found
                        found_sol = (U_f, V_f, W_f)
                    break

    if found_sol is not None:
        U_f, V_f, W_f = found_sol
        best_rank = found_rank
        best_result = {
            "U": [[x / 2.0 for x in row] for row in U_f],
            "V": [[x / 2.0 for x in row] for row in V_f],
            "W": [[x / 2.0 for x in row] for row in W_f]
        }
        
    return best_result
# EVOLVE-BLOCK-END
