"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # Build the rank-42 Strassen decomposition
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42
    
    def add_strassen_block(a_idx, b_idx, w_idx, start_col):
        c = start_col
        # M1 = (a00+a11)(b00+b11)
        U[a_idx[0]][c] += 1; U[a_idx[3]][c] += 1
        V[b_idx[0]][c] += 1; V[b_idx[3]][c] += 1
        W[w_idx[0]][c] += 1; W[w_idx[3]][c] += 1
        # M2 = (a10+a11)b00
        U[a_idx[2]][c+1] += 1; U[a_idx[3]][c+1] += 1
        V[b_idx[0]][c+1] += 1
        W[w_idx[2]][c+1] += 1; W[w_idx[3]][c+1] -= 1
        # M3 = a00(b01-b11)
        U[a_idx[0]][c+2] += 1
        V[b_idx[1]][c+2] += 1; V[b_idx[3]][c+2] -= 1
        W[w_idx[1]][c+2] += 1; W[w_idx[3]][c+2] += 1
        # M4 = a11(b10-b00)
        U[a_idx[3]][c+3] += 1
        V[b_idx[2]][c+3] += 1; V[b_idx[0]][c+3] -= 1
        W[w_idx[0]][c+3] += 1; W[w_idx[2]][c+3] += 1
        # M5 = (a00+a01)b11
        U[a_idx[0]][c+4] += 1; U[a_idx[1]][c+4] += 1
        V[b_idx[3]][c+4] += 1
        W[w_idx[0]][c+4] -= 1; W[w_idx[1]][c+4] += 1
        # M6 = (a10-a00)(b00+b01)
        U[a_idx[2]][c+5] += 1; U[a_idx[0]][c+5] -= 1
        V[b_idx[0]][c+5] += 1; V[b_idx[1]][c+5] += 1
        W[w_idx[3]][c+5] += 1
        # M7 = (a01-a11)(b10+b11)
        U[a_idx[1]][c+6] += 1; U[a_idx[3]][c+6] -= 1
        V[b_idx[2]][c+6] += 1; V[b_idx[3]][c+6] += 1
        W[w_idx[0]][c+6] += 1

    a_L = [0, 1, 4, 5]
    a_R = [2, 3, 6, 7]
    
    b_1_T = [0, 1, 6, 7]
    b_1_B = [12, 13, 18, 19]
    w_1 = [0, 2, 1, 3]
    
    add_strassen_block(a_L, b_1_T, w_1, 0)
    add_strassen_block(a_R, b_1_B, w_1, 7)
    
    b_2_T = [2, 3, 8, 9]
    b_2_B = [14, 15, 20, 21]
    w_2 = [4, 6, 5, 7]
    
    add_strassen_block(a_L, b_2_T, w_2, 14)
    add_strassen_block(a_R, b_2_B, w_2, 21)
    
    b_3_T = [4, 5, 10, 11]
    b_3_B = [16, 17, 22, 23]
    w_3 = [8, 10, 9, 11]
    
    add_strassen_block(a_L, b_3_T, w_3, 28)
    add_strassen_block(a_R, b_3_B, w_3, 35)
    
    # Convert to float arrays
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Precompute index arrays for fast verification
    # For each (i,j,k), we need U[i*m+j, :], V[j*p+k, :], W[k*n+i, :]
    idx_i = np.repeat(np.arange(n), m * p)
    idx_j = np.tile(np.repeat(np.arange(m), p), n)
    idx_k = np.tile(np.arange(p), n * m)
    U_idx = idx_i * m + idx_j  # index into U rows
    V_idx = idx_j * p + idx_k  # index into V rows
    W_idx = idx_k * n + idx_i  # index into W rows
    
    def verify_fast(U_arr, V_arr, W_arr, new_rank, tol=1e-7):
        """Vectorized verification: check all (i,j,k) tuples at once."""
        # U_arr: (8, new_rank), V_arr: (24, new_rank), W_arr: (12, new_rank)
        # For each term r: contribution to (i,j,k) is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Total for (i,j,k) = sum_r U[U_idx, r] * V[V_idx, r] * W[W_idx, r]
        # This is a sum of element-wise products
        # Use einsum: sum over r of U[U_idx, r] * V[V_idx, r] * W[W_idx, r]
        # Shape: U_idx has length n*m*p=48
        # U[U_idx, :] -> (48, new_rank)
        # V[V_idx, :] -> (48, new_rank)
        # W[W_idx, :] -> (48, new_rank)
        # Element-wise product, then sum along axis=1
        prod = U_arr[U_idx, :] * V_arr[V_idx, :] * W_arr[W_idx, :]
        totals = prod.sum(axis=1)
        return np.all(np.abs(totals - 1.0) <= tol)
    
    # Build the tensor matrix: column r = flattened (u_r ⊗ v_r ⊗ w_r)
    # Using einsum for speed
    # M[i*m+j, j*p+k, k*n+i, r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Flatten to (2304, rank)
    flat_terms = []
    for r in range(rank):
        term = np.outer(np.outer(U_np[:, r], V_np[:, r]), W_np[:, r])
        flat_terms.append(term.flatten())
    M = np.array(flat_terms).T  # shape (2304, 42)
    
    # Best result so far
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank
    
    def to_list(arr):
        result = []
        for row in arr:
            result.append([])
            for val in row:
                # Round to nearest half-integer
                frac = round(val * 2) / 2
                if abs(frac - round(frac)) < 1e-10:
                    frac = int(round(frac))
                result[-1].append(frac)
        return result
    
    def eliminate_columns_fast(U_in, V_in, W_in, M_in, max_eliminations, deadline, tol=1e-9, random_seed=None):
        """Eliminate dependent columns using SVD null space - optimized."""
        new_U = U_in.copy()
        new_V = V_in.copy()
        new_W = W_in.copy()
        eliminated = set()
        current_rank = M_in.shape[1]
        
        rng = np.random.RandomState(random_seed) if random_seed is not None else None
        
        for iteration in range(max_eliminations):
            if time.time() > deadline:
                break
            
            remaining = [c for c in range(current_rank) if c not in eliminated]
            if len(remaining) <= 1:
                break
            
            M_sub = M_in[:, remaining]
            
            try:
                U_sub, s_sub, Vt_sub = np.linalg.svd(M_sub, full_matrices=False)
            except np.linalg.LinAlgError:
                break
            
            eff_rank_sub = int(np.sum(s_sub > tol))
            
            if eff_rank_sub >= len(remaining):
                break
            
            null_sub = Vt_sub[eff_rank_sub:, :]
            
            # Pick the best column to eliminate - vectorized
            abs_null = np.abs(null_sub)
            if rng is not None:
                noise = rng.random_sample(null_sub.shape) * 0.05
                scores = abs_null + noise
            else:
                scores = abs_null
            
            # Find the entry with max score
            flat_idx = np.argmax(scores)
            nv_idx = flat_idx // len(remaining)
            local_c = flat_idx % len(remaining)
            best_col = remaining[local_c]
            
            if scores[nv_idx, local_c] < 1e-8:
                break
            
            nv = null_sub[nv_idx]
            pivot_val = nv[local_c]
            
            if abs(pivot_val) < 1e-10:
                continue
            
            # Eliminate: for each other column, subtract (coeff * column of best_col)
            coeffs = -nv / pivot_val
            coeffs[local_c] = 0.0
            
            # Only update columns where coeff is significant
            mask = np.abs(coeffs) > 1e-10
            if not np.any(mask):
                eliminated.add(best_col)
                continue
            
            # Update U, V, W for all remaining columns
            for local_c2, global_c2 in enumerate(remaining):
                if local_c2 == local_c:
                    continue
                if abs(coeffs[local_c2]) > 1e-10:
                    new_U[:, global_c2] += coeffs[local_c2] * new_U[:, best_col]
                    new_V[:, global_c2] += coeffs[local_c2] * new_V[:, best_col]
                    new_W[:, global_c2] += coeffs[local_c2] * new_W[:, best_col]
            
            eliminated.add(best_col)
        
        if not eliminated:
            return None, None, None, 0
        
        keep = [c for c in range(current_rank) if c not in eliminated]
        new_rank = len(keep)
        return new_U[:, keep], new_V[:, keep], new_W[:, keep], new_rank
    
    # Strategy 1: Systematic SVD-based elimination with multiple tolerances
    for tol in [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]:
        if time.time() > deadline - 30:
            break
        
        elim_U, elim_V, elim_W, elim_rank = eliminate_columns_fast(
            U_np, V_np, W_np, M, 10, deadline - 30, tol=tol
        )
        
        if elim_U is not None and elim_rank < best_rank:
            if verify_fast(elim_U, elim_V, elim_W, elim_rank, tol=1e-7):
                best_rank = elim_rank
                best_result = {
                    "U": to_list(elim_U),
                    "V": to_list(elim_V),
                    "W": to_list(elim_W)
                }
    
    # Strategy 2: Random restarts with different seeds
    if time.time() < deadline - 25:
        for seed in range(100):
            if time.time() > deadline - 25:
                break
            
            elim_U, elim_V, elim_W, elim_rank = eliminate_columns_fast(
                U_np, V_np, W_np, M, 10, deadline - 25, tol=1e-9, random_seed=seed
            )
            
            if elim_U is not None and elim_rank < best_rank:
                if verify_fast(elim_U, elim_V, elim_W, elim_rank, tol=1e-7):
                    best_rank = elim_rank
                    best_result = {
                        "U": to_list(elim_U),
                        "V": to_list(elim_V),
                        "W": to_list(elim_W)
                    }
    
    # Strategy 3: High precision check
    if time.time() < deadline - 20:
        U_h, s_h, Vt_h = np.linalg.svd(M, full_matrices=False)
        for thresh in [1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10]:
            eff_r = int(np.sum(s_h > thresh))
            if eff_r < best_rank:
                elim_U, elim_V, elim_W, elim_rank = eliminate_columns_fast(
                    U_np, V_np, W_np, M, best_rank - eff_r + 2, deadline - 20, tol=thresh
                )
                if elim_U is not None and elim_rank < best_rank:
                    if verify_fast(elim_U, elim_V, elim_W, elim_rank, tol=1e-7):
                        best_rank = elim_rank
                        best_result = {
                            "U": to_list(elim_U),
                            "V": to_list(elim_V),
                            "W": to_list(elim_W)
                        }
    
    # Strategy 4: Aggressive elimination
    if time.time() < deadline - 15:
        for tol in [1e-10, 1e-11, 1e-12]:
            if time.time() > deadline - 15:
                break
            
            elim_U, elim_V, elim_W, elim_rank = eliminate_columns_fast(
                U_np, V_np, W_np, M, 15, deadline - 15, tol=tol
            )
            
            if elim_U is not None and elim_rank < best_rank:
                if verify_fast(elim_U, elim_V, elim_W, elim_rank, tol=1e-7):
                    best_rank = elim_rank
                    best_result = {
                        "U": to_list(elim_U),
                        "V": to_list(elim_V),
                        "W": to_list(elim_W)
                    }
    
    return best_result
# EVOLVE-BLOCK-END
