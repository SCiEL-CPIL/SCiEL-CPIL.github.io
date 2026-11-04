"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
from itertools import product, combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # Build the matrix multiplication tensor T[i,j,k] for C[i,k] = sum_j A[i,j]*B[j,k]
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # Convention: W[k*n + i, r] contributes to C[i,k]
    # So T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
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
    
    # Verify the decomposition
    def verify_decomp(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if total != 1:
                        return False
        return True
    
    if not verify_decomp(U, V, W, rank):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i*m+j][r] = 1
                    V[j*p+k][r] = 1
                    W[k*n+i][r] = 1
        return {"U": U, "V": V, "W": W}
    
    # Convert to float arrays
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Build the tensor matrix: column r = flattened (u_r ⊗ v_r ⊗ w_r)
    flat_terms = []
    for r in range(rank):
        term = np.outer(np.outer(U_np[:, r], V_np[:, r]), W_np[:, r])
        flat_terms.append(term.flatten())
    M = np.array(flat_terms).T  # shape (2304, 42)
    
    # Best result so far
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank
    
    def verify_decomp_float(U_arr, V_arr, W_arr, tol=1e-6):
        """Verify decomposition with floating point arrays."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(U_arr.shape[1]):
                        total += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                    if abs(total - 1.0) > tol:
                        return False
        return True
    
    def to_half_integer_list(arr):
        """Convert float array to list of int/half-integer values."""
        result = []
        for row in arr:
            result.append([])
            for val in row:
                frac = round(val * 2) / 2
                if abs(frac - round(frac)) < 1e-10:
                    frac = int(round(frac))
                result[-1].append(frac)
        return result
    
    # Strategy 1: SVD-based null space elimination
    def eliminate_columns(M_full, U_full, V_full, W_full, max_iter=50, time_limit=None):
        """Iteratively eliminate columns using SVD null space."""
        rank_full = M_full.shape[1]
        new_U = U_full.copy()
        new_V = V_full.copy()
        new_W = W_full.copy()
        eliminated = set()
        
        for iteration in range(max_iter):
            if time_limit is not None and time.time() > time_limit:
                break
                
            remaining = [c for c in range(rank_full) if c not in eliminated]
            if len(remaining) <= 1:
                break
            
            M_sub = M_full[:, remaining]
            
            try:
                U_sub, s_sub, Vt_sub = np.linalg.svd(M_sub, full_matrices=False)
            except np.linalg.LinAlgError:
                break
            
            if len(s_sub) == 0:
                break
            tol_sub = max(1e-10, s_sub[0] * 1e-10)
            eff_rank_sub = int(np.sum(s_sub > tol_sub))
            
            if eff_rank_sub >= len(remaining):
                break
            
            null_sub = Vt_sub[eff_rank_sub:, :]
            
            best_col = None
            best_score = -1
            best_null_idx = 0
            
            for nv_idx in range(null_sub.shape[0]):
                nv = null_sub[nv_idx]
                for local_c, global_c in enumerate(remaining):
                    score = abs(nv[local_c])
                    if score > best_score and score > 1e-8:
                        best_score = score
                        best_col = global_c
                        best_null_idx = nv_idx
            
            if best_col is None:
                break
            
            nv = null_sub[best_null_idx]
            local_best = remaining.index(best_col)
            pivot_val = nv[local_best]
            
            if abs(pivot_val) < 1e-10:
                continue
            
            for local_c, global_c in enumerate(remaining):
                if global_c == best_col:
                    continue
                coeff = -nv[local_c] / pivot_val
                if abs(coeff) > 1e-10:
                    new_U[:, global_c] += coeff * new_U[:, best_col]
                    new_V[:, global_c] += coeff * new_V[:, best_col]
                    new_W[:, global_c] += coeff * new_W[:, best_col]
            
            eliminated.add(best_col)
        
        if not eliminated:
            return None, None, None, rank_full
        
        keep = [c for c in range(rank_full) if c not in eliminated]
        new_rank = len(keep)
        return new_U[:, keep], new_V[:, keep], new_W[:, keep], new_rank
    
    try:
        U_svd, s_vals, Vt_svd = np.linalg.svd(M, full_matrices=False)
        tol = 1e-9
        effective_rank = int(np.sum(s_vals > tol))
        
        if effective_rank < rank:
            elim_U, elim_V, elim_W, elim_rank = eliminate_columns(M, U_np, V_np, W_np, 
                                                                    max_iter=rank - effective_rank + 10,
                                                                    time_limit=deadline - 30)
            if elim_U is not None and elim_rank < best_rank:
                if verify_decomp_float(elim_U, elim_V, elim_W, tol=1e-6):
                    best_rank = elim_rank
                    best_result = {
                        "U": to_half_integer_list(elim_U),
                        "V": to_half_integer_list(elim_V),
                        "W": to_half_integer_list(elim_W)
                    }
    except Exception:
        pass
    
    # Strategy 2: Support pruning - enumerate sparse rank-40 submatrices
    # For each pair of columns to remove, check if the remaining 40 columns
    # span the same tensor space (i.e., the removed columns are in the span)
    try:
        if time.time() < deadline - 20:
            np.random.seed(123)
            target_rank = best_rank - 2  # Try rank 40 first
            
            # Precompute: for each column, check if it can be expressed as
            # linear combination of others using the full tensor matrix
            # M has shape (n*m*p, rank) = (48, 42)
            
            # Enumerate pairs of columns to remove
            # For each pair (i,j), check if M[:,i] and M[:,j] are in the
            # span of the remaining columns
            found_rank40 = False
            
            # Use QR with column pivoting to find which columns are "redundant"
            # More efficient: check all pairs systematically
            for pair in combinations(range(rank), 2):
                if time.time() > deadline - 15:
                    break
                
                i, j = pair
                keep_cols = [c for c in range(rank) if c not in (i, j)]
                
                # Check if columns i and j are in the span of keep_cols
                M_keep = M[:, keep_cols]  # (48, 40)
                M_test = np.column_stack([M[:, i], M[:, j]])  # (48, 2)
                
                # Check rank(M_keep) vs rank(M_keep + M_test)
                # If rank doesn't increase, the test columns are in the span
                try:
                    rank_keep = np.linalg.matrix_rank(M_keep, tol=1e-8)
                    rank_combined = np.linalg.matrix_rank(np.column_stack([M_keep, M_test]), tol=1e-8)
                    
                    if rank_combined == rank_keep:
                        # Columns i and j are in the span of keep_cols
                        # This means we can potentially eliminate them
                        # But we need to verify the actual decomposition works
                        # with the original U, V, W restricted to keep_cols
                        # (since the tensor entries must still sum to 1)
                        
                        # The key insight: if M[:,i] is in span(M[:,keep]),
                        # then there exist coefficients alpha such that
                        # M[:,i] = sum_k alpha_k * M[:,keep_k]
                        # This means u_i ⊗ v_i ⊗ w_i = sum_k alpha_k * u_k ⊗ v_k ⊗ w_k
                        # which implies the tensor decomposition still works
                        # when we remove column i (and similarly for j)
                        
                        # Verify by checking the actual tensor entries
                        U_test = U_np[:, keep_cols]
                        V_test = V_np[:, keep_cols]
                        W_test = W_np[:, keep_cols]
                        
                        if verify_decomp_float(U_test, V_test, W_test, tol=1e-6):
                            new_rank = len(keep_cols)
                            if new_rank < best_rank:
                                best_rank = new_rank
                                best_result = {
                                    "U": to_half_integer_list(U_test),
                                    "V": to_half_integer_list(V_test),
                                    "W": to_half_integer_list(W_test)
                                }
                                found_rank40 = True
                                break
                except np.linalg.LinAlgError:
                    continue
            
            # If we found rank 40, try to find rank 39
            if found_rank40 and time.time() < deadline - 15:
                target_rank39 = 39
                # Rebuild with the new best result
                U_np2 = np.array(best_result["U"], dtype=float)
                V_np2 = np.array(best_result["V"], dtype=float)
                W_np2 = np.array(best_result["W"], dtype=float)
                rank2 = U_np2.shape[1]
                
                # Build new tensor matrix
                flat_terms2 = []
                for r in range(rank2):
                    term = np.outer(np.outer(U_np2[:, r], V_np2[:, r]), W_np2[:, r])
                    flat_terms2.append(term.flatten())
                M2 = np.array(flat_terms2).T
                
                for pair in combinations(range(rank2), 2):
                    if time.time() > deadline - 15:
                        break
                    
                    i, j = pair
                    keep_cols = [c for c in range(rank2) if c not in (i, j)]
                    
                    M_keep = M2[:, keep_cols]
                    M_test = np.column_stack([M2[:, i], M2[:, j]])
                    
                    try:
                        rank_keep = np.linalg.matrix_rank(M_keep, tol=1e-8)
                        rank_combined = np.linalg.matrix_rank(np.column_stack([M_keep, M_test]), tol=1e-8)
                        
                        if rank_combined == rank_keep:
                            U_test = U_np2[:, keep_cols]
                            V_test = V_np2[:, keep_cols]
                            W_test = W_np2[:, keep_cols]
                            
                            if verify_decomp_float(U_test, V_test, W_test, tol=1e-6):
                                new_rank = len(keep_cols)
                                if new_rank < best_rank:
                                    best_rank = new_rank
                                    best_result = {
                                        "U": to_half_integer_list(U_test),
                                        "V": to_half_integer_list(V_test),
                                        "W": to_half_integer_list(W_test)
                                    }
                                    break
                    except np.linalg.LinAlgError:
                        continue
    except Exception:
        pass
    
    # Strategy 3: Random restart elimination (reduced budget since Strategy 2 is primary)
    try:
        if time.time() < deadline - 10:
            np.random.seed(42)
            best_random_rank = best_rank
            best_random_result = best_result
            
            max_trials = 10
            for trial in range(max_trials):
                if time.time() > deadline - 10:
                    break
                
                new_U = U_np.copy()
                new_V = V_np.copy()
                new_W = W_np.copy()
                eliminated = set()
                
                col_order = list(range(rank))
                np.random.shuffle(col_order)
                
                for _ in range(5):
                    if time.time() > deadline - 10:
                        break
                    
                    remaining = [c for c in col_order if c not in eliminated]
                    if len(remaining) <= 1:
                        break
                    
                    M_sub = M[:, remaining]
                    
                    try:
                        U_sub, s_sub, Vt_sub = np.linalg.svd(M_sub, full_matrices=False)
                    except np.linalg.LinAlgError:
                        break
                    
                    if len(s_sub) == 0:
                        break
                    tol_sub = max(1e-10, s_sub[0] * 1e-10)
                    eff_rank_sub = int(np.sum(s_sub > tol_sub))
                    
                    if eff_rank_sub >= len(remaining):
                        break
                    
                    null_sub = Vt_sub[eff_rank_sub:, :]
                    
                    best_col = None
                    best_score = -1
                    best_null_idx = 0
                    
                    for nv_idx in range(null_sub.shape[0]):
                        nv = null_sub[nv_idx]
                        for local_c, global_c in enumerate(remaining):
                            score = abs(nv[local_c]) + np.random.random() * 0.05
                            if score > best_score and abs(nv[local_c]) > 1e-8:
                                best_score = score
                                best_col = global_c
                                best_null_idx = nv_idx
                    
                    if best_col is None:
                        break
                    
                    nv = null_sub[best_null_idx]
                    local_best = remaining.index(best_col)
                    pivot_val = nv[local_best]
                    
                    if abs(pivot_val) < 1e-10:
                        continue
                    
                    for local_c, global_c in enumerate(remaining):
                        if global_c == best_col:
                            continue
                        coeff = -nv[local_c] / pivot_val
                        if abs(coeff) > 1e-10:
                            new_U[:, global_c] += coeff * new_U[:, best_col]
                            new_V[:, global_c] += coeff * new_V[:, best_col]
                            new_W[:, global_c] += coeff * new_W[:, best_col]
                    
                    eliminated.add(best_col)
                
                if eliminated:
                    keep = [c for c in range(rank) if c not in eliminated]
                    new_rank = len(keep)
                    final_U = new_U[:, keep]
                    final_V = new_V[:, keep]
                    final_W = new_W[:, keep]
                    
                    if verify_decomp_float(final_U, final_V, final_W, tol=1e-6) and new_rank < best_random_rank:
                        best_random_rank = new_rank
                        best_random_result = {
                            "U": to_half_integer_list(final_U),
                            "V": to_half_integer_list(final_V),
                            "W": to_half_integer_list(final_W)
                        }
            
            if best_random_rank < best_rank:
                best_rank = best_random_rank
                best_result = best_random_result
    except Exception:
        pass
    
    return best_result
# EVOLVE-BLOCK-END
