"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
from itertools import product

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
    
    # Also build a smaller verification matrix using just the 48 (i,j,k) entries
    # This is much faster to work with
    small_M = np.zeros((n * m * p, rank))  # 48 x 42
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = i * m * p + j * p + k
                for r in range(rank):
                    small_M[idx, r] = U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
    
    # Best result so far
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank
    
    # Strategy 1: SVD-based null space elimination (improved)
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
            
            # Adaptive tolerance based on largest singular value
            if len(s_sub) == 0:
                break
            tol_sub = max(1e-10, s_sub[0] * 1e-10)
            eff_rank_sub = int(np.sum(s_sub > tol_sub))
            
            if eff_rank_sub >= len(remaining):
                break
            
            null_sub = Vt_sub[eff_rank_sub:, :]
            
            # Find the best column to eliminate using all null vectors
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
            
            # Eliminate by adding multiples of the pivot column to others
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
    
    def eliminate_columns_direct(M_full, U_full, V_full, W_full, time_limit=None):
        """Direct linear algebra: find null-space relations via QR with column pivoting."""
        rank_full = M_full.shape[1]
        new_U = U_full.copy()
        new_V = V_full.copy()
        new_W = W_full.copy()
        eliminated = set()
        
        remaining = list(range(rank_full))
        
        while len(remaining) > 1:
            if time_limit is not None and time.time() > time_limit:
                break
            
            M_sub = M_full[:, remaining]
            n_cols = len(remaining)
            n_rows = M_sub.shape[0]
            
            try:
                # QR with column pivoting
                Q, R, P = np.linalg.qr(M_sub, mode='complete')
                # P is the permutation matrix; R is upper triangular
                # The rank is determined by the diagonal of R
                diag = np.abs(np.diag(R))
                tol = max(1e-9, diag[0] * 1e-9) if len(diag) > 0 else 1e-9
                eff_rank = int(np.sum(diag > tol))
            except Exception:
                break
            
            if eff_rank >= n_cols:
                break
            
            # Find null space vectors from R
            # R is (n_rows, n_cols), upper triangular
            # Null space: solve R[:, :eff_rank] * x = -R[:, eff_rank:] for free variables
            # Actually, for QR with pivoting, the last (n_cols - eff_rank) columns of R are zero
            # The null space basis vectors have 1 in positions eff_rank..n_cols-1
            
            # Simpler: use SVD for null space but only when needed
            try:
                U_sub, s_sub, Vt_sub = np.linalg.svd(M_sub, full_matrices=False)
            except np.linalg.LinAlgError:
                break
            
            if len(s_sub) == 0:
                break
            tol_sub = max(1e-10, s_sub[0] * 1e-10)
            eff_rank_sub = int(np.sum(s_sub > tol_sub))
            
            if eff_rank_sub >= n_cols:
                break
            
            null_sub = Vt_sub[eff_rank_sub:, :]
            
            # For each null vector, find the column with largest coefficient and eliminate it
            for nv_idx in range(null_sub.shape[0]):
                if time_limit is not None and time.time() > time_limit:
                    break
                if len(remaining) <= 1:
                    break
                
                nv = null_sub[nv_idx]
                # Find column with largest absolute value
                max_local = int(np.argmax(np.abs(nv)))
                max_global = remaining[max_local]
                pivot_val = nv[max_local]
                
                if abs(pivot_val) < 1e-8:
                    continue
                
                # Eliminate this column
                for local_c, global_c in enumerate(remaining):
                    if global_c == max_global:
                        continue
                    coeff = -nv[local_c] / pivot_val
                    if abs(coeff) > 1e-10:
                        new_U[:, global_c] += coeff * new_U[:, max_global]
                        new_V[:, global_c] += coeff * new_V[:, max_global]
                        new_W[:, global_c] += coeff * new_W[:, max_global]
                
                eliminated.add(max_global)
                remaining = [c for c in remaining if c not in eliminated]
                
                # Recompute M_sub for next iteration
                if len(remaining) <= 1:
                    break
        
        if not eliminated:
            return None, None, None, rank_full
        
        keep = [c for c in range(rank_full) if c not in eliminated]
        new_rank = len(keep)
        return new_U[:, keep], new_V[:, keep], new_W[:, keep], new_rank
    
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
    
    try:
        # Use the smaller matrix for SVD - much faster
        U_svd, s_vals, Vt_svd = np.linalg.svd(small_M, full_matrices=False)
        tol = 1e-9
        effective_rank = int(np.sum(s_vals > tol))
        
        if effective_rank < rank:
            # Greedy elimination - try multiple strategies
            # Strategy A: SVD-based on full matrix
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
            
            # Strategy B: Direct elimination with different ordering
            if time.time() < deadline - 25:
                elim_U2, elim_V2, elim_W2, elim_rank2 = eliminate_columns_direct(M, U_np, V_np, W_np,
                                                                                  time_limit=deadline - 25)
                if elim_U2 is not None and elim_rank2 < best_rank:
                    if verify_decomp_float(elim_U2, elim_V2, elim_W2, tol=1e-6):
                        best_rank = elim_rank2
                        best_result = {
                            "U": to_half_integer_list(elim_U2),
                            "V": to_half_integer_list(elim_V2),
                            "W": to_half_integer_list(elim_W2)
                        }
            
            # Strategy C: Use null space from small_M to guide elimination
            if time.time() < deadline - 20 and effective_rank < rank:
                null_small = Vt_svd[effective_rank:, :]
                # For each null vector, try to eliminate the column with largest coefficient
                new_U = U_np.copy()
                new_V = V_np.copy()
                new_W = W_np.copy()
                eliminated = set()
                
                for nv_idx in range(null_small.shape[0]):
                    if time.time() > deadline - 20:
                        break
                    if len([c for c in range(rank) if c not in eliminated]) <= 1:
                        break
                    
                    nv = null_small[nv_idx]
                    remaining = [c for c in range(rank) if c not in eliminated]
                    
                    # Find column with largest absolute value in null vector
                    max_local = int(np.argmax(np.abs(nv)))
                    max_global = remaining[max_local]
                    pivot_val = nv[max_local]
                    
                    if abs(pivot_val) < 1e-8:
                        continue
                    
                    for local_c, global_c in enumerate(remaining):
                        if global_c == max_global:
                            continue
                        coeff = -nv[local_c] / pivot_val
                        if abs(coeff) > 1e-10:
                            new_U[:, global_c] += coeff * new_U[:, max_global]
                            new_V[:, global_c] += coeff * new_V[:, max_global]
                            new_W[:, global_c] += coeff * new_W[:, max_global]
                    
                    eliminated.add(max_global)
                
                if eliminated:
                    keep = [c for c in range(rank) if c not in eliminated]
                    new_rank = len(keep)
                    final_U = new_U[:, keep]
                    final_V = new_V[:, keep]
                    final_W = new_W[:, keep]
                    
                    if verify_decomp_float(final_U, final_V, final_W, tol=1e-6) and new_rank < best_rank:
                        best_rank = new_rank
                        best_result = {
                            "U": to_half_integer_list(final_U),
                            "V": to_half_integer_list(final_V),
                            "W": to_half_integer_list(final_W)
                        }
    except Exception:
        pass
    
    # Strategy 2: Random restart elimination with different orderings
    try:
        if time.time() < deadline - 20:
            np.random.seed(42)
            best_random_rank = best_rank
            best_random_result = best_result
            
            # Use remaining time budget for random restarts
            time_per_trial = max(1.0, (deadline - 15 - time.time()) / 40)
            max_trials = 40
            
            for trial in range(max_trials):
                if time.time() > deadline - 15:
                    break
                
                new_U = U_np.copy()
                new_V = V_np.copy()
                new_W = W_np.copy()
                eliminated = set()
                
                # Randomize the order in which we consider columns for elimination
                col_order = list(range(rank))
                np.random.shuffle(col_order)
                
                for _ in range(rank - effective_rank + 5):
                    if time.time() > deadline - 15:
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
                    
                    # Pick elimination column with randomness to explore different paths
                    best_col = None
                    best_score = -1
                    best_null_idx = 0
                    
                    for nv_idx in range(null_sub.shape[0]):
                        nv = null_sub[nv_idx]
                        for local_c, global_c in enumerate(remaining):
                            score = abs(nv[local_c]) + np.random.random() * 0.1
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
    
    # Strategy 3: Try to find linear dependencies using exact rational arithmetic
    try:
        if time.time() < deadline - 10:
            from fractions import Fraction
            
            # Convert to Fraction for exact arithmetic
            def to_frac_list(arr):
                result = []
                for row in arr:
                    result.append([])
                    for val in row:
                        f = Fraction(val).limit_denominator(1000000)
                        result[-1].append(f)
                return result
            
            # Check for exact linear dependencies among the 42 columns
            # Use Gaussian elimination over Q to find the rank
            def exact_rank(mat):
                """Compute rank over Q using Gaussian elimination."""
                mat = [row[:] for row in mat]  # deep copy
                n_rows = len(mat)
                n_cols = len(mat[0]) if n_rows > 0 else 0
                rank = 0
                for col in range(n_cols):
                    # Find pivot
                    pivot_row = None
                    for row in range(rank, n_rows):
                        if mat[row][col] != 0:
                            pivot_row = row
                            break
                    if pivot_row is None:
                        continue
                    # Swap rows
                    mat[rank], mat[pivot_row] = mat[pivot_row], mat[rank]
                    # Eliminate below
                    for row in range(rank + 1, n_rows):
                        factor = mat[row][col] / mat[rank][col]
                        for c in range(col, n_cols):
                            mat[row][c] -= factor * mat[rank][c]
                    rank += 1
                return rank
            
            # Build the tensor matrix as fractions
            # This is expensive, so only do it if we have time
            if time.time() < deadline - 5:
                # Sample a subset of tensor entries to check for dependencies
                # Use all 48 (i,j,k) combinations
                tensor_entries = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            row = []
                            for r in range(rank):
                                val = U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                                row.append(Fraction(val))
                            tensor_entries.append(row)
                
                # The tensor matrix has 48 rows and 42 columns
                # Compute its rank over Q
                try:
                    tensor_rank = exact_rank(tensor_entries)
                    if tensor_rank < rank:
                        # There are linear dependencies!
                        # We need to find them and use them to reduce rank
                        # This is complex, so for now just note it
                        pass
                except Exception:
                    pass
    except Exception:
        pass
    
    return best_result
# EVOLVE-BLOCK-END
