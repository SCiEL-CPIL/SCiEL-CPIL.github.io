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
    
    def build_small_M(U_mat, V_mat, W_mat, remaining_cols):
        """Build the 48 x len(remaining_cols) verification matrix."""
        small = np.zeros((n * m * p, len(remaining_cols)))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    for li, gc in enumerate(remaining_cols):
                        small[idx, li] = U_mat[i*m+j, gc] * V_mat[j*p+k, gc] * W_mat[k*n+i, gc]
        return small
    
    def eliminate_one_column(U_mat, V_mat, W_mat, remaining_cols, time_limit=None):
        """Try to eliminate one column from remaining_cols using null space of the 48-entry tensor.
        Returns (new_U, new_V, new_W, eliminated_col) or (None, None, None, None) if no elimination possible."""
        if time_limit is not None and time.time() > time_limit:
            return None, None, None, None
        
        if len(remaining_cols) <= 1:
            return None, None, None, None
        
        M_sub = build_small_M(U_mat, V_mat, W_mat, remaining_cols)
        n_cols = len(remaining_cols)
        
        try:
            U_sub, s_sub, Vt_sub = np.linalg.svd(M_sub, full_matrices=False)
        except np.linalg.LinAlgError:
            return None, None, None, None
        
        if len(s_sub) == 0:
            return None, None, None, None
        tol_sub = max(1e-10, s_sub[0] * 1e-10)
        eff_rank_sub = int(np.sum(s_sub > tol_sub))
        
        if eff_rank_sub >= n_cols:
            return None, None, None, None
        
        null_sub = Vt_sub[eff_rank_sub:, :]
        
        # Find the best null vector and column to eliminate
        best_col = None
        best_score = -1
        best_null_idx = 0
        
        for nv_idx in range(null_sub.shape[0]):
            nv = null_sub[nv_idx]
            for local_c, global_c in enumerate(remaining_cols):
                score = abs(nv[local_c])
                if score > best_score and score > 1e-8:
                    best_score = score
                    best_col = global_c
                    best_null_idx = nv_idx
        
        if best_col is None:
            return None, None, None, None
        
        nv = null_sub[best_null_idx]
        local_best = remaining_cols.index(best_col)
        pivot_val = nv[local_best]
        
        if abs(pivot_val) < 1e-10:
            return None, None, None, None
        
        # Eliminate by adding multiples of the pivot column to others
        for local_c, global_c in enumerate(remaining_cols):
            if global_c == best_col:
                continue
            coeff = -nv[local_c] / pivot_val
            if abs(coeff) > 1e-10:
                U_mat[:, global_c] += coeff * U_mat[:, best_col]
                V_mat[:, global_c] += coeff * V_mat[:, best_col]
                W_mat[:, global_c] += coeff * W_mat[:, best_col]
        
        return U_mat, V_mat, W_mat, best_col
    
    # Strategy 1: Iterative elimination
    try:
        new_U = U_np.copy()
        new_V = V_np.copy()
        new_W = W_np.copy()
        eliminated = set()
        remaining = list(range(rank))
        
        max_elim_attempts = 20
        for attempt in range(max_elim_attempts):
            if time.time() > deadline - 30:
                break
            if len(remaining) <= 1:
                break
            
            new_U, new_V, new_W, elim_col = eliminate_one_column(
                new_U, new_V, new_W, remaining, time_limit=deadline - 30)
            
            if elim_col is None:
                break
            
            eliminated.add(elim_col)
            remaining = [c for c in remaining if c != elim_col]
        
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
            
            max_trials = 50
            for trial in range(max_trials):
                if time.time() > deadline - 10:
                    break
                
                new_U = U_np.copy()
                new_V = V_np.copy()
                new_W = W_np.copy()
                eliminated = set()
                
                # Randomize the order in which we consider columns for elimination
                col_order = list(range(rank))
                np.random.shuffle(col_order)
                remaining = list(col_order)
                
                max_elim_attempts = 20
                for _ in range(max_elim_attempts):
                    if time.time() > deadline - 10:
                        break
                    if len(remaining) <= 1:
                        break
                    
                    new_U, new_V, new_W, elim_col = eliminate_one_column(
                        new_U, new_V, new_W, remaining, time_limit=deadline - 10)
                    
                    if elim_col is None:
                        break
                    
                    eliminated.add(elim_col)
                    remaining = [c for c in remaining if c != elim_col]
                
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
    
    # Strategy 3: Try a different base construction with more redundancy
    # Use a rank-48 schoolbook decomposition and try to reduce it
    try:
        if time.time() < deadline - 5:
            rank2 = n * m * p  # 48
            U2 = [[0] * rank2 for _ in range(n * m)]
            V2 = [[0] * rank2 for _ in range(m * p)]
            W2 = [[0] * rank2 for _ in range(p * n)]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        r = i * m * p + j * p + k
                        U2[i*m+j][r] = 1
                        V2[j*p+k][r] = 1
                        W2[k*n+i][r] = 1
            
            U2_np = np.array(U2, dtype=float)
            V2_np = np.array(V2, dtype=float)
            W2_np = np.array(W2, dtype=float)
            
            # Try to reduce rank-48 to rank-42 or lower
            new_U2 = U2_np.copy()
            new_V2 = V2_np.copy()
            new_W2 = W2_np.copy()
            eliminated2 = set()
            remaining2 = list(range(rank2))
            
            max_elim_attempts2 = 15
            for _ in range(max_elim_attempts2):
                if time.time() > deadline - 3:
                    break
                if len(remaining2) <= 42:  # Don't go below 42 from this start
                    break
                
                new_U2, new_V2, new_W2, elim_col2 = eliminate_one_column(
                    new_U2, new_V2, new_W2, remaining2, time_limit=deadline - 3)
                
                if elim_col2 is None:
                    break
                
                eliminated2.add(elim_col2)
                remaining2 = [c for c in remaining2 if c != elim_col2]
            
            if eliminated2:
                keep2 = [c for c in range(rank2) if c not in eliminated2]
                new_rank2 = len(keep2)
                final_U2 = new_U2[:, keep2]
                final_V2 = new_V2[:, keep2]
                final_W2 = new_W2[:, keep2]
                
                if verify_decomp_float(final_U2, final_V2, final_W2, tol=1e-6) and new_rank2 < best_rank:
                    best_rank = new_rank2
                    best_result = {
                        "U": to_half_integer_list(final_U2),
                        "V": to_half_integer_list(final_V2),
                        "W": to_half_integer_list(final_W2)
                    }
    except Exception:
        pass
    
    return best_result
# EVOLVE-BLOCK-END
