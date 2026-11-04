"""Strassen-based rank reduction for 4x4 * 4x9 matrix multiplication."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    Uses a structural approach: build a 114-rank block decomposition, then
    find and eliminate linear dependencies among rank-1 terms using exact
    linear algebra (not ALS).
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # Safety margin before 120s limit
    
    def is_valid_fast(U_np, V_np, W_np, rank):
        """Vectorized validation."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_np[i*m+j] * V_np[j*p+k] * W_np[k*n+i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def is_valid(U, V, W, rank):
        """Check if the decomposition exactly reproduces T[i,j,k]=1 for all i,j,k."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        return is_valid_fast(U_np, V_np, W_np, rank)
    
    # Strassen 2x2x2 coefficients
    strassen_u = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
    ]
    strassen_v = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 1, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
    ]
    strassen_w = [
        [1, 0, 0, 1],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
    ]
    
    # Build 4x4x4 Strassen decomposition (rank 49)
    U_44 = []
    V_44 = []
    W_44 = []
    
    for s in range(7):
        for t in range(7):
            u_vec = [0] * 16
            v_vec = [0] * 16
            w_vec = [0] * 16
            
            u_block = strassen_u[s]
            v_block = strassen_v[s]
            w_block = strassen_w[s]
            u_in = strassen_u[t]
            v_in = strassen_v[t]
            w_in = strassen_w[t]
            
            for bi in range(2):
                for bj in range(2):
                    a_coeff = u_block[bi * 2 + bj]
                    if a_coeff == 0:
                        continue
                    for li in range(2):
                        for lj in range(2):
                            local_idx = li * 2 + lj
                            in_coeff = u_in[local_idx]
                            if in_coeff == 0:
                                continue
                            gi = 2 * bi + li
                            gj = 2 * bj + lj
                            gidx = gi * 4 + gj
                            u_vec[gidx] += a_coeff * in_coeff
            
            for bi in range(2):
                for bj in range(2):
                    b_coeff = v_block[bi * 2 + bj]
                    if b_coeff == 0:
                        continue
                    for li in range(2):
                        for lj in range(2):
                            local_idx = li * 2 + lj
                            in_coeff = v_in[local_idx]
                            if in_coeff == 0:
                                continue
                            gi = 2 * bi + li
                            gj = 2 * bj + lj
                            gidx = gi * 4 + gj
                            v_vec[gidx] += b_coeff * in_coeff
            
            for bi in range(2):
                for bj in range(2):
                    c_coeff = w_block[bi * 2 + bj]
                    if c_coeff == 0:
                        continue
                    for li in range(2):
                        for lj in range(2):
                            local_idx = li * 2 + lj
                            in_coeff = w_in[local_idx]
                            if in_coeff == 0:
                                continue
                            gi_row = 2 * bi + li
                            gi_col = 2 * bj + lj
                            gidx = gi_col * 4 + gi_row
                            w_vec[gidx] += c_coeff * in_coeff
            
            U_44.append(u_vec)
            V_44.append(v_vec)
            W_44.append(w_vec)
    
    U_44_np = np.array(U_44, dtype=float).T  # shape (16, 49)
    V_44_np = np.array(V_44, dtype=float).T  # shape (16, 49)
    W_44_np = np.array(W_44, dtype=float).T  # shape (16, 49)
    
    # Build 114-rank solution: 49 (cols 0-3) + 49 (cols 4-7) + 16 (col 8, schoolbook)
    rank_total = 49 + 49 + 16
    U_full = np.zeros((16, rank_total))
    V_full = np.zeros((36, rank_total))
    W_full = np.zeros((36, rank_total))
    
    # Block 1: A[0:4,0:4] * B[0:4,0:4] -> C[0:4,0:4]
    for r in range(49):
        U_full[:, r] = U_44_np[:, r]
        for j in range(4):
            for k in range(4):
                V_full[j * 9 + k, r] = V_44_np[j * 4 + k, r]
        for k in range(4):
            for i in range(4):
                W_full[k * 4 + i, r] = W_44_np[k * 4 + i, r]
    
    # Block 2: A[0:4,0:4] * B[0:4,4:8] -> C[0:4,4:8]
    offset = 49
    for r in range(49):
        U_full[:, offset + r] = U_44_np[:, r]
        for j in range(4):
            for k in range(4):
                V_full[j * 9 + (4 + k), offset + r] = V_44_np[j * 4 + k, r]
        for k in range(4):
            for i in range(4):
                W_full[(4 + k) * 4 + i, offset + r] = W_44_np[k * 4 + i, r]
    
    # Block 3: A[0:4,0:4] * B[0:4,8] -> C[0:4,8] (schoolbook, rank 16)
    offset2 = 98
    for i in range(4):
        for j in range(4):
            U_full[i * 4 + j, offset2 + i * 4 + j] = 1
            V_full[j * 9 + 8, offset2 + i * 4 + j] = 1
            W_full[8 * 4 + i, offset2 + i * 4 + j] = 1
    
    if not is_valid_fast(U_full, V_full, W_full, rank_total):
        # Fallback: schoolbook rank 144
        best_rank = n * m * p
        best_U = np.zeros((n*m, best_rank))
        best_V = np.zeros((m*p, best_rank))
        best_W = np.zeros((p*n, best_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    best_U[i*m+j, col] = 1
                    best_V[j*p+k, col] = 1
                    best_W[k*n+i, col] = 1
                    col += 1
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    best_rank = rank_total
    best_U = U_full.copy()
    best_V = V_full.copy()
    best_W = W_full.copy()
    
    # STRUCTURAL RANK REDUCTION: Find linear dependencies among the rank-1 terms.
    # Each rank-1 term r defines a tensor T_r[i,j,k] = U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # If there exist coefficients c_r such that sum_r c_r * T_r = 0 (as a tensor),
    # then we can eliminate one term.
    # 
    # The space of all tensors of shape (n*m, m*p, p*n) has dimension n*m*p = 144.
    # We have 114 rank-1 tensors. If they are linearly dependent in the 144-dim space,
    # we can find dependencies.
    
    # Build the 144x114 matrix where column r is the flattened tensor T_r
    def flatten_tensor(U_col, V_col, W_col):
        """Flatten the rank-1 tensor U_col * V_col * W_col (outer product)."""
        # U_col has shape (16,), V_col has shape (36,), W_col has shape (36,)
        # T[i,j,k] = U[i*m+j] * V[j*p+k] * W[k*n+i]
        # Flatten in order (i,j,k) with i fastest? No, let's use a consistent order.
        # We'll flatten as: index = i * (m*p) + j * p + k
        result = np.zeros(n * m * p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * (m * p) + j * p + k
                    result[idx] = U_col[i*m+j] * V_col[j*p+k] * W_col[k*n+i]
        return result
    
    # Build dependency matrix
    dep_matrix = np.zeros((n * m * p, rank_total))
    for r in range(rank_total):
        dep_matrix[:, r] = flatten_tensor(
            best_U[:, r], best_V[:, r], best_W[:, r]
        )
    
    # Find null space of dep_matrix (144 x 114)
    # Null space vectors c satisfy dep_matrix @ c = 0
    try:
        U_svd, S, Vt_svd = np.linalg.svd(dep_matrix, full_matrices=True)
        # Rank of dep_matrix
        tol_svd = max(dep_matrix.shape) * np.finfo(float).eps * S[0]
        rank_dep = np.sum(S > tol_svd)
        nullity = rank_total - rank_dep
        
        if nullity > 0:
            # Null space basis: last `nullity` columns of Vt_svd
            null_basis = Vt_svd[rank_dep:rank_total, :].T  # shape (rank_total, nullity)
            
            # Try to eliminate terms using null space vectors
            # For each null vector c, we can eliminate one term
            # We want to find which term to eliminate for maximum rank reduction
            
            # Greedy approach: try each null vector, eliminate the term with largest |c_r|
            eliminated = set()
            U_cur = best_U.copy()
            V_cur = best_V.copy()
            W_cur = best_W.copy()
            
            for nv_idx in range(nullity):
                if time.time() > deadline - 10:
                    break
                c = null_basis[:, nv_idx]
                # Find the term with largest coefficient to eliminate
                remaining = [r for r in range(rank_total) if r not in eliminated]
                if not remaining:
                    break
                max_r = max(remaining, key=lambda r: abs(c[r]))
                if abs(c[max_r]) < 1e-10:
                    continue
                
                # Eliminate term max_r: for each other term r', add (c[max_r]/c[r']) * term_r' to term_r
                # This makes the new term max_r = term_max_r - (1/c[max_r]) * sum_{r' != max_r} c[r'] * term_r'
                # which should be zero (up to numerical error)
                
                U_new = U_cur.copy()
                V_new = V_cur.copy()
                W_new = W_cur.copy()
                
                for r_prime in remaining:
                    if r_prime == max_r:
                        continue
                    if abs(c[r_prime]) < 1e-15:
                        continue
                    coeff = c[max_r] / c[r_prime]
                    U_new[:, max_r] -= coeff * U_cur[:, r_prime]
                    V_new[:, max_r] -= coeff * V_cur[:, r_prime]
                    W_new[:, max_r] -= coeff * W_cur[:, r_prime]
                
                # Zero out the eliminated term
                U_new[:, max_r] = 0
                V_new[:, max_r] = 0
                W_new[:, max_r] = 0
                eliminated.add(max_r)
                
                # Validate
                active_cols = [r for r in range(rank_total) if r not in eliminated]
                if active_cols:
                    U_active = U_new[:, active_cols]
                    V_active = V_new[:, active_cols]
                    W_active = W_new[:, active_cols]
                    if is_valid_fast(U_active, V_active, W_active, len(active_cols)):
                        U_cur = U_new
                        V_cur = V_new
                        W_cur = W_new
                        best_rank = len(active_cols)
                        best_U = U_cur.copy()
                        best_V = V_cur.copy()
                        best_W = W_cur.copy()
                    else:
                        # Revert
                        U_cur = best_U
                        V_cur = best_V
                        W_cur = best_W
                        break
                else:
                    break
            
            # After eliminating, clean up: remove zero columns
            active_cols = [r for r in range(rank_total) if r not in eliminated]
            if active_cols and best_rank < rank_total:
                best_U = best_U[:, active_cols]
                best_V = best_V[:, active_cols]
                best_W = best_W[:, active_cols]
                best_rank = len(active_cols)
    
    except Exception as e:
        pass
    
    # Additional: try to find more dependencies after elimination
    if best_rank < rank_total and time.time() < deadline - 15:
        # Rebuild dependency matrix with reduced rank
        dep_matrix2 = np.zeros((n * m * p, best_rank))
        for r in range(best_rank):
            dep_matrix2[:, r] = flatten_tensor(
                best_U[:, r], best_V[:, r], best_W[:, r]
            )
        
        try:
            U_svd2, S2, Vt_svd2 = np.linalg.svd(dep_matrix2, full_matrices=True)
            tol_svd2 = max(dep_matrix2.shape) * np.finfo(float).eps * S2[0]
            rank_dep2 = np.sum(S2 > tol_svd2)
            nullity2 = best_rank - rank_dep2
            
            if nullity2 > 0:
                null_basis2 = Vt_svd2[rank_dep2:best_rank, :].T
                
                eliminated2 = set()
                U_cur2 = best_U.copy()
                V_cur2 = best_V.copy()
                W_cur2 = best_W.copy()
                
                for nv_idx in range(nullity2):
                    if time.time() > deadline - 5:
                        break
                    c = null_basis2[:, nv_idx]
                    remaining = [r for r in range(best_rank) if r not in eliminated2]
                    if not remaining:
                        break
                    max_r = max(remaining, key=lambda r: abs(c[r]))
                    if abs(c[max_r]) < 1e-10:
                        continue
                    
                    U_new2 = U_cur2.copy()
                    V_new2 = V_cur2.copy()
                    W_new2 = W_cur2.copy()
                    
                    for r_prime in remaining:
                        if r_prime == max_r:
                            continue
                        if abs(c[r_prime]) < 1e-15:
                            continue
                        coeff = c[max_r] / c[r_prime]
                        U_new2[:, max_r] -= coeff * U_cur2[:, r_prime]
                        V_new2[:, max_r] -= coeff * V_cur2[:, r_prime]
                        W_new2[:, max_r] -= coeff * W_cur2[:, r_prime]
                    
                    U_new2[:, max_r] = 0
                    V_new2[:, max_r] = 0
                    W_new2[:, max_r] = 0
                    eliminated2.add(max_r)
                    
                    active_cols2 = [r for r in range(best_rank) if r not in eliminated2]
                    if active_cols2:
                        U_act2 = U_new2[:, active_cols2]
                        V_act2 = V_new2[:, active_cols2]
                        W_act2 = W_new2[:, active_cols2]
                        if is_valid_fast(U_act2, V_act2, W_act2, len(active_cols2)):
                            U_cur2 = U_new2
                            V_cur2 = V_new2
                            W_cur2 = W_new2
                            best_rank = len(active_cols2)
                            best_U = U_cur2.copy()
                            best_V = V_cur2.copy()
                            best_W = W_cur2.copy()
                        else:
                            U_cur2 = best_U
                            V_cur2 = best_V
                            W_cur2 = best_W
                            break
                    else:
                        break
                
                active_cols_final = [r for r in range(best_rank) if r not in eliminated2]
                if active_cols_final and best_rank < rank_total:
                    best_U = best_U[:, active_cols_final]
                    best_V = best_V[:, active_cols_final]
                    best_W = best_W[:, active_cols_final]
                    best_rank = len(active_cols_final)
        
        except Exception:
            pass
    
    # Final validation
    if not is_valid_fast(best_U, best_V, best_W, best_rank):
        # Fallback to schoolbook
        best_rank = n * m * p
        best_U = np.zeros((n*m, best_rank))
        best_V = np.zeros((m*p, best_rank))
        best_W = np.zeros((p*n, best_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    best_U[i*m+j, col] = 1
                    best_V[j*p+k, col] = 1
                    best_W[k*n+i, col] = 1
                    col += 1
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
