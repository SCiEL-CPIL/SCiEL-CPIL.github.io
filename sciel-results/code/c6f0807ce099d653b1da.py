"""
Rank-36 exact tensor decomposition for 2x4 by 4x5 matrix multiplication.
Uses Strassen for 2x2x2 blocks and schoolbook for 2x2x1 blocks.
Includes local search for rank reduction via linear dependency detection.
"""

import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 140

def _build_strassen_36():
    n, m, p = 2, 4, 5
    rank = 36

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    def add_strassen_222(i0, j0, k0, col_start):
        nonlocal col
        i0i = i0
        i1i = i0 + 1
        j0i = j0
        j1i = j0 + 1
        k0i = k0
        k1i = k0 + 1

        # M1 = (A00 + A11) * (B00 + B11)
        c = col_start
        U[i0i * m + j0i][c] += 1
        U[i1i * m + j1i][c] += 1
        V[j0i * p + k0i][c] += 1
        V[j1i * p + k1i][c] += 1
        W[k0i * n + i0i][c] += 1
        W[k1i * n + i1i][c] += 1
        col += 1

        # M2 = (A10 + A11) * B00
        c = col_start + 1
        U[i1i * m + j0i][c] += 1
        U[i1i * m + j1i][c] += 1
        V[j0i * p + k0i][c] += 1
        W[k0i * n + i1i][c] += 1
        W[k1i * n + i1i][c] -= 1
        col += 1

        # M3 = A00 * (B01 - B11)
        c = col_start + 2
        U[i0i * m + j0i][c] += 1
        V[j0i * p + k1i][c] += 1
        V[j1i * p + k1i][c] -= 1
        W[k1i * n + i0i][c] += 1
        W[k1i * n + i1i][c] += 1
        col += 1

        # M4 = A11 * (B10 - B00)
        c = col_start + 3
        U[i1i * m + j1i][c] += 1
        V[j1i * p + k0i][c] += 1
        V[j0i * p + k0i][c] -= 1
        W[k0i * n + i0i][c] += 1
        W[k0i * n + i1i][c] += 1
        col += 1

        # M5 = (A00 + A01) * B11
        c = col_start + 4
        U[i0i * m + j0i][c] += 1
        U[i0i * m + j1i][c] += 1
        V[j1i * p + k1i][c] += 1
        W[k0i * n + i0i][c] -= 1
        W[k1i * n + i0i][c] += 1
        col += 1

        # M6 = (A10 - A00) * (B00 + B01)
        c = col_start + 5
        U[i1i * m + j0i][c] += 1
        U[i0i * m + j0i][c] -= 1
        V[j0i * p + k0i][c] += 1
        V[j0i * p + k1i][c] += 1
        W[k1i * n + i1i][c] += 1
        col += 1

        # M7 = (A01 - A11) * (B10 + B11)
        c = col_start + 6
        U[i0i * m + j1i][c] += 1
        U[i1i * m + j1i][c] -= 1
        V[j1i * p + k0i][c] += 1
        V[j1i * p + k1i][c] += 1
        W[k0i * n + i0i][c] += 1
        col += 1

        return col_start + 7

    def add_schoolbook_2x2x1(i0, j0, k0, col_start):
        c = col_start
        for di in range(2):
            for dj in range(2):
                U[(i0 + di) * m + (j0 + dj)][c] = 1
                V[(j0 + dj) * p + k0][c] = 1
                W[k0 * n + (i0 + di)][c] = 1
                c += 1
        return c

    # Four Strassen blocks for the 2x4 by 4x4 part
    col = add_strassen_222(0, 0, 0, col)
    col = add_strassen_222(0, 2, 0, col)
    col = add_strassen_222(0, 0, 2, col)
    col = add_strassen_222(0, 2, 2, col)

    # Two schoolbook blocks for the 2x4 by 4x1 part
    col = add_schoolbook_2x2x1(0, 0, 4, col)
    col = add_schoolbook_2x2x1(0, 2, 4, col)

    assert col == rank, f"Expected rank {rank}, got {col}"

    return U, V, W


def _verify_decomposition(U, V, W, n=2, m=4, p=5):
    """Verify that the tensor decomposition is exact."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    rank = U_arr.shape[1]

    # Build the multiplication tensor T[i,j,k] = delta_{i,k} (in appropriate form)
    # The decomposition requires: for all i,j,k: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = T[i,j,k]
    # where T[i,j,k] = 1 if the (i,j) entry of A times (j,k) entry of B contributes to (i,k) entry of C
    # Actually, the tensor for matrix mult: T[i,j,k] = 1 for all valid i,j,k (it's the identity in the right sense)
    
    # More precisely: C[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor form: T[i,j,k] = 1 for all i,j,k (the multiplication tensor has all entries 1)
    # Wait, no. The multiplication tensor M has M[i,j,k] = 1 if and only if the product A[i,j]*B[j,k] 
    # appears in C[i,k]. For standard matrix multiplication, every triple (i,j,k) with valid indices
    # contributes, so M[i,j,k] = 1 for all valid i,j,k.
    
    # The decomposition says: M[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Note the trace convention: W[k*n+i, r] so W index is (k, i)
    
    # Let's verify: for each (i, j, k), check sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] == 1
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                if abs(s - 1.0) > max_err:
                    max_err = abs(s - 1.0)
    
    return max_err < 1e-9


def _try_reduce_rank(U, V, W, n=2, m=4, p=5, deadline=None):
    """
    Attempt to reduce the rank by finding linear dependencies among columns.
    Returns (U', V', W', new_rank) if successful, else (U, V, W, rank).
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    current_rank = U_arr.shape[1]
    best_U, best_V, best_W = U, V, W
    best_rank = current_rank
    
    if deadline is None:
        deadline = _DEADLINE
    
    # Strategy: try to find a column that is a linear combination of others
    # For each column r, check if removing it and adjusting others preserves the decomposition
    
    for target_col in range(current_rank):
        if time.time() > deadline - 5:
            break
        
        # Try to express column target_col as combination of other columns
        # The tensor constraint: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for all i,j,k
        # If we remove column r0, we need to add corrections to remaining columns
        
        # Simplified approach: check if the column is "redundant" by checking
        # if the 3-tensor formed by outer products has rank < current_rank
        
        # Build the flattening matrices
        # Flatten U: shape (n*m, rank)
        # Flatten V: shape (m*p, rank)  
        # Flatten W: shape (p*n, rank)
        
        # Check linear independence of the set of rank-1 tensors U_col ⊗ V_col ⊗ W_col
        # by checking if any column can be expressed as a linear combination
        
        # For efficiency, check if the matrix [U[:, r] for r != target_col] spans U[:, target_col]
        # in a way that's consistent with V and W
        
        # Simple heuristic: try pairwise column elimination
        # Check if column target_col is approximately a scalar multiple of another column
        for other_col in range(current_rank):
            if other_col == target_col:
                continue
            if time.time() > deadline - 5:
                break
            
            # Check if U[:, target_col] is proportional to U[:, other_col]
            U_tc = U_arr[:, target_col]
            U_oc = U_arr[:, other_col]
            
            if np.allclose(U_tc, 0) and np.allclose(U_oc, 0):
                continue
            
            # Check proportionality
            if np.any(np.abs(U_oc) > 1e-10):
                ratio_candidates = []
                for idx in range(len(U_tc)):
                    if abs(U_oc[idx]) > 1e-10:
                        ratio_candidates.append(U_tc[idx] / U_oc[idx])
                
                if ratio_candidates:
                    ratio = ratio_candidates[0]
                    if all(abs(rc - ratio) < 1e-9 for rc in ratio_candidates):
                        # Check V and W also proportional
                        V_tc = V_arr[:, target_col]
                        V_oc = V_arr[:, other_col]
                        W_tc = W_arr[:, target_col]
                        W_oc = W_arr[:, other_col]
                        
                        v_proportional = True
                        if np.any(np.abs(V_oc) > 1e-10):
                            v_ratios = []
                            for idx in range(len(V_tc)):
                                if abs(V_oc[idx]) > 1e-10:
                                    v_ratios.append(V_tc[idx] / V_oc[idx])
                            if v_ratios and not all(abs(vr - ratio) < 1e-9 for vr in v_ratios):
                                v_proportional = False
                        
                        w_proportional = True
                        if np.any(np.abs(W_oc) > 1e-10):
                            w_ratios = []
                            for idx in range(len(W_tc)):
                                if abs(W_oc[idx]) > 1e-10:
                                    w_ratios.append(W_tc[idx] / W_oc[idx])
                            if w_ratios and not all(abs(wr - ratio) < 1e-9 for wr in w_ratios):
                                w_proportional = False
                        
                        if v_proportional and w_proportional:
                            # Columns are proportional - can merge
                            # Remove target_col and scale other_col
                            new_rank = current_rank - 1
                            keep_cols = [c for c in range(current_rank) if c != target_col]
                            
                            new_U = np.zeros((U_arr.shape[0], new_rank))
                            new_V = np.zeros((V_arr.shape[0], new_rank))
                            new_W = np.zeros((W_arr.shape[0], new_rank))
                            
                            for new_idx, old_idx in enumerate(keep_cols):
                                if old_idx == other_col:
                                    new_U[:, new_idx] = U_arr[:, old_idx] * (1 + ratio)
                                    new_V[:, new_idx] = V_arr[:, old_idx]
                                    new_W[:, new_idx] = W_arr[:, old_idx]
                                else:
                                    new_U[:, new_idx] = U_arr[:, old_idx]
                                    new_V[:, new_idx] = V_arr[:, old_idx]
                                    new_W[:, new_idx] = W_arr[:, old_idx]
                            
                            # Verify
                            if _verify_decomposition(new_U.tolist(), new_V.tolist(), new_W.tolist()):
                                best_U = new_U.tolist()
                                best_V = new_V.tolist()
                                best_W = new_W.tolist()
                                best_rank = new_rank
                                U_arr = new_U
                                V_arr = new_V
                                W_arr = new_W
                                current_rank = new_rank
                                break
    
    return best_U, best_V, best_W, best_rank


def construct():
    n, m, p = 2, 4, 5
    
    # Build the base rank-36 solution
    U, V, W = _build_strassen_36()
    
    # Verify the base solution
    if not _verify_decomposition(U, V, W):
        raise ValueError("Base Strassen decomposition failed verification")
    
    # Try to reduce rank using local search
    remaining = _DEADLINE - time.time()
    if remaining > 10:
        U, V, W, rank = _try_reduce_rank(U, V, W, n, m, p, deadline=_DEADLINE - 5)
    else:
        rank = len(U[0])
    
    if time.time() > _DEADLINE - 2:
        raise TimeoutError("Deadline exceeded")
    
    return {"U": U, "V": V, "W": W}
