# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Attempt to find a rank-minimal exact tensor decomposition for 4x4 matrix multiplication.
    Starts from recursive Strassen (rank 49) and attempts local search for lower rank.
    Falls back to rank 49 if search doesn't improve.
    """
    start_time = time.time()
    DEADLINE = start_time + 140  # 10s safety margin
    
    n, m, p = 4, 4, 4
    np_dim = n * m  # 16
    
    # Build the 4x4 matrix multiplication tensor as a 3-tensor T[i,j,k] = delta[i+k,j] for 4x4*4x4->4x4
    # Actually: C[i,j] = sum_k A[i,k]*B[k,j]
    # Tensor T[a,b,c] where a indexes (i,k) of A, b indexes (k,j) of B, c indexes (i,j) of C
    # T[(i,k),(k,j),(i,j)] = 1
    
    # Build T as a 3D array of shape (16, 16, 16)
    T = np.zeros((16, 16, 16), dtype=np.int64)
    for i in range(n):
        for j in range(p):
            for k in range(m):
                T[i * m + k, k * p + j, i * n + j] = 1
    
    # Build rank-49 Strassen decomposition
    def build_strassen_49():
        sU = [
            [(1, 0, 0), (1, 1, 1)],
            [(1, 1, 0), (1, 1, 1)],
            [(1, 0, 0)],
            [(1, 1, 1)],
            [(1, 0, 0), (1, 0, 1)],
            [(1, 1, 0), (-1, 0, 0)],
            [(1, 0, 1), (-1, 1, 1)]
        ]
        sV = [
            [(1, 0, 0), (1, 1, 1)],
            [(1, 0, 0)],
            [(1, 0, 1), (-1, 1, 1)],
            [(1, 1, 0), (-1, 0, 0)],
            [(1, 1, 1)],
            [(1, 0, 0), (1, 0, 1)],
            [(1, 1, 0), (1, 1, 1)]
        ]
        sW = [
            [(1, 0, 0), (1, 1, 1)],
            [(1, 1, 0), (-1, 1, 1)],
            [(1, 0, 1), (1, 1, 1)],
            [(1, 0, 0), (1, 1, 0)],
            [(-1, 0, 0), (1, 0, 1)],
            [(1, 1, 1)],
            [(1, 0, 0)]
        ]
        
        rank = 49
        U = np.zeros((16, rank), dtype=np.int64)
        V = np.zeros((16, rank), dtype=np.int64)
        W = np.zeros((16, rank), dtype=np.int64)
        
        col = 0
        for i in range(7):
            for j in range(7):
                for (ci, ri, ci_col) in sU[i]:
                    for (cj, rj, cj_col) in sU[j]:
                        gr = 2*ri + rj
                        gc = 2*ci_col + cj_col
                        U[gr*m + gc, col] += ci * cj
                for (ci, ri, ci_col) in sV[i]:
                    for (cj, rj, cj_col) in sV[j]:
                        gr = 2*ri + rj
                        gc = 2*ci_col + cj_col
                        V[gr*p + gc, col] += ci * cj
                for (ci, ri, ci_col) in sW[i]:
                    for (cj, rj, cj_col) in sW[j]:
                        gr = 2*ri + rj
                        gc = 2*ci_col + cj_col
                        W[gc*n + gr, col] += ci * cj
                col += 1
        return U, V, W, rank
    
    U, V, W, best_rank = build_strassen_49()
    
    # Verify the decomposition
    def verify(U, V, W, rank):
        # Check that sum_r U[:,r] * V[:,r]^T * W[:,r]^T = T
        # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
        for i in range(16):
            for j in range(16):
                for k in range(16):
                    s = sum(U[i, r] * V[j, r] * W[k, r] for r in range(rank))
                    if s != T[i, j, k]:
                        return False
        return True
    
    # Verify initial
    if not verify(U, V, W, best_rank):
        # Something went wrong, use a simple rank-64 identity decomposition
        U = np.zeros((16, 64), dtype=np.int64)
        V = np.zeros((16, 64), dtype=np.int64)
        W = np.zeros((16, 64), dtype=np.int64)
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    r = i * m * p + k * p + j
                    U[i * m + k, r] = 1
                    V[k * p + j, r] = 1
                    W[i * n + j, r] = 1
        best_rank = 64
    
    # Local search: try to eliminate one tensor slice at a time
    # If column r can be expressed as a linear combination of others with integer/half-integer coefficients,
    # we can potentially reduce rank.
    
    def try_eliminate_column(U, V, W, rank, target_col, deadline):
        """Try to find integer/half-integer coefficients to express column target_col as combo of others."""
        if time.time() > deadline:
            return None, None, None, None
        
        other_cols = [c for c in range(rank) if c != target_col]
        
        # Check if the column vectors U[:,target], V[:,target], W[:,target] 
        # are in the span (over Q) of the other columns simultaneously
        # This is hard, so use a heuristic: try small integer combinations
        
        # Try to find small integers a_c such that:
        # U[:,t] = sum_c a_c * U[:,c], V[:,t] = sum_c a_c * V[:,c], W[:,t] = sum_c a_c * W[:,c]
        # for all c in other_cols
        
        # This is an overdetermined system. Use least squares to find a candidate, then round.
        
        # Build matrix: for each other column c, we have 3*16 = 48 equations
        # A @ a = b where A has rows from U,V,W columns and b is target column vectors
        
        A_mat = np.zeros((16 * 3, len(other_cols)), dtype=np.int64)
        b_vec = np.zeros(16 * 3, dtype=np.int64)
        
        for idx, c in enumerate(other_cols):
            A_mat[0:16, idx] = U[:, c]
            A_mat[16:32, idx] = V[:, c]
            A_mat[32:48, idx] = W[:, c]
        
        b_vec[0:16] = U[:, target_col]
        b_vec[16:32] = V[:, target_col]
        b_vec[32:48] = W[:, target_col]
        
        # Solve least squares
        try:
            sol, residuals, rank_ls, sv = np.linalg.lstsq(A_mat.astype(float), b_vec.astype(float), rcond=None)
        except:
            return None, None, None, None
        
        # Check if solution is close to integer/half-integer
        # Try rounding to nearest integer
        sol_round = np.round(sol)
        
        # Check if A @ sol_round ≈ b
        residual = np.abs(A_mat @ sol_round - b_vec).max()
        
        if residual < 1e-9:
            # Success! We can express target_col as combination of others
            # Remove target_col and merge
            new_rank = rank - 1
            new_U = np.zeros((16, new_rank), dtype=np.int64)
            new_V = np.zeros((16, new_rank), dtype=np.int64)
            new_W = np.zeros((16, new_rank), dtype=np.int64)
            
            # Copy other columns, adding contribution from target
            for new_c in range(new_rank):
                if new_c < target_col:
                    old_c = new_c
                else:
                    old_c = new_c + 1
                new_U[:, new_c] = U[:, old_c]
                new_V[:, new_c] = V[:, old_c]
                new_W[:, new_c] = W[:, old_c]
            
            # Add scaled target to the appropriate columns
            for idx, c in enumerate(other_cols):
                coeff = sol_round[idx]
                if coeff != 0:
                    if c < target_col:
                        new_c = c
                    else:
                        new_c = c - 1
                    new_U[:, new_c] += coeff * U[:, target_col]
                    new_V[:, new_c] += coeff * V[:, target_col]
                    new_W[:, new_c] += coeff * W[:, target_col]
            
            return new_U, new_V, new_W, new_rank
        
        # Try half-integers: multiply by 2 and round
        sol_half = np.round(sol * 2) / 2.0
        residual_half = np.abs(A_mat @ (sol_half * 2) - b_vec * 2).max() / 2.0
        
        if residual_half < 1e-9:
            # Success with half-integers
            # We need to handle the scaling
            # Actually, if coefficients are half-integers, we can't directly eliminate
            # because the other columns would need to be scaled too
            # For simplicity, skip half-integer case
            pass
        
        return None, None, None, None
    
    # Iteratively try to eliminate columns
    current_U, current_V, current_W, current_rank = U.copy(), V.copy(), W.copy(), best_rank
    improved = True
    max_iterations = 200
    
    while improved and current_rank > 25 and time.time() < DEADLINE:
        improved = False
        for target in range(current_rank):
            if time.time() > DEADLINE:
                break
            new_U, new_V, new_W, new_rank = try_eliminate_column(
                current_U, current_V, current_W, current_rank, target, DEADLINE
            )
            if new_rank is not None:
                # Verify the new decomposition
                if verify(new_U, new_V, new_W, new_rank):
                    current_U, current_V, current_W, current_rank = new_U, new_V, new_W, new_rank
                    best_rank = min(best_rank, current_rank)
                    improved = True
                    break
    
    # Return the best found
    U_out = current_U.tolist()
    V_out = current_V.tolist()
    W_out = current_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
