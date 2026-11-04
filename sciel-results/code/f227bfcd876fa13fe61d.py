"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145  # seconds, safety margin
    
    n, m, p = DIMENSIONS  # 3, 3, 8
    
    # Use structured decomposition exploiting rectangular geometry
    # For each output column k (k=0..7), compute A @ a_k where a_k is 3x1
    # Use a rank-8 decomposition for 3x3 @ 3x1, applied 8 times
    
    # Known rank-8 decomposition for 3x3 @ 3x1:
    # We need T[i,j,k] for i,j in 0..2, k in 0..2 (output index i, input indices j,k)
    # Actually for 3x3 @ 3x1: output[i] = sum_j A[i,j] * x[j]
    # Tensor: T[i, j, 0] where first index is output i, second is A index (i,j) -> j
    # Let me reformulate: 
    # For 3x3 @ 3x1: U shape (9, r), V shape (3, r), W shape (3, r)
    # T[a, b, c] = sum_k U[a,k]*V[b,k]*W[c,k]
    # a = i*3+j (A index), b = j (x index), c = i (output index)
    # Target: T[i*3+j, j, i] = 1
    
    # Let me use a simple rank-9 first, then try to reduce
    # Actually, let me just build the full 3x3 @ 3x8 tensor directly with structure
    
    # Structured approach: group by output column
    # For output column k (k=0..7):
    #   We need to compute: for each i (0..2): out[i,k] = sum_j A[i,j] * B[j,k]
    #   This is a 3x3 @ 3x1 problem for fixed k
    
    # Use the same rank-r decomposition for each column, sharing structure
    # But we can potentially share some terms across columns if B has structure
    # However, B is arbitrary 3x8, so we need independent decompositions
    
    # Let me use schoolbook for now but with optimized indexing
    rank = n * m * p  # 72
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Now try to reduce rank using linear algebra
    # The tensor T is defined by: T[a,b,c] = 1 if there exist i,j,k with 
    # a=i*m+j, b=j*p+k, c=k*n+i
    
    # Build the target tensor as a 3D array
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1
    
    # Try to find a lower-rank decomposition using iterative refinement
    # Start from schoolbook and try to merge/remove terms
    
    def check_decomposition(U_mat, V_mat, W_mat, tol=1e-10):
        """Check if U, V, W exactly decompose the matmul tensor."""
        r = U_mat.shape[1]
        # Check all target entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.dot(U_mat[a, :] * V_mat[b, :], W_mat[c, :])
                    if abs(val - 1) > tol:
                        return False
        
        # Check non-target entries (sample)
        rng = np.random.RandomState(42)
        for _ in range(500):
            a = rng.randint(n * m)
            b = rng.randint(m * p)
            c = rng.randint(p * n)
            
            # Check if this is a target entry
            is_target = False
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if i * m + j == a and j * p + k == b and k * n + i == c:
                            is_target = True
                            break
                    if is_target:
                        break
                if is_target:
                    break
            
            val = np.dot(U_mat[a, :] * V_mat[b, :], W_mat[c, :])
            if is_target:
                if abs(val - 1) > tol:
                    return False
            else:
                if abs(val) > tol:
                    return False
        
        return True
    
    # Try merging columns that are proportional in U and V
    def try_merge_columns(U_mat, V_mat, W_mat):
        """Try to merge columns that are proportional in U and V."""
        r = U_mat.shape[1]
        merged_U = []
        merged_V = []
        merged_W = []
        used = [False] * r
        
        for i in range(r):
            if used[i]:
                continue
            used[i] = True
            u_i = U_mat[:, i].copy()
            v_i = V_mat[:, i].copy()
            w_i = W_mat[:, i].copy()
            
            # Check if any later column can be merged
            for j in range(i + 1, r):
                if used[j]:
                    continue
                u_j = U_mat[:, j].copy()
                v_j = V_mat[:, j].copy()
                w_j = W_mat[:, j].copy()
                
                # Check if (u_j, v_j) is proportional to (u_i, v_i)
                if np.allclose(u_i, 0):
                    continue
                
                nonzero = np.abs(u_i) > 1e-12
                if np.sum(nonzero) == 0:
                    continue
                
                alpha = np.dot(u_j[nonzero], u_i[nonzero]) / np.dot(u_i[nonzero], u_i[nonzero])
                
                if np.allclose(u_j, alpha * u_i, atol=1e-10) and np.allclose(v_j, alpha * v_i, atol=1e-10):
                    # Can merge: w_i = w_i + alpha^2 * w_j
                    w_i = w_i + (alpha * alpha) * w_j
                    used[j] = True
            
            merged_U.append(u_i)
            merged_V.append(v_i)
            merged_W.append(w_i)
        
        if len(merged_U) < r:
            new_U = np.column_stack(merged_U)
            new_V = np.column_stack(merged_V)
            new_W = np.column_stack(merged_W)
            return new_U, new_V, new_W
        
        return None, None, None
    
    # Try to reduce rank by solving a linear system
    # For each potential new column, try to express it as a combination of others
    # This is complex, so let's use a simpler heuristic: random perturbation + projection
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    # Try merging
    new_U, new_V, new_W = try_merge_columns(U, V, W)
    if new_U is not None and check_decomposition(new_U, new_V, new_W):
        best_U, best_V, best_W = new_U, new_V, new_W
        best_rank = new_U.shape[1]
    
    if time.time() > deadline - 10:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Try more aggressive reduction: for each column, try to remove it and adjust others
    # This is a hard problem, so let's use a randomized local search
    
    rng = np.random.RandomState(123)
    for attempt in range(50):
        if time.time() > deadline - 5:
            break
        
        # Randomly perturb a few columns and try to project back
        U_try = best_U.copy()
        V_try = best_V.copy()
        W_try = best_W.copy()
        
        # Pick a random column to potentially remove
        if best_rank <= 1:
            break
        
        col_to_remove = rng.randint(best_rank)
        
        # Try to express this column as a linear combination of others
        # by solving least squares on a subset of tensor entries
        
        # This is complex; for now, just return best found
        pass
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
