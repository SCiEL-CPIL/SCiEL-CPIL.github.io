import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Shapes
    dim_u = n * m  # 20
    dim_v = m * p  # 25
    dim_w = p * n  # 20
    
    # Target Tensor T
    # T is zero everywhere except T[i*m + j, j*p + k, k*n + i] = 1
    T = np.zeros((dim_u, dim_v, dim_w), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1
            
    # Schoolbook fallback
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((dim_u, rank), dtype=np.int64)
        V = np.zeros((dim_v, rank), dtype=np.int64)
        W = np.zeros((dim_w, rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    start_time = time.time()
    time_limit = 110.0 # Leave 10s margin

    # Known rank for 4x5x5 is 60 (via Winograd-like or block decomposition)
    # Lower bound is max(n*m, m*p, p*n) ? No.
    # Lower bound is max(n, m, p) * min(n, m, p) ? No.
    # Theoretical lower bound is often higher.
    # Let's try to find a decomposition with rank < 100.
    
    # Strategy:
    # 1. Try to construct a decomposition using a structured approach if possible?
    #    The prompt suggests Winograd for 1x5x5.
    #    Rank of 1x5x5 convolution is 8 (Winograd F(8,5)).
    #    So 4 rows * 8 = 32? No, the B matrix is shared.
    #    Actually, matrix multiplication is not just independent convolutions because B is the same for all rows.
    #    However, if we view it as A (4x5) * B (5x5).
    #    C = A B.
    #    Each row of C is a linear combination of rows of B.
    #    This is a linear map from R^5 to R^5 applied to 4 vectors.
    
    # Let's stick to the search approach but improve the local search.
    # The current search is slow because it recomputes the full tensor error.
    # We can speed up error computation.
    
    # T_hat[a, b, c] = sum_k U[a, k] V[b, k] W[c, k]
    # Error = sum_{a,b,c} (T[a,b,c] - T_hat[a,b,c])^2
    # Since T is sparse (only 100 non-zeros), we can just check those.
    # But wait, T_hat might have non-zeros elsewhere, which would also contribute to error.
    # So we must check all (a,b,c) or at least ensure T_hat is zero where T is zero.
    # Given the small dimensions (20, 25, 20), total entries is 10,000.
    # Computing T_hat takes rank * 10,000 ops.
    # If rank=60, that's 600,000 ops. Very fast.
    # The bottleneck in the previous code was likely the Python loops or inefficient einsum if not optimized.
    # Let's use a faster error computation.
    
    def compute_error_fast(U, V, W):
        # U: (20, R), V: (25, R), W: (20, R)
        # T_hat = U @ V^T @ W^T? No.
        # T_hat[a, b, c] = sum_k U[a, k] V[b, k] W[c, k]
        # We can compute this as:
        # M1 = U @ V.T  (20 x 25) -> M1[a, b] = sum_k U[a, k] V[b, k]
        # T_hat = M1 @ W.T? No.
        # T_hat[a, b, c] = sum_k (U[a, k] V[b, k]) W[c, k]
        # This is a 3-tensor contraction.
        # We can compute it as:
        # For each k, add outer(U[:, k], V[:, k]) * W[:, k] to T_hat.
        # Or:
        # T_hat = np.einsum('ak,bk,ck->abc', U, V, W)
        # Let's verify if einsum is fast enough.
        # 20*25*20 = 10000 elements.
        # Rank ~ 60.
        # 60 * 10000 = 600,000 multiplies.
        # This is trivial for numpy.
        
        T_hat = np.einsum('ak,bk,ck->abc', U, V, W)
        return np.sum((T - T_hat) ** 2)

    # Let's try a smarter initialization.
    # Use the schoolbook solution and try to merge columns.
    # Or start with a random solution and refine.
    
    # Let's try to find a rank 60 solution.
    # We will iterate rank from 60 to 100.
    
    for rank in range(60, 101):
        if time.time() - start_time > time_limit:
            break
            
        # Try a few random restarts
        max_restarts = 20
        for r in range(max_restarts):
            if time.time() - start_time > time_limit:
                break
                
            # Random init with integers -1, 0, 1
            # To make it more likely to find a solution, we can bias towards sparsity
            U = np.random.choice([-1, 0, 1], size=(dim_u, rank))
            V = np.random.choice([-1, 0, 1], size=(dim_v, rank))
            W = np.random.choice([-1, 0, 1], size=(dim_w, rank))
            
            current_error = compute_error_fast(U, V, W)
            if current_error == 0:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
            # Local search
            # We want to minimize error.
            # We will do a hill-climbing step.
            
            for step in range(5000):
                if time.time() - start_time > time_limit:
                    break
                    
                # Pick a random matrix and index
                mat_idx = np.random.choice(3)
                if mat_idx == 0:
                    mat = U
                    dim1, dim2 = dim_u, rank
                elif mat_idx == 1:
                    mat = V
                    dim1, dim2 = dim_v, rank
                else:
                    mat = W
                    dim1, dim2 = dim_w, rank
                    
                idx1 = np.random.randint(0, dim1)
                idx2 = np.random.randint(0, dim2)
                
                old_val = mat[idx1, idx2]
                
                # Try changing value by +1, -1, or setting to 0
                # We can also try swapping with another element?
                # Let's stick to +/- 1 and 0 for now.
                
                candidates = [old_val + 1, old_val - 1, 0]
                if old_val != 0:
                    candidates.append(0) # already there if 0 is in list? No, 0 is added.
                # Remove duplicates and old_val
                candidates = [c for c in candidates if c != old_val]
                
                best_candidate = None
                best_new_error = current_error
                
                for c in candidates:
                    mat[idx1, idx2] = c
                    new_error = compute_error_fast(U, V, W)
                    if new_error < best_new_error:
                        best_new_error = new_error
                        best_candidate = c
                    mat[idx1, idx2] = old_val
                
                if best_candidate is not None:
                    mat[idx1, idx2] = best_candidate
                    current_error = best_new_error
                    if current_error == 0:
                        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                else:
                    # No improvement found in this step.
                    # To escape local minima, we can accept a slightly worse move occasionally (Simulated Annealing)
                    # But for exact integer decomposition, we probably want strict improvement.
                    # Let's add a random restart if stuck for too long?
                    # For now, just continue.
                    pass

    return get_schoolbook()

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
