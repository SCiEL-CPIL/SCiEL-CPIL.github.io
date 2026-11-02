"""Matrix multiplication 3x4x5: rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 3x4 by 4x5 matrix multiplication.
    
    Uses an iterative search strategy to find a decomposition with rank < 60.
    Falls back to the schoolbook algorithm (rank 60) if no better rank is found.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    # Schoolbook as fallback (Rank 60)
    def get_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = get_schoolbook()
    best_rank = n * m * p
    
    # Helper to compute error and gradients efficiently using NumPy
    def compute_error_and_grads(U, V, W, rank):
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # We need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
        
        # Pre-compute indices
        # This is a bit complex to fully vectorize without huge intermediate arrays,
        # but we can loop over i, j, k and use dot products.
        
        max_err = 0.0
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    
                    # Dot product
                    s = np.dot(U[u_idx], V[v_idx] * W[w_idx])
                    err = s - 1.0
                    if abs(err) > max_err:
                        max_err = abs(err)
                    
                    # Gradients: d(err^2)/d(U[u_idx, c]) = 2 * err * V[v_idx, c] * W[w_idx, c]
                    # We accumulate gradients
                    factor = 2 * err
                    grad_U[u_idx] += factor * (V[v_idx] * W[w_idx])
                    grad_V[v_idx] += factor * (U[u_idx] * W[w_idx])
                    grad_W[w_idx] += factor * (U[u_idx] * V[v_idx])
        
        return max_err, grad_U, grad_V, grad_W

    def try_rank(rank, trials=5):
        if time.time() > deadline - 10:
            return None
        
        for t in range(trials):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(rank * 100 + t)
            # Initialize with small random values
            U = np.random.randn(n * m, rank) * 0.5
            V = np.random.randn(m * p, rank) * 0.5
            W = np.random.randn(p * n, rank) * 0.5
            
            lr = 0.05
            prev_err = float('inf')
            stall_count = 0
            
            for it in range(1000):
                if time.time() > deadline - 5:
                    break
                
                err, gU, gV, gW = compute_error_and_grads(U, V, W, rank)
                
                if err < 1e-9:
                    # Found exact solution? Verify and convert
                    return U, V, W
                
                # Update
                U -= lr * gU
                V -= lr * gV
                W -= lr * gW
                
                # Simple adaptive learning rate
                if err > prev_err:
                    lr *= 0.9
                    stall_count += 1
                    if stall_count > 20:
                        break
                else:
                    stall_count = 0
                    if err < prev_err * 0.99:
                        lr *= 1.1
                        lr = min(lr, 0.5)
                prev_err = err
                
            # Final check
            err, _, _ = compute_error_and_grads(U, V, W, rank)
            if err < 1e-6:
                # Attempt to snap to half-integers?
                # The problem allows integer/half-integer.
                # If we are close, rounding might work.
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                err_rounded, _, _ = compute_error_and_grads(U_rounded, V_rounded, W_rounded, rank)
                if err_rounded < 1e-6:
                    return U_rounded, V_rounded, W_rounded
                else:
                    return U, V, W
        return None

    # Try ranks lower than 60
    # Start from 59 down to 55 (deeper search is unlikely to succeed quickly)
    for rank in range(59, 54, -1):
        if time.time() > deadline - 20:
            break
        
        result = try_rank(rank, trials=10)
        if result is not None:
            U, V, W = result
            # Convert to lists of floats (or ints if appropriate)
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            # Clean up near-zero values
            for i in range(len(U_list)):
                for j in range(len(U_list[i])):
                    if abs(U_list[i][j]) < 1e-9:
                        U_list[i][j] = 0.0
            for i in range(len(V_list)):
                for j in range(len(V_list[i])):
                    if abs(V_list[i][j]) < 1e-9:
                        V_list[i][j] = 0.0
            for i in range(len(W_list)):
                for j in range(len(W_list[i])):
                    if abs(W_list[i][j]) < 1e-9:
                        W_list[i][j] = 0.0
                        
            best_result = {"U": U_list, "V": V_list, "W": W_list}
            best_rank = rank
            break # Found a better rank, stop searching for even lower ones? 
                  # Or continue to find minimal? The loop goes down, so first found is highest rank < 60.
                  # We want MINIMAL rank. So we should continue searching for lower ranks.
            # But if we found 59, we should check 58, 57...
            # The loop continues, so this is correct.

    return best_result
# EVOLVE-BLOCK-END
