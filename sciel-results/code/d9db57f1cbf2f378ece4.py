"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication.
    
    Strategy: Use the schoolbook decomposition as a baseline, then attempt to find
    a lower-rank decomposition using tensor rank algorithms.
    
    The tensor T[i,j,k] for multiplying an (n x m) matrix by an (m x p) matrix
    is defined as T[i,j,k] = delta[i,0] * delta[j,1] * delta[k,2] where the
    multiplication is C[i,k] = sum_j A[i,j] * B[j,k].
    
    For the trace convention W[k*n+i, r], the tensor entries are:
    T[k*n+i, j*p+k, i*m+j] ... actually let me be more careful.
    
    The matrix multiplication tensor T has dimensions (n*p) x (m*p) x (n*m) or similar.
    Let me use the standard formulation:
    
    For C = A * B where A is n x m and B is m x p:
    C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    
    The tensor T has entries T[i, j, k] = 1 if we're computing the (i,k) entry
    using A[i,j] and B[j,k]. So the tensor is in R^{n x m x p} with T[i,j,k] = 1
    for all i,j,k (since each product term A[i,j]*B[j,k] contributes to C[i,k]).
    
    Wait, that's not right. The tensor for matrix multiplication is:
    T_{i,j,k} = delta_{i,k} ... no.
    
    Let me think again. The bilinear map (A, B) -> C = A*B is a tensor in
    (R^{n*m})^* x (R^{m*p})^* x R^{n*p}.
    
    The tensor T has entries T[(i,k), (i,j), (j,k)] = 1 for valid indices,
    meaning T is indexed by (output_index, A_index, B_index).
    
    With the trace convention W[k*n+i, r], the output index is k*n + i.
    
    So T[(k*n+i), (i*m+j), (j*p+k)] = 1 for all valid i, j, k.
    
    The rank of this tensor is the minimum number of rank-1 tensors needed
    to express it.
    
    For n=4, m=5, p=6, the schoolbook rank is n*m*p = 120.
    
    We can try to find a lower rank using:
    1. Known results from tensor decomposition
    2. Numerical optimization approaches
    3. Exploiting structure (e.g., embedding in larger tensors)
    
    Given the time constraint (150s), I'll implement a practical approach:
    1. Start with schoolbook (rank 120)
    2. Try to find improvements using random initialization + gradient descent
    3. Use the deadline guard to return the best result found
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 120
    
    # Build the target tensor T of shape (p*n, n*m, m*p)
    # T[k*n+i, i*m+j, j*p+k] = 1
    T = np.zeros((p * n, n * m, m * p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[k * n + i, i * m + j, j * p + k] = 1.0
    
    # Schoolbook decomposition
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W, rank
    
    U_best, V_best, W_best, rank_best = schoolbook()
    
    # Try to find a lower-rank decomposition
    # We'll use a simple approach: try ranks from a lower bound up to rank_best - 1
    # The lower bound for tensor rank is max(n*m, m*p, p*n) = max(20, 30, 24) = 30
    # But realistically, for 4x5x6, known results suggest rank around 100-110
    
    # Try a few candidate ranks with random initialization
    candidate_ranks = []
    for r in range(max(30, rank_best - 20), rank_best):
        candidate_ranks.append(r)
    
    # Also try some specific values
    candidate_ranks = sorted(set(candidate_ranks + [95, 100, 105, 110, 115]))
    candidate_ranks = [r for r in candidate_ranks if r < rank_best]
    
    def verify_decomposition(U, V, W, T, tol=1e-8):
        """Check if U, V, W decompose T exactly."""
        rank = U.shape[1]
        # Reconstruct T from U, V, W
        # T_reconstructed[a, b, c] = sum_r U[b, r] * V[c, r] * W[a, r]
        # Using einsum: T[a,b,c] = sum_r W[a,r] * U[b,r] * V[c,r]
        T_rec = np.einsum('ar,br,cr->abc', W, U, V)
        return np.allclose(T_rec, T, atol=tol, rtol=tol)
    
    def try_rank(rank, num_trials=5, max_iters=200, lr=0.01):
        """Try to find a decomposition of given rank."""
        best_loss = float('inf')
        best_U = None
        best_V = None
        best_W = None
        
        for trial in range(num_trials):
            # Random initialization
            U = np.random.randn(n * m, rank) * 0.1
            V = np.random.randn(m * p, rank) * 0.1
            W = np.random.randn(p * n, rank) * 0.1
            
            # Simple gradient descent
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                
                # Compute T_rec
                T_rec = np.einsum('ar,br,cr->abc', W, U, V)
                loss = np.sum((T_rec - T) ** 2)
                
                if loss < best_loss:
                    best_loss = loss
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                
                if loss < 1e-12:
                    break
                
                # Compute gradients
                # dL/dW[a,r] = 2 * sum_{b,c} (T_rec[a,b,c] - T[a,b,c]) * U[b,r] * V[c,r]
                # dL/dU[b,r] = 2 * sum_{a,c} (T_rec[a,b,c] - T[a,b,c]) * W[a,r] * V[c,r]
                # dL/dV[c,r] = 2 * sum_{a,b} (T_rec[a,b,c] - T[a,b,c]) * W[a,r] * U[b,r]
                
                diff = T_rec - T
                
                dW = 2 * np.einsum('abc,br,cr->ar', diff, U, V)
                dU = 2 * np.einsum('abc,ar,cr->br', diff, W, V)
                dV = 2 * np.einsum('abc,ar,br->cr', diff, W, U)
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
            
            if time.time() > deadline:
                break
        
        if best_U is not None and best_loss < 1e-8:
            return best_U, best_V, best_W, True
        return None, None, None, False
    
    # Try candidate ranks from low to high
    for rank in sorted(candidate_ranks):
        if time.time() > deadline:
            break
        
        success = False
        for trial_set in range(2):  # 2 sets of trials per rank
            if time.time() > deadline:
                break
            
            U, V, W, found = try_rank(rank, num_trials=3, max_iters=100, lr=0.05)
            if found:
                rank_best = rank
                U_best = U
                V_best = V
                W_best = W
                success = True
                break
        
        if success:
            break
    
    # Final verification
    if verify_decomposition(U_best, V_best, W_best, T, tol=1e-8):
        pass
    else:
        # Fall back to schoolbook
        U_best, V_best, W_best, rank_best = schoolbook()
    
    # Convert to lists for return
    U_list = [[float(U_best[i, j]) for j in range(U_best.shape[1])] for i in range(U_best.shape[0])]
    V_list = [[float(V_best[i, j]) for j in range(V_best.shape[1])] for i in range(V_best.shape[0])]
    W_list = [[float(W_best[i, j]) for j in range(W_best.shape[1])] for i in range(W_best.shape[0])]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
