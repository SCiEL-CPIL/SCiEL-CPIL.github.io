# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for evaluator overhead
    
    n, m, p = 3, 5, 6
    rank_schoolbook = n * m * p  # 90
    
    # 1. Construct Schoolbook Fallback (Rank 90)
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    def to_list(mat):
        return mat.tolist()
    
    best_decomp = {"U": to_list(U_sb), "V": to_list(V_sb), "W": to_list(W_sb)}
    
    # 2. Attempt to find a lower rank using a fast, vectorized heuristic search
    # We try ranks 89 down to 85. 
    # For each rank, we do a few random restarts with a short gradient descent.
    # This is much faster than scipy.optimize for this non-convex problem.
    
    def compute_error(U, V, W):
        # U: (nm, r), V: (mp, r), W: (pn, r)
        # Error = sum_{i,j,k} (sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] - 1)^2
        # We can vectorize this partially.
        # Reshape U to (n, m, r)
        # Reshape V to (m, p, r)
        # Reshape W to (p, n, r)
        
        U_resh = U.reshape(n, m, -1)
        V_resh = V.reshape(m, p, -1)
        W_resh = W.reshape(p, n, -1)
        
        # We need sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] for all i,j,k
        # This is a tensor contraction.
        # Let's iterate over i, j, k for simplicity in the inner loop if r is small, 
        # or use einsum.
        # np.einsum('ijc,jkc,kic->ijk', U_resh, V_resh, W_resh)
        # Note: W index is k*n+i, so W_resh[k, i, c]
        
        try:
            T = np.einsum('ijc,jkc,kic->ijk', U_resh, V_resh, W_resh)
        except Exception:
            return np.inf
            
        error = np.sum((T - 1.0) ** 2)
        return error

    def gradient_step(U, V, W, lr):
        # Calculate gradients
        U_resh = U.reshape(n, m, -1)
        V_resh = V.reshape(m, p, -1)
        W_resh = W.reshape(p, n, -1)
        
        T = np.einsum('ijc,jkc,kic->ijk', U_resh, V_resh, W_resh)
        diff = T - 1.0
        
        # dU[i,j,c] = sum_{k} diff[i,j,k] * V[j,k,c] * W[k,i,c]
        # dV[j,k,c] = sum_{i} diff[i,j,k] * U[i,j,c] * W[k,i,c]
        # dW[k,i,c] = sum_{j} diff[i,j,k] * U[i,j,c] * V[j,k,c]
        
        # Using einsum for gradients
        dU = np.einsum('ijk,jkc,kic->ijc', diff, V_resh, W_resh)
        dV = np.einsum('ijk,ijc,kic->jkc', diff, U_resh, W_resh)
        dW = np.einsum('ijk,ijc,jkc->kic', diff, U_resh, V_resh)
        
        U -= lr * dU.flatten()
        V -= lr * dV.flatten()
        W -= lr * dW.flatten()
        
        return U, V, W

    found_lower_rank = False
    
    # Try a few lower ranks
    for target_rank in [89, 88, 87, 86, 85]:
        if time.time() > deadline - 20:
            break
            
        # Try a few random restarts
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(attempt * 100 + target_rank)
            # Initialize with small random values
            U = np.random.randn(n*m, target_rank) * 0.5
            V = np.random.randn(m*p, target_rank) * 0.5
            W = np.random.randn(p*n, target_rank) * 0.5
            
            # Short gradient descent
            lr = 0.01
            for step in range(50):
                if time.time() > deadline - 10:
                    break
                U, V, W = gradient_step(U, V, W, lr)
                
                # Check convergence/validity
                err = compute_error(U, V, W)
                if err < 1e-6:
                    # Try rounding to half-integers
                    U_r = np.round(U * 2) / 2.0
                    V_r = np.round(V * 2) / 2.0
                    W_r = np.round(W * 2) / 2.0
                    
                    err_r = compute_error(U_r, V_r, W_r)
                    if err_r < 1e-9:
                        best_decomp = {"U": to_list(U_r), "V": to_list(V_r), "W": to_list(W_r)}
                        found_lower_rank = True
                        break
                        
            if found_lower_rank:
                break
                
        if found_lower_rank:
            break
            
    return best_decomp
# EVOLVE-BLOCK-END
