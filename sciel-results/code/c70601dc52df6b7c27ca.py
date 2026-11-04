"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 3, 8
    
    # Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # All other entries are 0
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p  # 72
    U_school = np.zeros((n*m, rank_school), dtype=np.float64)
    V_school = np.zeros((m*p, rank_school), dtype=np.float64)
    W_school = np.zeros((p*n, rank_school), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1.0
                V_school[j*p+k, col] = 1.0
                W_school[k*n+i, col] = 1.0
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    def verify(U, V, W, rank):
        """Verify exact decomposition: T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]"""
        # Compute the reconstructed tensor
        # For each r, the outer product U[:,r] ⊗ V[:,r] ⊗ W[:,r]
        # Sum over r
        recon = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for r in range(rank):
            recon += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
        return np.allclose(recon, T, atol=1e-8)
    
    def try_rank(rank, max_iters=200, lr=0.01):
        """Try to find a rank decomposition using alternating least squares."""
        # Random initialization
        rng = np.random.RandomState(42)
        U = rng.randn(n*m, rank) * 0.1
        V = rng.randn(m*p, rank) * 0.1
        W = rng.randn(p*n, rank) * 0.1
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Update U: for each i, U[i,:] = (T[i,:,:] / (V ⊙ W))^T * ... 
            # Actually, we need to solve: sum_r U[i,r] * V[j,r] * W[k,r] = T[i,j,k]
            # For fixed V, W: U[i,r] = sum_{j,k} T[i,j,k] * V[j,r] * W[k,r] / (sum_{j,k} V[j,r]^2 * W[k,r]^2)
            # This is a linear system. Let's use a simpler gradient-based approach.
            
            # Gradient descent on the objective: sum_{i,j,k} (sum_r U[i,r]*V[j,r]*W[k,r] - T[i,j,k])^2
            # Gradient w.r.t. U[i,r]: 2 * sum_{j,k} (sum_r' U[i,r']*V[j,r']*W[k,r'] - T[i,j,k]) * V[j,r] * W[k,r]
            
            # Compute current reconstruction
            recon = np.zeros((n*m, m*p, p*n), dtype=np.float64)
            for r in range(rank):
                recon += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
            
            err = recon - T
            
            # Update U
            dU = np.zeros_like(U)
            for r in range(rank):
                # dU[i,r] = sum_{j,k} err[i,j,k] * V[j,r] * W[k,r]
                dU[:,r] = np.einsum('ijk,jr,kr->i', err, V[:,r], W[:,r])
            U -= lr * dU
            
            # Update V
            dV = np.zeros_like(V)
            for r in range(rank):
                dV[:,r] = np.einsum('ijk,ir,kr->j', err, U[:,r], W[:,r])
            V -= lr * dV
            
            # Update W
            dW = np.zeros_like(W)
            for r in range(rank):
                dW[:,r] = np.einsum('ijk,ir,jr->k', err, U[:,r], V[:,r])
            W -= lr * dW
        
        # Check if we found a valid decomposition
        recon = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for r in range(rank):
            recon += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
        if np.allclose(recon, T, atol=1e-6):
            return U, V, W, True
        
        return None, None, None, False
    
    # Try ranks from low to high
    # For 3x3*3x8, known lower bounds suggest rank >= 40 or so
    # Let's try from 40 up to 71
    for rank in range(40, rank_school):
        if time.time() > deadline - 10:
            break
        
        U, V, W, success = try_rank(rank, max_iters=100, lr=0.005)
        if success:
            best_U, best_V, best_W = U, V, W
            best_rank = rank
            break
    
    # Convert to lists for the interface
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
