"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Number of equations: n*m*p
    num_eq = n * m * p
    
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    def verify(U, V, W, tol=1e-6):
        """Check if the decomposition is valid."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(len(U[0])):
                        val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def verify_numpy(U_arr, V_arr, W_arr, tol=1e-6):
        """Vectorized verification."""
        rank = U_arr.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U_arr[i*m+j, :]
                    v_val = V_arr[j*p+k, :]
                    w_val = W_arr[k*n+i, :]
                    val = np.sum(u_val * v_val * w_val)
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Start with schoolbook decomposition
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_U = U_school.tolist()
    best_V = V_school.tolist()
    best_W = W_school.tolist()
    best_rank = rank_school
    
    # Try to reduce rank by finding linear combinations
    # Strategy: Try ranks from best_rank-1 downward, use random init + optimization
    
    def try_rank(target_rank, max_iters=5000, seed=42):
        """Try to find a decomposition of given rank."""
        if time.time() > deadline - 10:
            return None
        
        rng = np.random.RandomState(seed)
        
        # Initialize with random values
        U_arr = rng.randn(n*m, target_rank) * 0.5
        V_arr = rng.randn(m*p, target_rank) * 0.5
        W_arr = rng.randn(p*n, target_rank) * 0.5
        
        # Use a simple gradient-free approach: random restarts + local perturbation
        best_loss = float('inf')
        best_U = U_arr.copy()
        best_V = V_arr.copy()
        best_W = W_arr.copy()
        
        for restart in range(20):
            if time.time() > deadline - 10:
                break
            
            U_arr = rng.randn(n*m, target_rank) * 0.5
            V_arr = rng.randn(m*p, target_rank) * 0.5
            W_arr = rng.randn(p*n, target_rank) * 0.5
            
            for it in range(max_iters):
                if time.time() > deadline - 10:
                    break
                
                # Compute loss
                loss = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                            loss += (val - 1.0)**2
                
                if loss < best_loss:
                    best_loss = loss
                    best_U = U_arr.copy()
                    best_V = V_arr.copy()
                    best_W = W_arr.copy()
                
                if best_loss < 1e-10:
                    break
                
                # Perturb one random element
                mat = rng.randint(0, 3)
                if mat == 0:
                    row = rng.randint(0, n*m)
                    col = rng.randint(0, target_rank)
                    U_arr[row, col] += rng.randn() * 0.1
                elif mat == 1:
                    row = rng.randint(0, m*p)
                    col = rng.randint(0, target_rank)
                    V_arr[row, col] += rng.randn() * 0.1
                else:
                    row = rng.randint(0, p*n)
                    col = rng.randint(0, target_rank)
                    W_arr[row, col] += rng.randn() * 0.1
        
        if best_loss < 1e-6:
            return best_U, best_V, best_W
        return None
    
    # Try to reduce rank
    for target_rank in range(best_rank - 1, 15, -1):
        if time.time() > deadline - 15:
            break
        
        result = try_rank(target_rank, max_iters=3000)
        if result is not None:
            U_arr, V_arr, W_arr = result
            # Verify
            if verify_numpy(U_arr, V_arr, W_arr, tol=1e-6):
                best_U = U_arr.tolist()
                best_V = V_arr.tolist()
                best_W = W_arr.tolist()
                best_rank = target_rank
                break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
