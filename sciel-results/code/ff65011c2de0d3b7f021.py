"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np
import random

def _verify(U, V, W, n=3, m=3, p=3, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    T_decomp = np.zeros((n, m, p))
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_decomp[i, j, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    T_target = np.ones((n, m, p))
    return np.allclose(T_decomp, T_target, atol=tol)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145
    
    # Start with schoolbook rank 27
    best_rank = n * m * p
    best_U = [[0] * best_rank for _ in range(n * m)]
    best_V = [[0] * best_rank for _ in range(m * p)]
    best_W = [[0] * best_rank for _ in range(p * n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                best_U[i * m + j][col] = 1
                best_V[j * p + k][col] = 1
                best_W[k * n + i][col] = 1
    
    # Try to find rank-23 decomposition using structured patterns
    # Based on known results, rank-23 is achievable for 3x3x3 matmul
    # Use a more systematic search with constrained entry values
    
    candidate_vals = [-1, 0, 1]  # Start with simple integer values
    
    # First, try a structured approach based on Strassen-like patterns
    # For 3x3, we can try to find a rank-23 decomposition
    
    # Strategy: Use random restarts with local search
    for target_rank in [23, 24, 25, 26]:
        if time.time() > deadline:
            break
        
        max_attempts = 3000
        attempts = 0
        
        while attempts < max_attempts and time.time() < deadline:
            attempts += 1
            
            # Generate sparse random matrices with structured patterns
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            for col in range(target_rank):
                # Each column should have a specific sparsity pattern
                # U: typically 2-3 non-zeros
                nz_u = random.randint(2, 3)
                idx_u = random.sample(range(n * m), nz_u)
                for idx in idx_u:
                    U[idx, col] = random.choice(candidate_vals)
                
                # V: typically 2-3 non-zeros  
                nz_v = random.randint(2, 3)
                idx_v = random.sample(range(m * p), nz_v)
                for idx in idx_v:
                    V[idx, col] = random.choice(candidate_vals)
                
                # W: typically 2-3 non-zeros
                nz_w = random.randint(2, 3)
                idx_w = random.sample(range(p * n), nz_w)
                for idx in idx_w:
                    W[idx, col] = random.choice(candidate_vals)
            
            # Verify
            if _verify(U, V, W, n, m, p):
                best_rank = target_rank
                best_U = U.tolist()
                best_V = V.tolist()
                best_W = W.tolist()
                break
        
        if best_rank <= target_rank:
            break
    
    # If we haven't found rank 23, try with half-integers
    if best_rank > 23 and time.time() < deadline - 30:
        candidate_vals = [-1.5, -1, -0.5, 0, 0.5, 1, 1.5]
        
        for target_rank in [23, 24]:
            if time.time() > deadline:
                break
            
            max_attempts = 2000
            attempts = 0
            
            while attempts < max_attempts and time.time() < deadline:
                attempts += 1
                
                U = np.zeros((n * m, target_rank))
                V = np.zeros((m * p, target_rank))
                W = np.zeros((p * n, target_rank))
                
                for col in range(target_rank):
                    nz_u = random.randint(2, 4)
                    idx_u = random.sample(range(n * m), nz_u)
                    for idx in idx_u:
                        U[idx, col] = random.choice(candidate_vals)
                    
                    nz_v = random.randint(2, 4)
                    idx_v = random.sample(range(m * p), nz_v)
                    for idx in idx_v:
                        V[idx, col] = random.choice(candidate_vals)
                    
                    nz_w = random.randint(2, 4)
                    idx_w = random.sample(range(p * n), nz_w)
                    for idx in idx_w:
                        W[idx, col] = random.choice(candidate_vals)
                
                if _verify(U, V, W, n, m, p):
                    best_rank = target_rank
                    best_U = U.tolist()
                    best_V = V.tolist()
                    best_W = W.tolist()
                    break
            
            if best_rank <= target_rank:
                break
    
    # Convert to lists with integer/half-integer entries
    def to_list(mat):
        result = []
        for row in mat:
            new_row = []
            for val in row:
                val = round(val * 2) / 2
                if val == int(val):
                    new_row.append(int(val))
                else:
                    new_row.append(val)
            result.append(new_row)
        return result
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
