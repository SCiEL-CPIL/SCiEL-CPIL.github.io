"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import product

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W form a valid decomposition of the matrix multiplication tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build the tensor and check via einsum
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_reshaped = U_arr.reshape(n, m, rank)
    V_reshaped = V_arr.reshape(m, p, rank)
    W_reshaped = W_arr.reshape(p, n, rank)
    
    # For each (i, j, k), compute sum over c
    # Use broadcasting
    # U_reshaped: (n, m, rank)
    # V_reshaped: (m, p, rank)
    # W_reshaped: (p, n, rank)
    
    # We need to sum U[i,j,c] * V[j,k,c] * W[k,i,c] over c
    # Let's do it in a loop for clarity and speed with small dimensions
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(rank):
                    total += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(total - 1.0) > 1e-9:
                    return False
    return True

def _find_2x3x3_decomp(rank, max_iterations=50000, seed=None):
    """Search for a rank-r decomposition of 2x3x3 matrix multiplication.
    Uses random initialization with structured patterns."""
    n, m, p = 2, 3, 3
    rng = np.random.default_rng(seed)
    
    # Number of constraints: n*m*p = 18
    # We need sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1 for all (i,j,k)
    
    for iter in range(max_iterations):
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # Strategy: for each column c, pick a random (i,j,k) and set the corresponding entries to 1
        # But this might not work well. Let's try a more structured approach.
        
        # Alternative: use the fact that we can decompose into rank-1 tensors
        # Each rank-1 tensor is determined by (i,j,k) and has U[i,j]=1, V[j,k]=1, W[k,i]=1
        # We need to find a multiset of (i,j,k) such that for each (i',j',k'), the count is 1
        
        # This is equivalent to finding a decomposition of the all-ones tensor of shape (2,3,3)
        # into rank-1 tensors where each rank-1 tensor corresponds to a unique (i,j,k)
        
        # Actually, the tensor we want to decompose is T[i,j,k] = 1 for all (i,j,k)
        # A rank-1 tensor with U[i,j]=1, V[j,k]=1, W[k,i]=1 contributes 1 to T[i,j,k]
        # So we need to find a set of (i,j,k) pairs such that each (i',j',k') appears exactly once
        
        # But that's just the schoolbook decomposition! Rank = 18.
        
        # To get lower rank, we need to use non-unit values or overlapping patterns.
        
        # Let's try a different approach: use random values and solve
        # Or use a known construction
        
        # For rank 7, there's a known decomposition. Let me try to find it.
        
        # Actually, let's use a systematic search with small values
        # Try values from {-1, 0, 1}
        
        if rank <= 7:
            # Try to find a decomposition with values in {-1, 0, 1}
            # Use backtracking or random search
            
            # For rank 7, we can try to find a solution
            # Let's use a different strategy: start from schoolbook and merge terms
            
            # For now, let's just do random search with small integer values
            for c in range(rank):
                i_val = rng.integers(0, n)
                j_val = rng.integers(0, m)
                k_val = rng.integers(0, p)
                
                # Assign random small values
                u_val = rng.choice([-1, 1])
                v_val = rng.choice([-1, 1])
                w_val = rng.choice([-1, 1])
                
                U[i_val*m + j_val, c] = u_val
                V[j_val*p + k_val, c] = v_val
                W[k_val*n + i_val, c] = w_val
            
            # Verify
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        total = 0.0
                        for c in range(rank):
                            total += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        if abs(total - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                return U.tolist(), V.tolist(), W.tolist()
        else:
            # For higher ranks, use schoolbook-style but with random assignment
            # This should always find a solution if rank >= n*m*p
            if rank >= n*m*p:
                # Use schoolbook
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            U[i*m+j, col] = 1
                            V[j*p+k, col] = 1
                            W[k*n+i, col] = 1
                            col += 1
                return U.tolist(), V.tolist(), W.tolist()
    
    return None

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # For 2x3 x 3x9, we can split into three 2x3 x 3x3 blocks
    # If we can find a rank-7 decomposition for 2x3x3, total rank is 21
    # If we can find a rank-6 decomposition, total rank is 18
    
    # Try to find a low-rank decomposition for 2x3x3
    best_rank = None
    best_decomp = None
    
    # Try ranks from 6 to 18
    for r in range(6, 19):
        if time.time() > deadline - 20:
            break
        
        result = _find_2x3x3_decomp(r, max_iterations=10000, seed=r*1000)
        if result is not None:
            best_rank = r
            best_decomp = result
            break
    
    if best_decomp is not None:
        U_block, V_block, W_block = best_decomp
        block_rank = best_rank
        num_blocks = 3
        total_rank = num_blocks * block_rank
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        for g in range(num_blocks):
            for c in range(block_rank):
                col = g * block_rank + c
                for i in range(n):
                    for j in range(m):
                        U[i*m+j][col] = U_block[i*m+j][c]
                
                for j in range(m):
                    for k_local in range(3):
                        k_global = g * 3 + k_local
                        V[j*p + k_global][col] = V_block[j*3 + k_local][c]
                
                for k_local in range(3):
                    k_global = g * 3 + k_local
                    for i in range(n):
                        W[k_global*n + i][col] = W_block[k_local*n + i][c]
        
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
