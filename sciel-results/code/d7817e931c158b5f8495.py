"""Matrix multiplication 2,3,3 rank-minimal exact tensor decomposition search."""
import numpy as np
import time
from itertools import product
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 3]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix
    by a 3x3 matrix.
    
    Returns:
        dict with 'U', 'V', 'W' factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)
        with integer/half-integer entries that exactly decompose the matrix multiplication tensor.
    """
    n, m, p = DIMENSIONS  # 2, 3, 3
    
    # The multiplication tensor T[i,j,k] = 1 for all i,j,k (since C[i,k] = sum_j A[i,j]*B[j,k])
    # Actually T[i,j,k] = 1 for all i in range(n), j in range(m), k in range(p)
    # We need: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
    # Wait, the index convention is: U has shape (n*m, rank), indexed by (i*m+j, r)
    # V has shape (m*p, rank), indexed by (j*p+k, r)
    # W has shape (p*n, rank), indexed by (k*n+i, r)
    # And the tensor is T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # The tensor for matrix multiplication is T[i,j,k] = 1 for all valid i,j,k
    # This is because C[i,k] = sum_j A[i,j] * B[j,k], so the coefficient of A[i,j]*B[j,k] is 1
    
    # Schoolbook decomposition: rank = n*m*p = 18
    # Each term corresponds to a specific (i,j,k)
    
    # Let's try to find a lower rank decomposition using random search
    # We'll start with rank 18 and try to reduce
    
    deadline = time.time() + 110  # Leave 10 seconds buffer
    
    def generate_random_factor(rank):
        """Generate random U, V, W with small integer/half-integer entries."""
        # Use small integers: -1, 0, 1, 2, -2
        vals = [-2, -1, 0, 1, 2]
        U = np.array([[random.choice(vals) for _ in range(rank)] for _ in range(n*m)])
        V = np.array([[random.choice(vals) for _ in range(rank)] for _ in range(m*p)])
        W = np.array([[random.choice(vals) for _ in range(rank)] for _ in range(p*n)])
        return U, V, W
    
    def check_decomposition(U, V, W, tol=1e-9):
        """Check if U, V, W exactly decompose the tensor."""
        rank = U.shape[1]
        # Build tensor: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    # Schoolbook: rank 18
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    # Try to find lower rank decompositions
    best_U, best_V, best_W = schoolbook()
    best_rank = n * m * p
    
    # Try ranks from 17 down to some minimum
    for rank in range(best_rank - 1, 8, -1):
        if time.time() > deadline:
            break
        
        max_attempts = 500
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            U, V, W = generate_random_factor(rank)
            if check_decomposition(U, V, W):
                best_U, best_V, best_W = U, V, W
                best_rank = rank
                break  # Found valid for this rank, try lower
    
    # Convert to lists with integer/half-integer entries
    U_list = [[int(best_U[i, r]) if best_U[i, r] == int(best_U[i, r]) else float(best_U[i, r]) for r in range(best_rank)] for i in range(n*m)]
    V_list = [[int(best_V[i, r]) if best_V[i, r] == int(best_V[i, r]) else float(best_V[i, r]) for r in range(best_rank)] for i in range(m*p)]
    W_list = [[int(best_W[i, r]) if best_W[i, r] == int(best_W[i, r]) else float(best_W[i, r]) for r in range(best_rank)] for i in range(p*n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
