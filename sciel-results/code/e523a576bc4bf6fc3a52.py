"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3 by 4 matrix by a 4 by 5 matrix.
    The output factor follows the official trace convention W[k*n+i,r].
    
    Strategy:
    1. Start with the schoolbook decomposition (rank 60).
    2. Attempt to find a lower rank decomposition using a randomized search with local optimization.
    3. If a lower rank decomposition is found and can be rounded to integer/half-integer entries 
       while maintaining exactness, return it. Otherwise, return the schoolbook decomposition.
       
    The tensor T is defined by T[i,j,k] = 1 for all i,j,k.
    We want to write T as a sum of rank-1 tensors:
       T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 145.0  # Safety margin before 150s limit
    
    rank_schoolbook = n * m * p  # 60
    
    # Build schoolbook decomposition
    U = np.zeros((n * m, rank_schoolbook), dtype=float)
    V = np.zeros((m * p, rank_schoolbook), dtype=float)
    W = np.zeros((p * n, rank_schoolbook), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    def check_decomposition(U, V, W, tol=1e-9):
        """Check if U, V, W exactly decompose the tensor."""
        U_flat = U.reshape(n * m, -1)
        V_flat = V.reshape(m * p, -1)
        W_flat = W.reshape(p * n, -1)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U_flat[i * m + j]
                    v_row = V_flat[j * p + k]
                    w_row = W_flat[k * n + i]
                    
                    val = np.dot(u_row * v_row, w_row)
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def round_to_half(x):
        return np.round(x * 2) / 2
    
    # Try to find a lower rank decomposition
    # We will try rank 55 to 59.
    # For each rank, we will do a few random restarts.
    
    for target_rank in range(55, 60):
        if time.time() > deadline - 5:
            break
        
        # Random restarts
        for _ in range(10):
            if time.time() > deadline - 5:
                break
            
            # Initialize U, V, W with random values
            U_rand = np.random.randn(n * m, target_rank)
            V_rand = np.random.randn(m * p, target_rank)
            W_rand = np.random.randn(p * n, target_rank)
            
            # Check if this random initialization works (unlikely)
            if check_decomposition(U_rand, V_rand, W_rand):
                best_rank = target_rank
                best_U, best_V, best_W = U_rand, V_rand, W_rand
                break
        
        if best_rank < rank_schoolbook:
            break
    
    # Convert to lists for the required interface
    # Ensure entries are integers or half-integers if possible.
    # We should round the values to the nearest half-integer and check if it still works.
    
    if best_rank < rank_schoolbook:
        U_rounded = round_to_half(best_U)
        V_rounded = round_to_half(best_V)
        W_rounded = round_to_half(best_W)
        
        if check_decomposition(U_rounded, V_rounded, W_rounded):
            best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
    
    U_list = [[float(best_U[i, j]) for j in range(best_rank)] for i in range(n * m)]
    V_list = [[float(best_V[i, j]) for j in range(best_rank)] for i in range(m * p)]
    W_list = [[float(best_W[i, j]) for j in range(best_rank)] for i in range(p * n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
