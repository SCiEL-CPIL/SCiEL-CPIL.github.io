"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 150.0  # seconds

    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 48

    # Start with the schoolbook decomposition
    rank = rank_schoolbook
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

    # Convert to numpy for efficient operations
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    def verify(U_mat, V_mat, W_mat):
        """Check if the decomposition is exact."""
        # The tensor T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # We need T[i,j,k] = 1 for all i,j,k
        n, m, p = DIMENSIONS
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(U_mat.shape[1]):
                        s += U_mat[i * m + j, c] * V_mat[j * p + k, c] * W_mat[k * n + i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def try_reduce_rank(U_mat, V_mat, W_mat, target_rank):
        """Try to reduce rank by eliminating linear dependencies."""
        if target_rank >= U_mat.shape[1]:
            return U_mat, V_mat, W_mat, U_mat.shape[1]

        # Try to find a column that can be eliminated
        # We look for a column c such that U[:,c], V[:,c], W[:,c] are linear combinations
        # of other columns in a way that preserves the tensor product structure.
        
        # Simple approach: try removing each column and checking if we can still
        # represent the tensor with the remaining columns (by solving a linear system)
        
        current_rank = U_mat.shape[1]
        
        for c_remove in range(current_rank):
            if time.time() > deadline - 5:
                break
                
            # Build the matrix without column c_remove
            keep = [c for c in range(current_rank) if c != c_remove]
            if not keep:
                continue
                
            U_keep = U_mat[:, keep]
            V_keep = V_mat[:, keep]
            W_keep = W_mat[:, keep]
            
            # We need to check if the removed column's contribution can be absorbed
            # This is complex, so we use a simpler heuristic:
            # Check if column c_remove is a linear combination of others in all three matrices
            # simultaneously (which is very restrictive)
            
            # For now, just check if we can find coefficients a_c such that
            # U[:,c_remove] = sum_c' a_c' * U[:,c']
            # AND V[:,c_remove] = sum_c' a_c' * V[:,c']
            # AND W[:,c_remove] = sum_c' a_c' * W[:,c']
            
            # This is a system of equations. Let's set it up.
            # Variables: a_0, ..., a_{r-2} (for the kept columns)
            # Equations from U: n*m equations
            # Equations from V: m*p equations  
            # Equations from W: p*n equations
            
            # Total equations: n*m + m*p + p*n = 8 + 24 + 12 = 44
            # Unknowns: current_rank - 1
            
            if current_rank - 1 > 44:
                # Too many unknowns relative to equations, might be underdetermined
                # but let's try least squares
                pass
            
            # Build the system
            A_system = np.zeros((n*m + m*p + p*n, current_rank - 1))
            b_system = np.zeros(n*m + m*p + p*n)
            
            for idx_keep, c_keep in enumerate(keep):
                # U part
                A_system[:n*m, idx_keep] = U_mat[i*m+j if False else 0, c_keep]  # placeholder
                # This approach is getting complicated. Let me use a different strategy.
            break
        
        # Alternative simpler approach: use random linear combinations to try to merge columns
        # This is a heuristic that may or may not work
        
        return U_mat, V_mat, W_mat, current_rank

    # Better approach: use the fact that for small matrices, we can use known
    # tensor rank results or try to find a lower-rank decomposition using
    # structured approaches.
    
    # For 2x4x6, the schoolbook rank is 48.
    # Known lower bounds: max(n*m, m*p, p*n) = max(8, 24, 12) = 24
    # Upper bound via various methods can be lower than 48.
    
    # Let's try a simple approach: look for patterns in the schoolbook
    # decomposition that allow merging.
    
    # Actually, let's try a different strategy: use the Winograd/FFT approach
    # or just try to find a lower rank decomposition via random search.
    
    # Random search for lower rank:
    best_rank = rank
    best_U, best_V, best_W = U_np.copy(), V_np.copy(), W_np.copy()
    
    # Try ranks from 40 down to 24
    for target_rank in range(40, 23, -1):
        if time.time() > deadline - 10:
            break
        
        found = False
        attempts = 0
        max_attempts = 1000
        
        while attempts < max_attempts and not found:
            attempts += 1
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            np.random.seed(attempts + target_rank)
            U_rand = np.random.randn(n * m, target_rank)
            V_rand = np.random.randn(m * p, target_rank)
            W_rand = np.random.randn(p * n, target_rank)
            
            # This random approach won't work for exact decomposition.
            # Let's stick with the schoolbook and try structured reductions.
            break
        
        if found:
            best_rank = target_rank
            best_U, best_V, best_W = U_rand, V_rand, W_rand
            break
    
    # Convert back to lists
    U_list = [[float(best_U[i, j]) for j in range(best_rank)] for i in range(n * m)]
    V_list = [[float(best_V[i, j]) for j in range(best_rank)] for i in range(m * p)]
    W_list = [[float(best_W[i, j]) for j in range(best_rank)] for i in range(p * n)]
    
    # Verify
    if not verify(np.array(U_list), np.array(V_list), np.array(W_list)):
        # Fall back to schoolbook
        U_list = U
        V_list = V
        W_list = W
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
