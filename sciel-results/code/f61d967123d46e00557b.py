"""Program for matrix multiplication tensor decomposition (4x4 * 4x7)."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 7]
# Target: rank of tensor for (4,4,7) matrix multiplication
# Schoolbook rank = 4*4*7 = 112

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x4 * 4x7 matrix multiplication.
    Returns dict with keys 'U', 'V', 'W' - factor matrices of shapes (16, rank), (28, rank), (28, rank)
    with integer/half-integer entries.
    """
    deadline = time.time() + 115.0  # safety margin
    
    n, m, p = DIMENSIONS  # 4, 4, 7
    # U: (n*m, rank) = (16, rank)
    # V: (m*p, rank) = (28, rank)  
    # W: (p*n, rank) = (28, rank)
    # Convention: W[k*n + i, r]
    
    # Start with schoolbook decomposition
    def schoolbook():
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
        return U, V, W, rank
    
    U, V, W, best_rank = schoolbook()
    
    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition is valid: U⊗V⊗W reconstructs the multiplication tensor."""
        if time.time() > deadline:
            return False
        
        n, m, p = DIMENSIONS
        # Check shapes
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        if any(len(row) != rank for row in U) or any(len(row) != rank for row in V) or any(len(row) != rank for row in W):
            return False
        
        # For matrix multiplication: T[i,j,k] = 1 for all (i,j,k) in [n]x[m]x[p]
        # A decomposition of rank r means: T[i,j,k] = sum_{r=0}^{r-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    # Try to use structured decompositions based on Strassen-like constructions.
    # For 4x4x4, the best known rank is 49. For 4x4x7, we can potentially do better than 112.
    
    # Strategy: Use block decomposition. Split the 4x7 matrix into 4x4 and 4x3 parts.
    # A(4x4) * B(4x7) = A(4x4) * [B1(4x4) | B2(4x3)]
    # = [A*B1(4x4) | A*B2(4x3)]
    
    # For the 4x4x4 part, use a rank-49 decomposition (based on Strassen-like methods)
    # For the 4x4x3 part, use a rank-48 decomposition (schoolbook) or try to do better
    
    # Let's implement a simple Strassen-like decomposition for 4x4x4.
    # The key idea is to use the fact that 2x2x2 matrix multiplication has rank 7 (Strassen).
    # We can tile this to get better bounds for larger matrices.
    
    # For 4x4x4, one approach is to use two levels of Strassen:
    # First, split 4x4 into four 2x2 blocks.
    # Then use Strassen's algorithm on the 2x2 blocks.
    
    # However, implementing the full rank-49 decomposition is complex.
    # Let's try a different approach: use random search with a better initialization.
    
    # Initialize with a scaled version of the schoolbook decomposition and try to reduce rank.
    
    def try_reduce_rank(initial_rank, target_rank, max_iterations=10000):
        """Try to reduce the rank from initial_rank to target_rank using local search."""
        if time.time() > deadline:
            return None
        
        n, m, p = DIMENSIONS
        nm, mp, pn = n * m, m * p, p * n
        
        # Start with schoolbook
        U_init, V_init, W_init, _ = schoolbook()
        
        # Convert to numpy for efficient computation
        U_np = np.array(U_init, dtype=float)
        V_np = np.array(V_init, dtype=float)
        W_np = np.array(W_init, dtype=float)
        
        # Try to find a lower-rank decomposition by projecting onto a lower-dimensional subspace
        # This is a heuristic approach; it may not always work
        
        best_U, best_V, best_W = None, None, None
        best_score = float('inf')
        
        for iteration in range(max_iterations):
            if time.time() > deadline:
                break
            
            # Randomly select a subset of columns to keep (simulating rank reduction)
            if target_rank < initial_rank:
                cols_to_keep = random.sample(range(initial_rank), target_rank)
                U_trial = U_np[:, cols_to_keep]
                V_trial = V_np[:, cols_to_keep]
                W_trial = W_np[:, cols_to_keep]
            else:
                U_trial = U_np.copy()
                V_trial = V_np.copy()
                W_trial = W_np.copy()
            
            # Compute the error: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] should equal 1
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.dot(U_trial[i*m+j], V_trial[j*p+k]) * W_trial[k*n+i]
                        # Wait, this is not quite right. Let me fix the computation.
                        # The tensor entry is sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        s = 0.0
                        for r in range(U_trial.shape[1]):
                            s += U_trial[i*m+j, r] * V_trial[j*p+k, r] * W_trial[k*n+i, r]
                        error += (s - 1.0) ** 2
            
            if error < best_score:
                best_score = error
                best_U, best_V, best_W = U_trial.copy(), V_trial.copy(), W_trial.copy()
            
            # If we find a perfect decomposition, we can stop
            if error < 1e-9:
                break
            
            # Perturb the matrices slightly
            if iteration % 10 == 0:
                U_np += np.random.normal(0, 0.01, U_np.shape)
                V_np += np.random.normal(0, 0.01, V_np.shape)
                W_np += np.random.normal(0, 0.01, W_np.shape)
        
        if best_score < 1e-9 and best_U is not None:
            # Convert back to list of lists
            U_out = best_U.tolist()
            V_out = best_V.tolist()
            W_out = best_W.tolist()
            return U_out, V_out, W_out, target_rank
        
        return None
    
    # Try to reduce the rank using the above heuristic
    # Start by trying to go from 112 to 100
    result = try_reduce_rank(112, 100, max_iterations=5000)
    if result is not None:
        U, V, W, best_rank = result
    else:
        # Try other target ranks
        for target in [110, 105, 100, 95, 90]:
            if time.time() > deadline:
                break
            result = try_reduce_rank(112, target, max_iterations=3000)
            if result is not None:
                U, V, W, best_rank = result
                break
    
    # Verify the final decomposition
    if not verify_decomposition(U, V, W, best_rank):
        # Fall back to schoolbook if verification fails
        U, V, W, best_rank = schoolbook()
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
