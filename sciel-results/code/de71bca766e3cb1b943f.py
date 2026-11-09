"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 130  # safety margin before 150s limit
    n, m, p = DIMENSIONS
    
    # Start with trivial decomposition (rank = n*m*p)
    base_rank = n * m * p
    best_U = [[0] * base_rank for _ in range(n * m)]
    best_V = [[0] * base_rank for _ in range(m * p)]
    best_W = [[0] * base_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][col] = 1
                best_V[j * p + k][col] = 1
                best_W[k * n + i][col] = 1
                col += 1
    
    best_rank = base_rank
    best_result = (best_U, best_V, best_W)
    
    # The multiplication tensor for (2,3) x (3,9) -> (2,9)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor T[i,j,k] = 1 for all valid i,j,k (it's the identity-like tensor)
    # We need: sum_r U[i,j,r] * V[j,k,r] * W[i,k,r] = 1 for all i,j,k
    
    # Key insight: We can use the fact that for fixed i,k, the equation is
    # sum_j (A[i,j] * B[j,k]) = C[i,k]
    # This is a bilinear form. The tensor rank is the minimum number of rank-1 tensors.
    
    # Let's use a structured approach:
    # For each output (i,k), we need sum_j A[i,j]*B[j,k]
    # We can decompose this using intermediate variables.
    
    # Try to find a decomposition with rank close to the theoretical minimum.
    # Lower bound: max(n*m, m*p, p*n) = max(6, 27, 18) = 27
    # But actually, the rank can be less than n*m*p.
    
    # Strategy 1: Use the fact that we have only 3 middle indices.
    # For each j (0,1,2), we can create terms that handle A[i,j]*B[j,k]
    # This gives rank 3 * 2 * 9 = 54 (trivial)
    
    # Better: Try to share computations across j values.
    # Use Strassen-like ideas or other algebraic tricks.
    
    # Let's try a direct approach: search for rank 18, 19, ..., up to some limit
    # using random initialization with small integers/halves.
    
    def verify_decomposition(U, V, W):
        """Verify that the decomposition is exact."""
        rank = len(U[0]) if isinstance(U, list) else U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        if isinstance(U, list):
                            u_val = U[i*m + j][r]
                            v_val = V[j*p + k][r]
                            w_val = W[k*n + i][r]
                        else:
                            u_val = U[i*m + j, r]
                            v_val = V[j*p + k, r]
                            w_val = W[k*n + i, r]
                        total += u_val * v_val * w_val
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True
    
    # Try structured decompositions
    # Approach: For each pair (i,k), we need sum_j A[i,j]*B[j,k]
    # We can introduce rank-1 terms that "cover" multiple (i,j,k) triples.
    
    # Let's try random search with small integer coefficients
    np.random.seed(42)
    
    # Try ranks from a reasonable minimum up
    # The theoretical lower bound is max(n*m, m*p, p*n) = 27
    # But we might be able to do better.
    
    # First, try a few structured approaches
    # Approach 1: Use the "outer product" structure
    # For each j, create n*p terms (one for each i,k)
    # This gives rank = m * n * p = 54 (trivial)
    
    # Approach 2: Try to reduce by sharing across j
    # Use a basis for the middle dimension
    
    # Let's try random search with various rank values
    for target_rank in range(27, 40):
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            continue
        
        # Try multiple random initializations
        for trial in range(100):
            if time.time() > deadline:
                break
            
            U = np.zeros((n * m, target_rank), dtype=float)
            V = np.zeros((m * p, target_rank), dtype=float)
            W = np.zeros((p * n, target_rank), dtype=float)
            
            # Initialize with random small values (integers or halves)
            for r in range(target_rank):
                U[:, r] = np.random.choice([-1, -0.5, 0, 0.5, 1, 2], size=n*m)
                V[:, r] = np.random.choice([-1, -0.5, 0, 0.5, 1, 2], size=m*p)
                W[:, r] = np.random.choice([-1, -0.5, 0, 0.5, 1, 2], size=p*n)
            
            if verify_decomposition(U, V, W):
                best_rank = target_rank
                best_result = (U.tolist(), V.tolist(), W.tolist())
                break
    
    # Convert best result to list format
    if isinstance(best_result[0], np.ndarray):
        U_final = best_result[0].tolist()
        V_final = best_result[1].tolist()
        W_final = best_result[2].tolist()
    else:
        U_final = best_result[0]
        V_final = best_result[1]
        W_final = best_result[2]
    
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
