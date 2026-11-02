"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
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

def _validate(U, V, W):
    """Check if U, V, W form a valid decomposition.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    should equal 1 for all i,j,k.
    """
    n, m, p = DIMENSIONS
    rank = len(U[0])
    # Convert to numpy for speed
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                val = np.dot(U_np[i*m+j] * V_np[j*p+k], W_np[k*n+i])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _try_random_rank(rank, time_budget, seed=0):
    """Try to find a decomposition of given rank using random search with local optimization."""
    n, m, p = DIMENSIONS
    rng = np.random.RandomState(seed)
    
    # We need to find U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # such that for all i,j,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    best_err = float('inf')
    best_result = None
    start_time = time.time()
    
    for attempt in range(200):
        if time.time() - start_time > time_budget:
            break
        
        # Random initialization
        U = rng.randn(n*m, rank)
        V = rng.randn(m*p, rank)
        W = rng.randn(p*n, rank)
        
        # Simple gradient descent / alternating minimization
        lr = 0.01
        for step in range(500):
            if time.time() - start_time > time_budget:
                break
            
            # Compute errors
            err = 0.0
            # For each (i,j,k), the constraint is sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
            # Gradient with respect to U[i*m+j,c] is V[j*p+k,c]*W[k*n+i,c]
            
            # This is expensive to do naively. Let's use a simpler approach:
            # Just check error and do a few random perturbations.
            
            # Compute total error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j] * V[j*p+k], W[k*n+i])
                        err += (val - 1.0)**2
            
            if err < best_err:
                best_err = err
                if err < 1e-8:
                    # Found a solution!
                    return U, V, W, err
            
            # Random perturbation
            if attempt % 5 == 0:
                U += 0.01 * rng.randn(n*m, rank)
                V += 0.01 * rng.randn(m*p, rank)
                W += 0.01 * rng.randn(p*n, rank)
        
        if time.time() - start_time > time_budget:
            break
    
    return best_result

def _try_structural_rank(rank, time_budget):
    """Try structured decompositions for specific ranks."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Try rank 12 (lower bound is max(n*m, m*p, p*n) = max(6,12,8) = 12)
    if rank == 12:
        # Try to find a rank-12 decomposition
        # This is the theoretical minimum
        pass
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # First, get the schoolbook solution as fallback
    fallback = _schoolbook()
    
    # The theoretical lower bound is max(n*m, m*p, p*n) = max(6, 12, 8) = 12
    # Try to find decompositions with rank < 24 (schoolbook)
    
    # For 2x3x4 matrix multiplication, the known minimal rank is 12
    # (this is the Strassen-like bound for this specific case)
    
    # Let's try to construct a rank-12 decomposition analytically
    # or find one via search
    
    # Known result: the bilinear complexity of 2x3x4 matrix multiplication is 12
    # Let's try to find it
    
    # Try various ranks from 12 upward
    for rank in range(12, 24):
        if time.time() > deadline - 30:
            break
        
        # Try random search
        result = _try_random_rank(rank, 20, seed=rank)
        if result is not None:
            U, V, W, err = result
            if err < 1e-8:
                # Convert to lists
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                if _validate(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
    
    # If we didn't find a better solution, return schoolbook
    return fallback
# EVOLVE-BLOCK-END
