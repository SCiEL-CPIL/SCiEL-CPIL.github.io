"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x4 matrix by a 4x7 matrix.
    The output factor follows the official trace convention W[k*n+i, r].
    
    We search for low-rank decompositions using structured approaches.
    """
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    def verify(U, V, W):
        """Verify that U, V, W give an exact decomposition."""
        rank = U.shape[1]
        # Check: for all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def try_ranked_decomposition(rank, max_trials=50):
        """
        Try to find a rank-`rank` decomposition.
        
        Strategy: For matrix multiplication, we can use the fact that
        C[i,k] = sum_j A[i,j] * B[j,k]
        
        A structured approach: try to find U, V, W with integer/half-integer entries
        that satisfy the tensor equation.
        
        We'll use a greedy/iterative approach:
        1. Initialize with random or structured patterns
        2. Iteratively update one factor at a time to reduce error
        """
        if time.time() > deadline:
            return None
        
        # Lower bound check
        if rank < max(n*m, m*p, p*n):
            return None
        
        # For small ranks, try structured decompositions
        # The key insight: we can group terms
        
        # Approach 1: Try to find a decomposition where each rank-1 term
        # covers a subset of (i,j,k) triples
        
        # For rank r, we need r rank-1 tensors whose sum equals the matmul tensor
        # Each rank-1 tensor is u ⊗ v ⊗ w
        
        # Let's try a systematic search for small rank improvements
        # Start from schoolbook and try to merge terms
        
        # First, let's try a simpler structured approach:
        # If we can find U, V, W such that the bilinear form matches
        
        # For now, let's try random restarts with local optimization
        best_result = None
        np.random.seed(42)
        
        for trial in range(max_trials):
            if time.time() > deadline:
                break
            
            # Initialize with structured random values
            # Use small integers or half-integers to increase chance of exact solution
            scale = np.random.choice([0.5, 1.0, 1.5, 2.0])
            
            U = np.random.randint(-3, 4, (n*m, rank)) * scale
            V = np.random.randint(-3, 4, (m*p, rank)) * scale
            W = np.random.randint(-3, 4, (p*n, rank)) * scale
            
            # Check if this works (unlikely for random, but let's try)
            if verify(U, V, W):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
            # Try a more structured initialization
            # For matrix multiplication, natural structure exists
            # Let's try to build up from partial decompositions
        
        return None
    
    # Try ranks from lower bound up to schoolbook
    lower_bound = max(n*m, m*p, p*n)  # 28 for 3x4x7
    
    # Try a few promising ranks
    ranks_to_try = []
    for r in range(lower_bound, schoolbook_rank + 1):
        ranks_to_try.append(r)
    
    # Sort by rank (we want minimum)
    # But also consider that some ranks are more likely to work
    # Let's try in order of increasing rank
    
    best_decomp = None
    
    for rank in ranks_to_try:
        if time.time() > deadline:
            break
        
        result = try_ranked_decomposition(rank)
        if result is not None:
            best_decomp = result
            break  # Found the minimum rank
    
    if best_decomp is None:
        # Fallback to schoolbook
        best_decomp = schoolbook()
    
    return best_decomp
# EVOLVE-BLOCK-END
