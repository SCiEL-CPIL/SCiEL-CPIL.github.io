"""
Matrix multiplication 3x5x7 - rank-minimal tensor decomposition.

The task is to find factor matrices U, V, W such that for all i in [0,n), j in [0,m), k in [0,p):
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
and for all other (a,b,c) triples, the sum is 0.

This is the standard matrix multiplication tensor of size (n*m) x (m*p) x (p*n).
"""
import numpy as np
import time
import random
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x5 * 5x7 matrix multiplication.
    """
    n, m, p = 3, 5, 7
    nm = n * m  # 15
    mp = m * p  # 35
    pn = p * n  # 21
    
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Build the support: set of (a, b, c) triples where the tensor is 1
    support = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                support.add((a, b, c))
    
    # Schoolbook decomposition: rank = n*m*p = 105
    # Each (i,j,k) triple gets its own rank-1 term
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank), dtype=np.float64)
        V = np.zeros((mp, rank), dtype=np.float64)
        W = np.zeros((pn, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1.0
                    V[b, col] = 1.0
                    W[c, col] = 1.0
                    col += 1
        return U, V, W, rank
    
    # Verify a decomposition
    def verify(U, V, W, tol=1e-9):
        rank = U.shape[1]
        # Compute tensor T[a,b,c] = sum_c U[a,c]*V[b,c]*W[c,c]
        # Use einsum for efficiency
        T = np.einsum('ac,bc,dc->abd', U, V, W)
        
        # Check support elements
        for (a, b, c) in support:
            if abs(T[a, b, c] - 1.0) > tol:
                return False
        
        # Check non-support elements are approximately 0
        # We only need to check that elements not in support are ~0
        # But checking all 15*35*21 = 11025 elements is fine
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    if (a, b, c) not in support:
                        if abs(T[a, b, c]) > tol:
                            return False
        return True

    best_U, best_V, best_W, best_rank = schoolbook()
    
    # Strategy: Use a structured "Winograd-like" approach for 3x3 blocks if possible,
    # or use a more effective random search with rational numbers.
    # Given the time limit, we try a "Greedy" approach:
    # 1. Start with schoolbook.
    # 2. Try to merge columns? No, that's hard.
    # 3. Try to find a lower rank decomposition using random search with a bias towards sparsity and small integers.
    
    rng = random.Random(42)
    
    # We know rank 105 is valid.
    # Let's try to find a valid decomposition with rank < 105.
    # Random search over rational numbers with small denominators is likely to fail for rank < 100.
    # However, we can try a specific known decomposition for 3x5x7 if we can construct it.
    
    # For now, let's improve the random search to be faster and try more attempts.
    # Also, we can try to use the fact that the tensor is symmetric in some sense? No.
    
    # Let's try a "Local Search" starting from schoolbook.
    # We can try to perturb the schoolbook decomposition and see if we can reduce rank.
    # This is difficult.
    
    # Instead, let's try a "Random Search" with a wider range of values and more attempts.
    # But the probability of hitting a valid rank-100 decomposition randomly is extremely low.
    
    # A better strategy: Use the "Winograd" algorithm for 2x2 blocks.
    # The 3x5x7 multiplication can be decomposed into overlapping 2x2 blocks.
    # This is complex to implement correctly from scratch.
    
    # Let's try a different approach: "Integer Programming" via brute force on a subset?
    # No, too large.
    
    # Let's stick to the random search but make it smarter.
    # We can use "Simulated Annealing" to minimize the number of non-zero columns?
    # No, the constraints are equality constraints.
    
    # Let's try to find a decomposition with rank 100-104 using random search with integers.
    # If we fail, we return schoolbook.
    
    # Optimization: Vectorize the verification check for random candidates?
    # The verify function is the bottleneck.
    
    # Let's try to use a "Candidate Generation" approach.
    # Generate random U, V, W and check if they satisfy the equations.
    
    # To make the search more effective, we can fix some entries to be 0 or 1.
    # For example, in the schoolbook decomposition, each column has exactly one 1 in U, V, W.
    # We can try to generate candidates that are "close" to this structure.
    
    # However, for now, let's just run the existing random search but with more attempts
    # and a slightly larger range, and hope for a lucky hit? Unlikely.
    
    # Let's try to use the "Winograd" 3x3 algorithm for the 3x3 sub-blocks.
    # The 3x5x7 can be viewed as:
    # A (3x5) = [A1 (3x3), A2 (3x2)]
    # B (5x7) = [B11 (3x3), B12 (3x4); B21 (2x3), B22 (2x4)]
    # This is not a standard block multiplication.
    
    # Let's try a simple heuristic:
    # Try ranks from 104 down to 90.
    # For each rank, try 1000 random initializations.
    # Use a faster verification.
    
    def fast_verify(U, V, W, tol=1e-9):
        # Compute T
        T = np.einsum('ac,bc,dc->abd', U, V, W)
        
        # Check support
        # We can vectorize this check if we flatten T and support
        # But support is a set of tuples.
        # Let's just loop, it's 105 items.
        for (a, b, c) in support:
            if abs(T[a, b, c] - 1.0) > tol:
                return False
        
        # Check non-support
        # This is the expensive part.
        # We can check if T is zero outside support.
        # Create a mask
        mask = np.zeros((nm, mp, pn), dtype=bool)
        for (a, b, c) in support:
            mask[a, b, c] = True
        
        # Check if any non-masked element is non-zero
        # We can do this by checking if T has any non-zero where mask is False.
        # T[~mask] -> this creates a large array.
        # Better: np.any(np.abs(T[~mask]) > tol)
        
        # To avoid creating the large boolean array every time, we can pre-compute the indices.
        return True # Placeholder, will implement below
    
    # Pre-compute indices for non-support
    # This is a one-time cost.
    non_support_indices = []
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                if (a, b, c) not in support:
                    non_support_indices.append((a, b, c))
    
    def fast_verify2(U, V, W, tol=1e-9):
        T = np.einsum('ac,bc,dc->abd', U, V, W)
        
        # Check support
        for (a, b, c) in support:
            if abs(T[a, b, c] - 1.0) > tol:
                return False
        
        # Check non-support
        for (a, b, c) in non_support_indices:
            if abs(T[a, b, c]) > tol:
                return False
        
        return True

    # Try random search
    # We will try ranks from 104 down to 95.
    # 100 attempts per rank.
    for target_rank in range(104, 95, -1):
        if time.time() > deadline - 20:
            break
        
        for attempt in range(100):
            if time.time() > deadline - 20:
                break
            
            # Random small integer entries
            U = np.array([[rng.randint(-2, 2) for _ in range(target_rank)] for _ in range(nm)], dtype=np.float64)
            V = np.array([[rng.randint(-2, 2) for _ in range(target_rank)] for _ in range(mp)], dtype=np.float64)
            W = np.array([[rng.randint(-2, 2) for _ in range(target_rank)] for _ in range(pn)], dtype=np.float64)
            
            if fast_verify2(U, V, W):
                best_U, best_V, best_W, best_rank = U, V, W, target_rank
                break
        
        if best_rank < 105:
            break
    
    # If we found something better, return it
    if best_rank < 105:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    
    # Otherwise, return schoolbook
    U, V, W, rank = schoolbook()
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
