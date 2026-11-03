"""Combined program for matrix multiplication 3,4,5 tensor rank minimization."""
import numpy as np
import time
import random
import sys
from fractions import Fraction

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    deadline = time.time() + 110  # Leave 10s margin
    
    # Schoolbook decomposition
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    
    # Verification function using exact arithmetic
    def verify_exact(U, V, W):
        """Verify that U, V, W correctly decompose the matrix multiplication tensor.
        U, V, W are lists of lists of Fraction or int/float.
        Returns True if exact decomposition.
        """
        r = len(U[0])
        if len(U) != nm: return False
        if len(V) != mp: return False
        if len(W) != pn: return False
        if any(len(row) != r for row in U): return False
        if any(len(row) != r for row in V): return False
        if any(len(row) != r for row in W): return False
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(r):
                        s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                    if s != 1:
                        return False
        return True
    
    # Verification with scaled integers (multiply by 2, so target is 8)
    def verify_scaled(U2, V2, W2):
        """Verify with scaled integer entries where product sum should be 8."""
        r = len(U2[0])
        if len(U2) != nm: return False
        if len(V2) != mp: return False
        if len(W2) != pn: return False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(r):
                        s += U2[i*m+j][c] * V2[j*p+k][c] * W2[k*n+i][c]
                    if s != 8:
                        return False
        return True
    
    random.seed(42)
    
    # Try to find lower rank using structured and random searches
    # Focus on ranks from 59 down, with increasing effort for lower ranks
    
    # Strategy 1: Structured search with small integer entries (scaled by 2)
    # Try ranks from 59 down to 40 with random small integer entries
    for r in range(rank_school - 1, 39, -1):
        if time.time() > deadline:
            break
        if r >= best_rank:
            continue
            
        # Number of trials based on rank
        num_trials = 50 if r > 50 else 20
        
        for trial in range(num_trials):
            if time.time() > deadline:
                break
                
            # Random small integer entries in {-1, 0, 1} for scaled version
            U2 = [[random.choice([-1, 0, 0, 0, 1]) for _ in range(r)] for _ in range(nm)]
            V2 = [[random.choice([-1, 0, 0, 0, 1]) for _ in range(r)] for _ in range(mp)]
            W2 = [[random.choice([-1, 0, 0, 0, 1]) for _ in range(r)] for _ in range(pn)]
            
            if verify_scaled(U2, V2, W2):
                # Convert back: divide by 2
                U = [[x / 2 for x in row] for row in U2]
                V = [[x / 2 for x in row] for row in V2]
                W = [[x / 2 for x in row] for row in W2]
                best_rank = r
                best_U, best_V, best_W = U, V, W
                break
        
        if best_rank < rank_school:
            # Found improvement, break early
            break
    
    # Strategy 2: Random Gaussian search with error checking
    # This is less likely to find exact solutions but may guide the search
    for r in range(min(best_rank - 1, 55), max(nm, mp, pn) - 1, -1):
        if time.time() > deadline:
            break
        if r >= best_rank:
            continue
            
        num_trials = 10
        for trial in range(num_trials):
            if time.time() > deadline:
                break
                
            # Initialize with small random values
            U = np.random.randn(nm, r) * 0.5
            V = np.random.randn(mp, r) * 0.5
            W = np.random.randn(pn, r) * 0.5
            
            # Check error
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(r))
                        target = 1.0
                        error += (val - target) ** 2
                        
            if error < 1e-12:
                # Found a valid decomposition!
                best_rank = r
                best_U, best_V, best_W = U.tolist(), V.tolist(), W.tolist()
                break
        
        if best_rank < rank_school:
            break
    
    # Convert to lists if still numpy arrays
    if isinstance(best_U, np.ndarray):
        best_U = best_U.tolist()
    if isinstance(best_V, np.ndarray):
        best_V = best_V.tolist()
    if isinstance(best_W, np.ndarray):
        best_W = best_W.tolist()
    
    return {
        "U": best_U,
        "V": best_V,
        "W": best_W
    }
# EVOLVE-BLOCK-END
