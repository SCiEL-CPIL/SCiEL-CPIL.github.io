"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s kill

    # Schoolbook decomposition of rank n*m*p
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
        return {"U": U, "V": V, "W": W}

    best = schoolbook()
    best_rank = n * m * p

    # Nonzero entries of the tensor
    nz = [(i*m+j, j*p+k, k*n+i) for i in range(n) for j in range(m) for k in range(p)]

    # Verify a decomposition
    def verify(U, V, W):
        rank = len(U[0])
        # Check all required entries
        for (a, b, c) in nz:
            total = sum(U[a][t] * V[b][t] * W[c][t] for t in range(rank))
            if total != 1:
                return False
        # Check that all other entries are zero
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    if (a, b, c) not in nz:
                        total = sum(U[a][t] * V[b][t] * W[c][t] for t in range(rank))
                        if total != 0:
                            return False
        return True

    # Try to find a lower-rank decomposition
    # Strategy: Use a structured approach based on "separation" of indices
    
    # For now, let's try a few specific constructions
    
    # Construction 1: Try to use a "diagonal" W matrix
    # This won't work in general, but let's try
    
    # Construction 2: Use a "block" approach
    # Split the j-index into groups and handle each group separately
    
    # For 3x4x5, let's try to find a decomposition with rank < 60
    
    # Try a greedy approach: start with schoolbook and try to merge terms
    
    # Actually, let's try a different approach: use a constraint solver
    # We'll try to find U, V, W with small integer/half-integer entries
    
    # For time efficiency, let's try a few specific constructions
    
    # Construction: "Strassen-like" for rectangular matrices
    # This is complex, so let's try a simpler approach
    
    # Try rank = 50, 45, 40 and see if we can find valid decompositions
    # This is a heuristic search - try random structured decompositions
    
    # For now, return the schoolbook result since we haven't found a better one
    # The key insight is that we need to find a decomposition that satisfies
    # the tensor constraints exactly
    
    # Let's try a more systematic approach: use a "separation" of indices
    # For each (i, k), we need to compute sum_j A[i,j]*B[j,k]
    # We can try to use a "diagonal" structure in j
    
    # Construction: Use rank = n*p + m*p - p = p*(n+m-1) = 5*(3+4-1) = 30
    # This is a known construction for some cases
    
    def try_construction(rank):
        """Try to find a decomposition with the given rank."""
        if time.time() > deadline - 5:
            return None
        
        # Initialize U, V, W with zeros
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Try to assign values to satisfy the constraints
        # This is a complex constraint satisfaction problem
        # For now, let's try a simple heuristic
        
        # For each (i, j, k), we need sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] = 1
        # And for all other (a, b, c), the sum = 0
        
        # Let's try a "cyclic" assignment
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Assign a unique column for each (i, j, k)
                    col = (i * m * p + j * p + k) % rank
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
        
        # Verify
        if verify(U, V, W):
            return {"U": U, "V": V, "W": W}
        
        return None
    
    # Try different ranks
    for rank in range(30, 60):
        if time.time() > deadline - 10:
            break
        result = try_construction(rank)
        if result is not None:
            if rank < best_rank:
                best = result
                best_rank = rank
            break
    
    return best
# EVOLVE-BLOCK-END
