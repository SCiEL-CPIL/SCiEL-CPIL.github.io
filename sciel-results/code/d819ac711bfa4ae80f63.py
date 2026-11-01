"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120
    
    # Try to find rank-23 decomposition
    # The tensor T[i,j,k] = 1 for all i,j,k in {0,1,2}
    # We need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    
    # Use known rank-23 construction
    # Based on the structure of matrix multiplication tensor
    
    # Let's try a systematic search with restricted values
    # First, try the known Strassen-based construction for 3x3
    
    # Actually, let me use a numerical optimization approach
    # with exact verification
    
    def verify(U, V, W):
        """Verify that the decomposition is exact."""
        n, m, p = DIMENSIONS
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def verify_half_integer(U, V, W):
        """Check all entries are integer or half-integer."""
        for matrix in [U, V, W]:
            for row in matrix:
                for val in row:
                    if abs(val * 2 - round(val * 2)) > 1e-9:
                        return False
        return True
    
    # Try rank 23 with a structured approach
    # Use the fact that we can build from 2x2 Strassen blocks
    
    # Known rank-23 construction for 3x3x3:
    # This is based on the work of Strassen and later refinements
    
    # Let me try to use a random initialization + optimization approach
    # with exact rational arithmetic verification
    
    best_result = None
    
    # First, try the schoolbook (rank 27) as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    # Now try to find rank-23
    # Use a structured initialization based on known patterns
    
    # Approach: Use the tensor network structure
    # The 3x3x3 MM tensor can be decomposed using 23 rank-1 terms
    
    # Let me try a specific known construction
    # Based on: "The rank of the 3x3x3 matrix multiplication tensor is 23"
    
    # One known construction uses the following pattern:
    # We can think of 3x3 as overlapping 2x2 blocks
    
    # Let me try random restarts with optimization
    rng = np.random.RandomState(42)
    
    for attempt in range(50):
        if time.time() > deadline - 30:
            break
            
        rank = 23
        # Initialize with random values from a restricted set
        vals = [0, 0.5, -0.5, 1, -1, 1.5, -1.5]
        
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        for c in range(rank):
            # Random sparse initialization
            for _ in range(3):
                idx = rng.randint(0, n*m)
                U[idx, c] = rng.choice(vals)
            for _ in range(3):
                idx = rng.randint(0, m*p)
                V[idx, c] = rng.choice(vals)
            for _ in range(3):
                idx = rng.randint(0, p*n)
                W[idx, c] = rng.choice(vals)
        
        # Check if this random assignment works
        if verify(U.tolist(), V.tolist(), W.tolist()):
            if verify_half_integer(U.tolist(), V.tolist(), W.tolist()):
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                return best_result
    
    # Try a more systematic approach
    # Use the known algebraic structure
    
    # The 3x3x3 MM tensor has rank 23. One construction:
    # Think of indices as pairs in {0,1,2}
    
    # Let me try a gradient-based approach with projection to half-integers
    
    for attempt in range(20):
        if time.time() > deadline - 30:
            break
            
        rank = 23
        U = np.random.randn(n*m, rank) * 0.5
        V = np.random.randn(m*p, rank) * 0.5
        W = np.random.randn(p*n, rank) * 0.5
        
        # Simple iterative refinement
        for iteration in range(100):
            if time.time() > deadline - 30:
                break
            
            # Compute error for each (i,j,k)
            errors = {}
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(rank):
                            s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        errors[(i,j,k)] = s - 1.0
            
            # Update U: for each (i,j,c), adjust to reduce error
            # dL/dU[i*m+j,c] = sum_k V[j*p+k,c] * W[k*n+i,c] * error[i,j,k]
            # But this is complex. Let me use a simpler approach.
            
            # Project to nearest half-integers
            U = np.round(U * 2) / 2
            V = np.round(V * 2) / 2
            W = np.round(W * 2) / 2
            
            # Check convergence
            max_err = max(abs(e) for e in errors.values())
            if max_err < 1e-9:
                break
        
        if verify(U.tolist(), V.tolist(), W.tolist()):
            if verify_half_integer(U.tolist(), V.tolist(), W.tolist()):
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                return best_result
    
    return best_result
# EVOLVE-BLOCK-END
