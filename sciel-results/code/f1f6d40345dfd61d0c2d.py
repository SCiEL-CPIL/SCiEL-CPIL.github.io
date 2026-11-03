"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import itertools
from fractions import Fraction

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x10 matrix.
    
    We use a heuristic search to find a lower rank decomposition than the schoolbook rank of 60.
    The condition is: for all i in [0,n), j in [0,m), k in [0,p):
    sum_{c=0}^{r-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    We search for integer/half-integer solutions with small rank.
    """
    n, m, p = DIMENSIONS
    
    # Try to find a decomposition with rank less than 60
    # We'll use a structured approach based on known techniques for small matrix multiplication
    
    # For 2x3 * 3x10, let's try rank 55 first, then go down
    # We'll use a random search with restarts
    
    def try_rank(r, max_iterations=1000):
        """Try to find a rank-r decomposition using random search."""
        for iteration in range(max_iterations):
            # Initialize with random values
            np.random.seed(iteration * 7919 + 42)
            
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            W = np.zeros((p * n, r))
            
            # Set some entries to 1, some to -1, some to 0
            for c in range(r):
                # Randomly choose which (i,j) pairs get non-zero U values
                for i in range(n):
                    for j in range(m):
                        if np.random.random() < 0.3:
                            U[i * m + j, c] = np.random.choice([-1, 1])
                
                for j in range(m):
                    for k in range(p):
                        if np.random.random() < 0.3:
                            V[j * p + k, c] = np.random.choice([-1, 1])
                
                for k in range(p):
                    for i in range(n):
                        if np.random.random() < 0.3:
                            W[k * n + i, c] = np.random.choice([-1, 1])
            
            # Check if this decomposition works
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        total = 0.0
                        for c in range(r):
                            total += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                        if abs(total - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                # Convert to fractions
                U_frac = [[Fraction(int(U[i, c])) for c in range(r)] for i in range(n * m)]
                V_frac = [[Fraction(int(V[i, c])) for c in range(r)] for i in range(m * p)]
                W_frac = [[Fraction(int(W[i, c])) for c in range(r)] for i in range(p * n)]
                return U_frac, V_frac, W_frac
        
        return None
    
    # Try ranks from 55 down to 50
    for r in range(55, 49, -1):
        result = try_rank(r, max_iterations=200)
        if result is not None:
            U_frac, V_frac, W_frac = result
            return {"U": U_frac, "V": V_frac, "W": W_frac}
    
    # Fallback to schoolbook if no lower rank found
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
# EVOLVE-BLOCK-END
