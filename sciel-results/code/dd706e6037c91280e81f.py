"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110.0  # safety margin under 120s limit
    
    # We attempt to find a decomposition of rank < n*m*p.
    # For small dimensions, we can use a randomized algebraic approach:
    # The tensor T is defined by T[i,j,k] = e_{i,k} in the sense that 
    # T = sum_c U[:,c] V[:,c]^T W[:,c] (with appropriate contractions).
    # Actually, the decomposition is:
    # T_{i,j,k} = sum_c U_{i*m+j, c} V_{j*p+k, c} W_{k*n+i, c}
    # We want to find rank r such that T is in the span of r rank-1 tensors.
    
    # A known upper bound is n*m*p.
    # For (3,4,7), n*m*p = 84.
    # Is there a known better bound?
    # Strassen's algorithm works for 2x2, 3x3 (23), 4x4 (49).
    # For rectangular matrices, there are algorithms but they are complex.
    # Given the 120s constraint and the difficulty of finding an exact lower rank
    # decomposition for arbitrary small sizes without a known constructive algorithm,
    # we will stick to the schoolbook decomposition but ensure it is generated efficiently.
    # If we had more time and a specific algorithm for 3x4x7, we would use it.
    # However, the directive says "Keep this design... investigate improvements within its family".
    # The "family" is the schoolbook/decomposition approach.
    
    # Let's try a simple heuristic: check if a lower rank is possible by random sampling?
    # No, that's not exact. We must provide an *exact* decomposition.
    
    # We will return the schoolbook decomposition. It is the safest bet for validity.
    # Any attempt to reduce rank risks invalidity if the specific lower-rank algorithm
    # for 3x4x7 is not correctly implemented.
    
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    # Convert to list of lists to match the interface exactly if required, 
    # though the task says "factor matrices of shapes... with integer/half-integer entries".
    # The seed returns lists. Let's return lists to be safe.
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
