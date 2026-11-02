"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 90
    
    # We will try to find a decomposition of rank < schoolbook_rank
    # using random initialization and iterative refinement.
    # The tensor T[i,j,k] = 1 if we are multiplying (i,j) of A by (j,k) of B.
    # Decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    best_rank = schoolbook_rank
    best_U = None
    best_V = None
    best_W = None
    
    deadline = time.time() + 140  # leave 10s safety margin
    
    # Try ranks from a small number up to schoolbook_rank
    # Start with a reasonable range: try ranks around 40-80
    # But we need integer/half-integer entries, so we'll round at the end.
    
    # First, let's establish a baseline with the schoolbook method
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    best_rank = schoolbook_rank
    
    # Try to find lower-rank decompositions
    # Use random initialization and check if the tensor is exactly represented
    # Since we need exact integer/half-integer entries, we'll use random rational numbers
    
    for target_rank in range(40, schoolbook_rank + 1):
        if time.time() > deadline - 20:
            break
        
        found = False
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            
            # Random initialization with small integers
            U = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(m * p)]
            W = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(p * n)]
            
            # Check if this exactly decomposes the tensor
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        expected = 1  # The tensor entry is always 1 for matrix mult
                        actual = sum(U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c] for c in range(target_rank))
                        if actual != expected:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                best_rank = target_rank
                best_U = U
                best_V = V
                best_W = W
                found = True
                break  # Found minimal for this rank, try lower
        
        if found:
            continue  # Try even lower ranks
        else:
            # If we couldn't find for this rank, it's likely the minimum is higher
            # But we don't know for sure, so keep going up
            pass
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
