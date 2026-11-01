"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import nnls

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    # Try to find a rank-minimal exact tensor decomposition
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # The matrix multiplication tensor T has entries T[i,j,k] = 1 if i,j,k correspond to a valid multiplication term
    # For (n x m) * (m x p) = (n x p), the tensor T has shape (n, m, p) where T[i,j,k] = 1 for all valid indices
    # Actually, the standard matrix multiplication tensor is defined as:
    # T[i, j, k] = 1 if the (i,j) element of A and (j,k) element of B contribute to (i,k) element of C
    # This means T is a 3D tensor of ones for all valid (i,j,k) combinations
    
    # The tensor can be decomposed as: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Wait, let me re-read the convention: W[k*n+i, r]
    # So the decomposition is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # For matrix multiplication (n x m) * (m x p), the standard tensor has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # This is because each (i,j,k) triple corresponds to A[i,j] * B[j,k] contributing to C[i,k]
    
    # Let's try to find a low-rank decomposition using numerical methods
    # First, let's try rank = n*m*p (schoolbook) as fallback
    rank_schoolbook = n * m * p
    
    # Try to find minimal rank
    best_rank = rank_schoolbook
    best_U = None
    best_V = None
    best_W = None
    
    # For small dimensions, we can try various ranks
    # The theoretical minimum rank for matrix multiplication is at least max(n*m, m*p, p*n)
    min_possible = max(n*m, m*p, p*n)
    
    # Try ranks from min_possible upward
    for rank in range(min_possible, rank_schoolbook + 1):
        if time.time() > deadline - 5:
            break
            
        try:
            # Try to find a decomposition with this rank
            # We need to solve for U, V, W such that:
            # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            
            # This is a bilinear/trilinear optimization problem
            # Let's use a simple iterative approach or try known constructions
            
            # For 2x2x16, let's see if there's a known low-rank decomposition
            # The standard schoolbook rank is 2*2*16 = 64
            
            # Let's try to use the fact that for matrix multiplication,
            # we can sometimes reduce the rank using algebraic identities
            
            # For now, let's verify if the schoolbook decomposition is valid
            if rank == rank_schoolbook:
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
                
                # Verify the decomposition
                # T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0
                            for r in range(rank):
                                s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                            # The tensor entry should be 1 (since all terms contribute)
                            if abs(s - 1) > 1e-10:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    best_rank = rank
                    best_U = U
                    best_V = V
                    best_W = W
                    break  # Found valid decomposition, return it
                
        except Exception:
            continue
    
    if best_U is not None:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Fallback: schoolbook
    rank = rank_schoolbook
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
