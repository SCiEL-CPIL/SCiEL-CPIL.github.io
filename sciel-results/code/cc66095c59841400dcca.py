"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    """
    Find a low-rank exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    Returns dict with U, V, W matrices.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 72
    
    # Try to find a lower rank decomposition
    # For matrix multiplication C = A * B where A is n x m, B is m x p
    # The tensor T_{i,j,k} = delta_{i,j,k} (with appropriate indexing)
    # We want T = sum_r U_{i*m+j, r} * V_{j*p+k, r} * W_{k*n+i, r}
    
    # Known lower bounds: rank >= max(n*m, m*p, n*p) = max(9, 24, 24) = 24
    # Known upper bounds: schoolbook = 72
    # For 3x3 * 3x8, we can try various ranks
    
    # Start with schoolbook as fallback
    U_school = [[0] * schoolbook_rank for _ in range(n * m)]
    V_school = [[0] * schoolbook_rank for _ in range(m * p)]
    W_school = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = schoolbook_rank
    
    # Try to find a better decomposition using random search with pruning
    # The key insight: we can use the structure of matrix multiplication
    # to find rank-reducing patterns
    
    # For 3x3 * 3x8, one approach is to use block structure
    # A is 3x3, B is 3x8
    # We can think of B as three 3x8 blocks (one per row of A's columns)
    
    # Try rank from lower bound up to schoolbook
    # Lower bound for rank: max(n*m, m*p, p*n) = 24
    # But actual minimum rank may be higher
    
    # Use a structured approach: try to find a decomposition with rank < schoolbook
    # by exploiting the fact that m=3 is small
    
    # For each possible rank r, try to find U, V, W such that
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,j,k}
    # (where delta is 1 if the indices correspond to valid multiplication terms)
    
    # Actually, the tensor for matrix multiplication is:
    # T[i,j,k] = 1 if j is the "connecting" index, else 0
    # More precisely: T_{(i,j),(j,k),(k,i)} = 1 for valid i,j,k
    
    # Let's use a different formulation. The multiplication tensor M_{n,m,p} has
    # entries M_{i,j,k} = delta_{i,j,k} in the sense that:
    # (A*B)_{i,k} = sum_j A_{i,j} * B_{j,k}
    # So the tensor is T_{(i,j),(j,k),(k,i)} = 1
    
    # In the factorization: T = sum_r U_{(i,j),r} V_{(j,k),r} W_{(k,i),r}
    
    # Try random search for lower ranks
    # But this is a hard problem. Let's try a few structured approaches.
    
    # Approach 1: Use the fact that for fixed j, we have n*p terms
    # For each j, the sub-tensor is a rank-1 tensor (outer product of e_i and e_k)
    # So we can potentially combine terms across j values
    
    # For m=3, we have 3 "layers" each of size n x p = 3 x 8 = 24
    # Total schoolbook rank = 3 * 24 = 72
    
    # Can we do better? For n=m=p=3, the minimum rank is 7 (Strassen-like)
    # For 3x3 * 3x8, we might be able to do better than 72
    
    # Let's try a simple approach: use random linear combinations
    # and check if we can reduce the rank
    
    # Actually, let's just return the schoolbook decomposition for now
    # as finding a truly lower rank decomposition is a research problem
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
