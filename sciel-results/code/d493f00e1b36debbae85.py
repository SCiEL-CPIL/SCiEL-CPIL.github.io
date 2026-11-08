"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product

def construct():
    """
    Strategy: Exploit the rectangular structure by fixing one matrix factor to an identity-like 
    pattern to decouple the 3x7 multiplication, reducing the problem to finding a low-rank 
    decomposition of a 2x7 tensor product with a 3x3 identity.
    
    For matrix multiplication C = A * B where A is n x m and B is m x p:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor is T[i,j,k] = A[i,j] * B[j,k]
    
    In the trace convention:
    - U has shape (n*m, rank) with index i*m + j
    - V has shape (m*p, rank) with index j*p + k  
    - W has shape (p*n, rank) with index k*n + i
    
    The decomposition requires: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    For n=2, m=3, p=7:
    The schoolbook decomposition has rank 42 (one term per (i,j,k) triple).
    
    However, we can potentially do better by exploiting structure.
    
    Key insight: For fixed j, the slice T[:,j,:] is a 2x7 matrix. 
    If we could decompose each slice with rank 2, and share terms across j, we might reduce rank.
    
    But the constraint is that U[i*m+j, r] depends on both i and j, V[j*p+k, r] depends on both j and k,
    and W[k*n+i, r] depends on both k and i.
    
    Let's try a different approach: use the fact that n=2 is very small.
    
    For each j and k, we need to represent the 2-vector [T[0,j,k], T[1,j,k]] = [A[0,j]*B[j,k], A[1,j]*B[j,k]].
    
    This can be written as: B[j,k] * [A[0,j], A[1,j]]
    
    So T[i,j,k] = A[i,j] * B[j,k]
    
    We can write this as a sum over r where each r corresponds to some combination.
    
    One approach: For each j, introduce 2 terms (one for each i value in A).
    Term r_{j,i}: U[i*m+j, r_{j,i}] = 1, V[j*p+k, r_{j,i}] = B[j,k]... but B[j,k] varies with k!
    
    This doesn't work directly because V depends on k.
    
    Alternative: Use the standard decomposition but try to find linear dependencies.
    
    Actually, let me think about this differently. The tensor T[i,j,k] = A[i,j]*B[j,k] is a rank-1 
    tensor in the sense that it's an outer product structure, but the "indices" are coupled through j.
    
    Let's consider: Can we write T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]?
    
    For the schoolbook: U[i,j,r_{ijk}] = 1, V[j,k,r_{ijk}] = 1, W[k,i,r_{ijk}] = 1 for r_{ijk} = i*m*p + j*p + k.
    
    To reduce rank, we need to find if some of these terms can be combined.
    
    For n=2, m=3, p=7, let's try a decomposition with rank = m * max(n,p) = 3*7 = 21?
    Or rank = max(n,p) * m = 7*3 = 21?
    
    Actually, for rectangular matrix multiplication, the known bounds suggest that for 2x3*3x7,
    the rank might be less than 42.
    
    Let me try: rank = n*p + m*(n-1)*(p-1) or similar formulas... 
    
    Actually, let's just use the schoolbook 42-rank decomposition which is guaranteed correct.
    Improving this requires finding a genuine lower-rank decomposition, which is a hard problem.
    
    For now, return the correct 42-rank decomposition.
    """
    n, m, p = DIMENSIONS
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
    
    # Convert to lists as the interface expects
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
