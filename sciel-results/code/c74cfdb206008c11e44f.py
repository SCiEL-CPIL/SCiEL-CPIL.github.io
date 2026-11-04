"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
_START_TIME = time.time()
_DEADLINE = 145.0  # Safety margin for 150s limit

def _time_exceeded():
    return (time.time() - _START_TIME) > _DEADLINE

def construct():
    """
    Constructs the schoolbook tensor decomposition for multiplying
    a 4x5 matrix by a 5x5 matrix.
    
    Rank = n * m * p = 100.
    
    U: shape (n*m, rank) = (20, 100)
    V: shape (m*p, rank) = (25, 100)
    W: shape (p*n, rank) = (20, 100)
    
    The decomposition satisfies:
    C[i,k] = sum_j A[i,j] B[j,k] = sum_c (sum_j U[i*m+j, c] A[i,j]) * (sum_k V[j*p+k, c] B[j,k]) ... 
    Actually, the standard bilinear form is:
    C[i,k] = sum_{c} W[k*n+i, c] * (sum_j U[i*m+j, c] A[i,j]) * (sum_j V[j*p+k, c] B[j,k])?
    
    No, the standard definition for matrix multiplication tensor decomposition is:
    The tensor T is in U_A \otimes U_B \otimes U_C.
    T = sum_{c=1}^r u_c \otimes v_c \otimes w_c
    where u_c is a linear form on A, v_c on B, w_c on C.
    
    If we use the basis where A is flattened to vector a, B to b, C to c:
    c_{ik} = sum_{j} a_{ij} b_{jk}
    
    The schoolbook decomposition uses r = n*m*p terms.
    For each (i,j,k), we have a term:
    u_{ijk} = e_{ij}^* (dual basis for A)
    v_{ijk} = e_{jk}^* (dual basis for B)
    w_{ijk} = e_{ik} (basis for C)
    
    In the interface provided:
    U[i*m+j, c] corresponds to the coefficient of A[i*m+j] in the linear form u_c.
    V[j*p+k, c] corresponds to the coefficient of B[j*p+k] in the linear form v_c.
    W[k*n+i, c] corresponds to the coefficient of C[k*n+i] in the linear form w_c.
    Note the W index: W[k*n+i, c]. This means the index into W is k*n+i.
    The output C is usually indexed C[i,k]. The task says "output factor follows the official trace convention W[k*n+i,r]".
    This implies the index in the W matrix corresponds to the output element at row i, col k?
    Let's check the schoolbook code in the prompt:
    W[k*n+i][column] = 1.
    So W index is k*n+i.
    This matches the convention that the vector of C is ordered by k (outer loop?) or i?
    Usually, if C is n x p, flattening row-major gives index i*p + k.
    Flattening column-major gives index k*n + i.
    The prompt uses W[k*n+i]. This suggests column-major ordering for C, or simply that the index is defined as k*n+i.
    We will stick to the prompt's convention: W index is k*n+i.
    
    The decomposition is:
    For c corresponding to (i,j,k):
    U[i*m+j, c] = 1
    V[j*p+k, c] = 1
    W[k*n+i, c] = 1
    
    This gives sum_c U[i*m+j', c] V[j'*p+k', c] W[k'*n+i', c] 
    Wait, the reconstruction is:
    (A*B)[i,k] = sum_j A[i,j] B[j,k]
    
    Using the decomposition:
    sum_c (sum_{i',j'} U[i',j',c] A[i',j']) * (sum_{j',k'} V[j',k',c] B[j',k']) * W[k,i,c]
    
    Let's check the schoolbook term for c=(i0, j0, k0):
    Term = A[i0, j0] * B[j0, k0] * W[k0*n+i0, c]
    
    If we sum over all c, for a fixed output (i,k):
    We pick terms where W[k*n+i, c] is non-zero.
    In schoolbook, W[k*n+i, c] is 1 only if c corresponds to (i, j, k) for some j.
    So for fixed i, k, the non-zero W entries are for c=(i, j, k) for j=0..m-1.
    For these c, U[i*m+j, c] = 1 (and 0 for other i',j').
    V[j*p+k, c] = 1 (and 0 for other j',k').
    
    So the sum becomes:
    sum_{j=0}^{m-1} A[i,j] * B[j,k] * 1
    Which is exactly (A*B)[i,k].
    
    This confirms the schoolbook decomposition is correct.
    """
    n, m, p = DIMENSIONS
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
# EVOLVE-BLOCK-END
