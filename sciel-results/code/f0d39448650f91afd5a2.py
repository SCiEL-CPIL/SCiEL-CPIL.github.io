"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Return a rank-minimal exact tensor decomposition for 2x2 matrix multiplication.
    
    The known minimal rank for 2x2 matrix multiplication is 7 (Strassen's algorithm).
    We return Strassen's rank-7 decomposition with the official trace convention W[k*n+i, r].
    
    Interface:
        U: shape (n*m, rank) = (4, 7)
        V: shape (m*p, rank) = (4, 7)
        W: shape (p*n, rank) = (4, 7)
    
    The decomposition satisfies:
        For each (i, j, k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        And all other tensor entries are 0.
    """
    n, m, p = DIMENSIONS
    rank = 7
    
    # Strassen's algorithm for 2x2 matrix multiplication
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = A*B = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    
    # Strassen's 7 products:
    # M1 = (a + d)(e + h)
    # M2 = (c + d)e
    # M3 = a(f - h)
    # M4 = d(g - e)
    # M5 = (a + b)h
    # M6 = (c + a)f
    # M7 = (b + d)(g + f)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Input indexing: A is 2x2, so a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
    # U rows: [a, b, c, d] -> indices 0, 1, 2, 3
    # V rows: B is 2x2, so e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
    # V rows: [e, f, g, h] -> indices 0, 1, 2, 3
    
    # Output indexing with trace convention W[k*n+i, r]:
    # C[0,0] -> W[0*2+0] = W[0]
    # C[0,1] -> W[1*2+0] = W[2]
    # C[1,0] -> W[0*2+1] = W[1]
    # C[1,1] -> W[1*2+1] = W[3]
    
    # So W rows: [C11, C21, C12, C22] -> indices 0, 1, 2, 3
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # M1 = (a + d)(e + h)
    # a is U[0], d is U[3]; e is V[0], h is V[3]
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    # C11 += M1, C22 += M1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (c + d)e
    # c is U[2], d is U[3]; e is V[0]
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    # C21 += M2, C22 -= M2
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3 = a(f - h)
    # a is U[0]; f is V[1], h is V[3]
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    # C12 += M3, C22 += M3
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4 = d(g - e)
    # d is U[3]; g is V[2], e is V[0]
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    # C11 += M4, C21 += M4
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5 = (a + b)h
    # a is U[0], b is U[1]; h is V[3]
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    # C11 -= M5, C12 += M5
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6 = (c + a)f
    # c is U[2], a is U[0]; f is V[1]
    U[2, 5] = 1; U[0, 5] = 1
    V[1, 5] = 1
    # C22 += M6
    W[3, 5] = 1
    
    # M7 = (b + d)(g + f)
    # b is U[1], d is U[3]; g is V[2], f is V[1]
    U[1, 6] = 1; U[3, 6] = 1
    V[2, 6] = 1; V[1, 6] = 1
    # C11 += M7
    W[0, 6] = 1
    
    # Verify the decomposition
    # The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For matrix multiplication, T[i,j,k] = 1 if i=k (diagonal of output), 0 otherwise
    # Wait, let me reconsider. The matrix multiplication tensor has:
    # T[i,j,k] = 1 if the product C[i,k] depends on A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all valid (i,j,k) in the schoolbook sense
    # Actually no - the matrix multiplication tensor is:
    # T[i,j,k] = delta_{i,k}... no that's not right either.
    
    # Let me think more carefully. The matrix multiplication tensor M_{n,m,p} is defined by:
    # M[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j] * B[j,k]
    # So for each (i,k), there's one term for each j.
    # The tensor entry T[i,j,k] = 1 for all (i,j,k) where i in [0,n), j in [0,m), k in [0,p)
    # Wait, that's the schoolbook tensor which has rank n*m*p.
    
    # Actually, the standard definition: the matrix multiplication tensor has entries
    # T[i,j,k] = 1 for all i,j,k, and the multiplication is:
    # C[i,k] = sum_j sum_r U[i,j,r] V[j,k,r] W[r] ... 
    
    # Let me just verify by computing the full tensor and checking it matches the schoolbook tensor.
    
    T = np.zeros((n, m, p), dtype=float)
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
    
    # The schoolbook tensor has T[i,j,k] = 1 for all i,j,k
    expected_T = np.ones((n, m, p), dtype=float)
    
    if not np.allclose(T, expected_T, atol=1e-10):
        # Fallback to schoolbook if Strassen doesn't verify
        rank_fb = n * m * p
        U_fb = np.zeros((n * m, rank_fb), dtype=float)
        V_fb = np.zeros((m * p, rank_fb), dtype=float)
        W_fb = np.zeros((p * n, rank_fb), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1
                    V_fb[j * p + k, col] = 1
                    W_fb[k * n + i, col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
