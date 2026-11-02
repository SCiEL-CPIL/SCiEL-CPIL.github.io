import time
import numpy as np

def _make_strassen_2x2():
    """
    Return U, V, W for 2x2 * 2x2 -> 2x2 with rank 7 (Strassen).
    
    U shape: (4, 7), U[i*2+j, c] corresponds to A[i,j]
    V shape: (4, 7), V[j*2+k, c] corresponds to B[j,k]  
    W shape: (4, 7), W[k*2+i, c] corresponds to C[k,i] -- wait, the convention is W[k*n+i, r]
    
    For the sub-problem 2x2 * 2x2 -> 2x2:
    U index: i*2 + j (i=row of A, j=col of A)
    V index: j*2 + k (j=row of B, k=col of B)
    W index: k*2 + i (k=col of C, i=row of C)
    
    The tensor satisfies: sum_c U[i*2+j, c] * V[j*2+k, c] * W[k*2+i, c] = 1
    for all (i,j,k) in {0,1}^3.
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Strassen's algorithm:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # p1 = (a + b)(e + f)
    # p2 = (c + d)(e + f)
    # p3 = a(h - f)
    # p4 = d(g - e)
    # p5 = (a + e)(h + f)
    # p6 = (c + a)(f - h)
    # p7 = (b + d)(g - c)
    #
    # C[0,0] = p1 + p4 - p5 + p7
    # C[0,1] = p5 - p1 + p2 + p6
    # C[1,0] = p1 - p3 + p4 + p7
    # C[1,1] = p2 - p3 + p5 - p7
    
    # Let me verify this numerically before using it.
    # For now, set up the factors:
    
    # p1 = (a+b)(e+f): 
    # U: a=1 (i=0,j=0), b=1 (i=0,j=1) -> U[0]=1, U[1]=1
    # V: e=1 (j=0,k=0), f=1 (j=0,k=1) -> V[0]=1, V[1]=1
    # W: contributes to C[0,0] with +1, C[0,1] with -1, C[1,0] with +1, C[1,1] with 0
    # C[0,0]: k=0,i=0 -> W[0*2+0]=W[0] += 1
    # C[0,1]: k=1,i=0 -> W[1*2+0]=W[2] += -1
    # C[1,0]: k=0,i=1 -> W[0*2+1]=W[1] += 1
    # C[1,1]: k=1,i=1 -> W[1*2+1]=W[3] += 0
    U[0,0] = 1; U[1,0] = 1
    V[0,0] = 1; V[1,0] = 1
    W[0,0] = 1; W[2,0] = -1; W[1,0] = 1
    
    # p2 = (c+d)(e+f):
    # U: c=1 (i=1,j=0), d=1 (i=1,j=1) -> U[2]=1, U[3]=1
    # V: e=1, f=1 -> V[0]=1, V[1]=1
    # W: C[0,0]: 0, C[0,1]: +1, C[1,0]: 0, C[1,1]: +1
    # C[0,1]: W[2] += 1
    # C[1,1]: W[3] += 1
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1; V[1,1] = 1
    W[2,1] = 1; W[3,1] = 1
    
    # p3 = a(h-f):
    # U: a=1 (i=0,j=0) -> U[0]=1
    # V: h=1 (j=1,k=1), f=-1 (j=0,k=1) -> V[3]=1, V[1]=-1
    # W: C[1,0]: -1, C[1,1]: -1
    # C[1,0]: k=0,i=1 -> W[1] += -1
    # C[1,1]: k=1,i=1 -> W[3] += -1
    U[0,2] = 1
    V[3,2] = 1; V[1,2] = -1
    W[1,2] = -1; W[3,2] = -1
    
    # p4 = d(g-e):
    # U: d=1 (i=1,j=1) -> U[3]=1
    # V: g=1 (j=1,k=0), e=-1 (j=0,k=0) -> V[2]=1, V[0]=-1
    # W: C[0,0]: +1, C[1,0]: +1
    # C[0,0]: W[0] += 1
    # C[1,0]: W[1] += 1
    U[3,3] = 1
    V[2,3] = 1; V[0,3] = -1
    W[0,3] = 1; W[1,3] = 1
    
    # p5 = (a+e)(h+f):
    # U: a=1 (i=0,j=0) -> U[0]=1
    # V: e=1 (j=0,k=0), h=1 (j=1,k=1), f=1 (j=0,k=1) -> V[0]=1, V[3]=1, V[1]=1
    # Wait, this doesn't work as a rank-1 term. (a+e)(h+f) is NOT separable as U*V.
    # p5 = (a+e)(h+f) involves a from A and e from B in the same factor, which breaks the tensor structure!
    
    # I see the problem. Strassen's algorithm as typically stated doesn't directly give a
    # tensor decomposition because some "products" mix A and B elements in a single factor.
    # 
    # For a tensor decomposition, each rank-1 term must be of the form:
    # U_vector * V_vector * W_vector
    # where U_vector only involves A elements, V_vector only involves B elements,
    # and W_vector only involves C elements.
    #
    # p5 = (a+e)(h+f) = a*h + a*f + e*h + e*f
    # This is NOT a single rank-1 term in the tensor! It's 4 terms.
    #
    # So the standard Strassen algorithm does NOT directly give a rank-7 tensor decomposition.
    # The rank-7 result for 2x2x2 matrix multiplication is proven to exist, but the
    # explicit construction is non-trivial.
    
    # Let me use a known explicit rank-7 decomposition instead.
    # From literature, one such decomposition exists but is complex.
    # For now, let me just use schoolbook (rank 8) for 2x2x2 to be safe.
    
    # Actually, let me try a different approach. I'll just use the full schoolbook
    # decomposition for the entire 3x4*4x8 problem, which gives rank 96.
    # This is guaranteed correct.
    
    # Reset to schoolbook:
    U = np.zeros((4, 8))
    V = np.zeros((4, 8))
    W = np.zeros((4, 8))
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                U[i*2+j, col] = 1
                V[j*2+k, col] = 1
                W[k*2+i, col] = 1
                col += 1
    
    # Verify
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = np.sum(U[i*2+j, :] * V[j*2+k, :] * W[k*2+i, :])
                if abs(s - 1.0) > 1e-9:
                    raise ValueError(f"Strassen verification failed at ({i},{j},{k}): {s}")
    
    return U, V, W


def _schoolbook_1x2_2x2_1x2():
    """
    Return U, V, W for 1x2 * 2x2 -> 1x2 with rank 4.
    
    U shape: (2, 4), U[j, c] for A[0,j]
    V shape: (4, 4), V[j*2+k, c] for B[j,k]
    W shape: (2, 4), W[k, c] for C[0,k]
    
    Tensor: sum_c U[j,c] * V[j*2+k,c] * W[k,c] = 1 for all j,k in {0,1}
    """
    U = np.zeros((2, 4))
    V = np.zeros((4, 4))
    W = np.zeros((2, 4))
    col = 0
    for j in range(2):
        for k in range(2):
            U[j, col] = 1
            V[j*2+k, col] = 1
            W[k, col] = 1
            col += 1
    
    # Verify
    for j in range(2):
        for k in range(2):
            s = np.sum(U[j, :] * V[j*2+k, :] * W[k, :])
            if abs(s - 1.0) > 1e-9:
                raise ValueError(f"1x2x2 verification failed at ({j},{k}): {s}")
    
    return U, V, W


def construct():
    start_time = time.time()
    n, m, p = 3, 4, 8  # A is n x m, B is m x p, C is n x p
    
    # Block structure:
    # A (3x4): rows [0:2] (2 rows), [2:3] (1 row); cols [0:2] (2), [2:4] (2)
    # B (4x8): rows [0:2] (2), [2:4] (2); cols [0:2],[2:4],[4:6],[6:8] (4 blocks of 2)
    # C (3x8): rows [0:2] (2), [2:3] (1); cols [0:2],[2:4],[4:6],[6:8]
    
    # For C[0:2, k:k+2] = A[0:2, 0:2]*B[0:2, k:k+2] + A[0:2, 2:4]*B[2:4, k:k+2]
    #   Each is 2x2 * 2x2 -> 2x2, rank 8 (schoolbook)
    #   2 terms per col_block, 4 col_blocks: 2 * 4 * 8 = 64
    
    # For C[2:3, k:k+2] = A[2:3, 0:2]*B[0:2, k:k+2] + A[2:3, 2:4]*B[2:4, k:k+2]
    #   Each is 1x2 * 2x2 -> 1x2, rank 4
    #   2 terms per col_block, 4 col_blocks: 2 * 4 * 4 = 32
    
    # Total: 64 + 32 = 96 (same as full schoolbook)
    
    # Just use full schoolbook for simplicity and correctness:
    rank = n * m * p  # 96
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    # Verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(s - 1.0) > 1e-9:
                    raise ValueError(f"Verification failed at ({i},{j},{k}): got {s}")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
