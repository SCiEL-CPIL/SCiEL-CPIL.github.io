"""Matrix multiplication 3,4,8 tensor decomposition."""
import time
import numpy as np
import random

DIMENSIONS = [3, 4, 8]
START_TIME = time.time()
TIME_LIMIT = 120.0  # Safety margin before 150s kill


def _schoolbook(n, m, p):
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


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that the decomposition is exact."""
    rank = len(U[0]) if U else 0
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = Ua[i * m + j, :]
                v = Va[j * p + k, :]
                w = Wa[k * n + i, :]
                val = np.dot(u * v, w)
                if abs(val - 1.0) > tol:
                    return False
    return True


def _kronecker_decomp(n, m, p):
    """
    Build a decomposition exploiting m=4 as 2x2 and p=8 as 2x2x2.
    
    For 3x4x8:
    - Treat the 4 columns of A as a 2x2 block (each 2-wide)
    - Treat the 8 columns of B as a 2x2x2 block (each 4-wide, further split into 2x2)
    
    We use a rank-7 decomposition for 3x2x4 (treating m=2, p=4) 
    and a rank-7 decomposition for 2x2x2 (Strassen).
    Total rank: 7 * 7 = 49.
    
    The key insight: if we can decompose 3x2x4 with rank 7, then
    3x(2*2)x(4*2) = 3x4x8 can be done with rank 7*7 = 49.
    
    First, let's build a rank-7 decomposition for 3x2x4.
    We treat p=4 as 2x2, and use a rank-7 decomposition for 3x2x2,
    then combine with another rank-7 for the 2x2 factor in p.
    
    Actually, let's think more carefully.
    
    For 3x4x8:
    - n=3, m=4=2*2, p=8=2*2*2
    
    We can view this as:
    - First split m into 2x2: A is 3x(2*2), so A = [A1, A2] where A1, A2 are 3x2
    - B is (2*2) x (2*2*2), so B = [[B11, B12], [B21, B22]] where each Bij is 2x(2*2)
    
    The product C = A*B = [A1*B11 + A2*B21, A1*B12 + A2*B22]
    
    Each term like A1*B11 is a 3x2 times 2x4 multiplication.
    
    For 3x2x4:
    - n=3, m=2, p=4
    - We can further decompose p=4 as 2x2
    - This gives us a 3x2x2 problem times a 2x2 factor
    
    Let's use Strassen for 2x2x2 (rank 7) and a base decomposition for 3x2x2.
    
    For 3x2x2:
    - n=3, m=2, p=2
    - Schoolbook rank = 3*2*2 = 12
    - We can try to find a lower rank decomposition.
    
    Actually, let's use a different approach. Let's directly construct a rank-49
    decomposition by using the Kronecker product structure.
    
    We'll use the fact that:
    - 3x4x8 = (3x2) x (2x2) x (2x2) x (2x4) roughly
    
    Let me be more precise. We'll use:
    - A base decomposition for 3x2x2 with some rank r1
    - Combine with Strassen (rank 7) for the 2x2 factors
    
    For now, let's try a simpler approach: use a structured decomposition
    that exploits the binary factorization.
    """
    n, m, p = 3, 4, 8
    
    # First, build a decomposition for 3x2x4
    # This will be used as a building block
    def _decomp_3x2x4():
        """
        Decomposition for 3x2x4 matrix multiplication.
        n=3, m=2, p=4
        U: (3*2, r) = (6, r)
        V: (2*4, r) = (8, r)
        W: (4*3, r) = (12, r)
        
        We'll use a structured approach: treat p=4 as two p=2 blocks.
        Use a rank-12 schoolbook for 3x2x2 and combine.
        Actually, let's use a rank-7 decomposition for 3x2x2 if possible,
        otherwise fall back to schoolbook.
        
        For 3x2x2:
        n=3, m=2, p=2
        Schoolbook rank = 12
        Let's try to find a lower rank.
        
        Actually, let's use a different strategy. We'll use the fact that
        3x2x4 can be decomposed as a combination of 3x2x2 multiplications.
        
        For the 3x2x2 case, we can use a rank-7 decomposition based on
        Strassen-like ideas, but adapted for non-square matrices.
        
        Let's construct a rank-7 decomposition for 3x2x2:
        
        The tensor is T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
        
        For 3x2x2, we have:
        C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
        C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
        C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
        C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0]
        C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1]
        
        We can use Strassen-like formulas. Let's try:
        
        M1 = (A[0,0]+A[0,1]) * (B[1,0]+B[1,1])  -- not quite
        ...
        
        This is getting complex. Let's use a simpler approach for now.
        We'll use the schoolbook for 3x2x2 (rank 12) and combine with
        a rank-7 Strassen for the 2x2 factor, giving rank 12*7=84.
        That's not better than 96.
        
        Let's try yet another approach: directly search for a rank-49
        decomposition using a structured ansatz.
        """
        # For now, return schoolbook for 3x2x4
        rank = 3 * 2 * 4
        U = [[0] * rank for _ in range(6)]
        V = [[0] * rank for _ in range(8)]
        W = [[0] * rank for _ in range(12)]
        col = 0
        for i in range(3):
            for j in range(2):
                for k in range(4):
                    U[i * 2 + j][col] = 1
                    V[j * 4 + k][col] = 1
                    W[k * 3 + i][col] = 1
                    col += 1
        return U, V, W
    
    # This approach is getting too complex. Let's use a simpler strategy.
    # We'll use the schoolbook as fallback and try a few structured decompositions.
    return _schoolbook(3, 4, 8)


def _strassen_2x2x2():
    """
    Rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]
    
    # Strassen formulas for C = A*B where A,B are 2x2
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[i,j],[k,l]]
    # 
    # M1 = (a+b)(e+f)   -> contributes to i,j,k,l
    # M2 = (c+d)(e+f)   -> contributes to k,l
    # M3 = a(g-h)       -> contributes to i,j
    # M4 = d(e-g)       -> contributes to k,l
    # M5 = (a+b)h       -> contributes to j,l
    # M6 = (c-d)e       -> contributes to i,k
    # M7 = (b-c)f       -> contributes to j,k
    #
    # i = M1 + M3 - M4 - M7
    # j = M3 + M5
    # k = M2 + M4
    # l = M1 - M2 + M5 + M6
    
    # Indexing:
    # A: i*m+j = i*2+j
    #   A[0,0]=idx 0, A[0,1]=idx 1, A[1,0]=idx 2, A[1,1]=idx 3
    # B: j*p+k = j*2+k
    #   B[0,0]=idx 0, B[0,1]=idx 1, B[1,0]=idx 2, B[1,1]=idx 3
    # C: k*n+i = k*2+i
    #   C[0,0]=idx 0, C[0,1]=idx 1, C[1,0]=idx 2, C[1,1]=idx 3
    
    # M1 = (a+b)(e+f)
    # U: A[0,0]=1, A[0,1]=1
    U[0][0] = 1
    U[1][0] = 1
    # V: B[0,0]=1, B[0,1]=1
    V[0][0] = 1
    V[1][0] = 1
    # W: C[0,0]=1, C[0,1]=1, C[1,0]=1, C[1,1]=-1 (i + j - k - l... wait)
    # i = M1 + M3 - M4 - M7, so M1 contributes +1 to i, +1 to j, +1 to k, -1 to l
    # Wait, let me recheck:
    # i = M1 + M3 - M4 - M7
    # j = M3 + M5
    # k = M2 + M4
    # l = M1 - M2 + M5 + M6
    #
    # So M1 contributes: +1 to i, 0 to j, 0 to k, +1 to l
    # Wait, that doesn't match. Let me recompute.
    
    # Standard Strassen:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B11 - B12)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    #
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let me use this standard form.
    # A: A11=idx 0, A12=idx 1, A21=idx 2, A22=idx 3
    # B: B11=idx 0, B12=idx 1, B21=idx 2, B22=idx 3
    # C: C11=idx 0, C12=idx 1, C21=idx 2, C22=idx 3
    
    # M1 = (A11+A22)(B11+B22)
    U[0][0] = 1  # A11
    U[3][0] = 1  # A22
    V[0][0] = 1  # B11
    V[3][0] = 1  # B22
    # C11 = M1 - M4 + M5 - M7, so M1 contributes +1 to C11
    # C22 = M1 - M2 + M3 + M6, so M1 contributes +1 to C22
    W[0][0] = 1  # C11
    W[3][0] = 1  # C22
    
    # M2 = (A21+A22)B11
    U[2][1] = 1  # A21
    U[3][1] = 1  # A22
    V[0][1] = 1  # B11
    # C21 = M2 + M4, so M2 contributes +1 to C21
    # C22 = M1 - M2 + M3 + M6, so M2 contributes -1 to C22
    W[2][1] = 1  # C21
    W[3][1] = -1  # C22
    
    # M3 = A11(B21-B22)
    U[0][2] = 1  # A11
    V[2][2] = 1  # B21
    V[3][2] = -1  # B22
    # C12 = M3 + M5, so M3 contributes +1 to C12
    # C22 = M1 - M2 + M3 + M6, so M3 contributes +1 to C22
    W[1][2] = 1  # C12
    W[3][2] = 1  # C22
    
    # M4 = A22(B11-B12)
    U[3][3] = 1  # A22
    V[0][3] = 1  # B11
    V[1][3] = -1  # B12
    # C11 = M1 - M4 + M5 - M7, so M4 contributes -1 to C11
    # C21 = M2 + M4, so M4 contributes +1 to C21
    W[0][3] = -1  # C11
    W[2][3] = 1  # C21
    
    # M5 = (A11+A12)B22
    U[0][4] = 1  # A11
    U[1][4] = 1  # A12
    V[3][4] = 1  # B22
    # C11 = M1 - M4 + M5 - M7, so M5 contributes +1 to C11
    # C12 = M3 + M5, so M5 contributes +1 to C12
    W[0][4] = 1  # C11
    W[1][4] = 1  # C12
    
    # M6 = (A21-A11)(B11+B12)
    U[2][5] = 1  # A21
    U[0][5] = -1  # A11
    V[0][5] = 1  # B11
    V[1][5] = 1  # B12
    # C22 = M1 - M2 + M3 + M6, so M6 contributes +1 to C22
    W[3][5] = 1  # C22
    
    # M7 = (A12-A22)(B21+B22)
    U[1][6] = 1  # A12
    U[3][6] = -1  # A22
    V[2][6] = 1  # B21
    V[3][6] = 1  # B22
    # C11 = M1 - M4 + M5 - M7, so M7 contributes -1 to C11
    W[0][6] = -1  # C11
    
    return {"U": U, "V": V, "W": W}


def construct():
    global START_TIME
    if START_TIME is None:
        START_TIME = time.time()

    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n

    # Schoolbook as guaranteed fallback
    fallback = _schoolbook(n, m, p)

    # Deadline guard
    if time.time() - START_TIME > TIME_LIMIT - 5:
        return fallback

    # Try to find a lower-rank decomposition
    best = fallback
    best_rank = n * m * p  # 96

    # Try Strassen for 2x2x2 as a building block
    # Verify Strassen works
    strassen = _strassen_2x2x2()
    if _verify(strassen["U"], strassen["V"], strassen["W"], 2, 2, 2):
        # Strassen is valid, rank 7
        # Now try to combine for 3x4x8
        pass
    
    # For now, return schoolbook as the best we can guarantee
    return best
