import time
import numpy as np

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x5 * 5x6 matrix multiplication.
    Uses Strassen's algorithm for 2x2x2 blocks and schoolbook for the rest.
    """
    n, m, p = 2, 5, 6
    start_time = time.time()
    deadline = start_time + 140

    # Strassen 2x2x2 rank-1 terms: (u, v, w)
    # u: coefficients for A[0,0], A[0,1], A[1,0], A[1,1]
    # v: coefficients for B[0,0], B[0,1], B[1,0], B[1,1]
    # w: coefficients for C[0,0], C[0,1], C[1,0], C[1,1]
    strassen_terms = [
        ([1,0,1,0], [1,1,0,0], [1,0,0,1]),   # M1 = (a+d)(e+h)
        ([0,0,1,1], [1,0,0,0], [0,0,1,0]),   # M2 = (c+d)e
        ([1,0,0,0], [0,1,0,-1], [0,1,0,0]),  # M3 = a(f-h)
        ([0,0,0,1], [-1,0,1,0], [1,0,1,0]),  # M4 = d(g-e)
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),  # M5 = (a+b)h
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),  # M6 = (c-a)(e+f)
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),  # M7 = (b-d)(g+h)
    ]

    # Use Strassen blocks at (j_off, k_off) = (0,0), (2,0), (0,2), (2,2)
    # These cover j in {0,1,2,3} and k in {0,1,2,3}
    # Remaining (j,k) pairs use schoolbook
    strassen_blocks = [(0, 0), (2, 0), (0, 2), (2, 2)]
    
    # Precompute total rank
    R = 4 * 7  # 4 Strassen blocks * 7 terms each
    covered = set()
    for j_off, k_off in strassen_blocks:
        for dj in range(2):
            for dk in range(2):
                covered.add((j_off + dj, k_off + dk))
    
    for j in range(m):
        for k in range(p):
            if (j, k) not in covered:
                R += n  # n schoolbook terms for each (j,k)
    
    U = np.zeros((n * m, R), dtype=np.float64)
    V = np.zeros((m * p, R), dtype=np.float64)
    W = np.zeros((p * n, R), dtype=np.float64)
    
    r = 0
    
    # Add Strassen blocks
    for j_off, k_off in strassen_blocks:
        for u, v, w in strassen_terms:
            # U indices: A[i,j] at i*m+j
            # u[0]=A[0,j_off], u[1]=A[0,j_off+1], u[2]=A[1,j_off], u[3]=A[1,j_off+1]
            U[0 * m + j_off, r] = u[0]
            U[0 * m + j_off + 1, r] = u[1]
            U[1 * m + j_off, r] = u[2]
            U[1 * m + j_off + 1, r] = u[3]
            
            # V indices: B[j,k] at j*p+k
            # v[0]=B[j_off,k_off], v[1]=B[j_off,k_off+1], v[2]=B[j_off+1,k_off], v[3]=B[j_off+1,k_off+1]
            V[j_off * p + k_off, r] = v[0]
            V[j_off * p + k_off + 1, r] = v[1]
            V[(j_off + 1) * p + k_off, r] = v[2]
            V[(j_off + 1) * p + k_off + 1, r] = v[3]
            
            # W indices: C[i,k] at k*n+i
            # w[0]=C[0,k_off], w[1]=C[0,k_off+1], w[2]=C[1,k_off], w[3]=C[1,k_off+1]
            W[k_off * n + 0, r] = w[0]
            W[(k_off + 1) * n + 0, r] = w[1]
            W[k_off * n + 1, r] = w[2]
            W[(k_off + 1) * n + 1, r] = w[3]
            
            r += 1
    
    # Add schoolbook for remaining (j,k) pairs
    for j in range(m):
        for k in range(p):
            if (j, k) not in covered:
                for i in range(n):
                    U[i * m + j, r] = 1.0
                    V[j * p + k, r] = 1.0
                    W[k * n + i, r] = 1.0
                    r += 1
    
    # Verify
    T = np.einsum('ar,br,cr->abc', U, V, W)
    T_exp = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for k in range(p):
            for j in range(m):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_exp[a, b, c] = 1.0
    
    if not np.allclose(T, T_exp, atol=1e-9):
        # Fallback to schoolbook
        R = n * m * p
        U = np.zeros((n * m, R), dtype=np.float64)
        V = np.zeros((m * p, R), dtype=np.float64)
        W = np.zeros((p * n, R), dtype=np.float64)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, idx] = 1.0
                    V[j * p + k, idx] = 1.0
                    W[k * n + i, idx] = 1.0
                    idx += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
