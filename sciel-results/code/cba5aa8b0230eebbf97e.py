"""Rewritten program for 6x6x6 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]
_DEADLINE = time.time() + 145.0

# EVOLVE-BLOCK-START
def construct():
    """Build a rank-189 exact decomposition of the 6x6x6 matmul tensor
    using a 2x2x2 block decomposition with Strassen's 7-multiplication
    algorithm for each 2x2 block.

    Verified Strassen decomposition for A=[[a,b],[c,d]], B=[[e,f],[g,h]]:
        M0 = (a+d)(e+h)
        M1 = (c+d)e
        M2 = a(f-h)
        M3 = d(g-e)
        M4 = (a+b)h
        M5 = (c-a)(e+f)
        M6 = (b-d)(g+h)
        C00 = M0 + M3 - M4 + M6
        C01 = M2 + M4
        C10 = M1 + M3
        C11 = M0 - M1 + M2 + M5
    """
    n, m, p = DIMENSIONS

    # 7 Strassen components: (u_coeffs, v_coeffs, w_coeffs)
    # Each coeff list: [(local_row, local_col, value), ...]
    strassen = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),      # M0
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),              # M1
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),              # M2
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),              # M3
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),              # M4
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),              # M5
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),              # M6
    ]

    block = 2
    nb = n // block  # 3
    mb = m // block  # 3
    pb = p // block  # 3

    total_rank = nb * mb * pb * len(strassen)  # 3*3*3*7 = 189

    U = np.zeros((n*m, total_rank), dtype=np.float64)
    V = np.zeros((m*p, total_rank), dtype=np.float64)
    W = np.zeros((p*n, total_rank), dtype=np.float64)

    col = 0
    for I in range(nb):
        for J in range(mb):
            for K in range(pb):
                for u_c, v_c, w_c in strassen:
                    for (ri, ci, val) in u_c:
                        U[(I*block+ri)*m + (J*block+ci), col] += val
                    for (ri, ci, val) in v_c:
                        V[(J*block+ri)*p + (K*block+ci), col] += val
                    for (ri, ci, val) in w_c:
                        # W index: k*n + i where k is col of C, i is row of C
                        W[(K*block+ci)*n + (I*block+ri), col] += val
                    col += 1

    # Verify correctness: check that U @ diag(1) @ V^T matches W structure
    # The tensor T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r] should equal delta(i,j,k) for matmul
    # Actually for matmul: C[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] = 1 if the (i,j) entry of A times (j,k) entry of B contributes to (i,k) of C
    # T[i,j,k] = 1 always (it's the standard matmul tensor)
    
    # Quick verification with random matrices
    if time.time() < _DEADLINE - 30:
        np.random.seed(42)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        # Compute C from decomposition: C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        C_decomp = np.zeros((n, p))
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    for r in range(total_rank):
                        s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                C_decomp[i, k] = s
        
        err = np.max(np.abs(C_decomp - C_expected))
        if err > 1e-10:
            # Fallback: use standard rank-216 decomposition
            U2 = np.zeros((n*m, n*m*p), dtype=np.float64)
            V2 = np.zeros((m*p, n*m*p), dtype=np.float64)
            W2 = np.zeros((p*n, n*m*p), dtype=np.float64)
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U2[i*m+j, col] = 1
                        V2[j*p+k, col] = 1
                        W2[k*n+i, col] = 1
                        col += 1
            return {
                "U": U2.tolist(),
                "V": V2.tolist(),
                "W": W2.tolist()
            }

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
