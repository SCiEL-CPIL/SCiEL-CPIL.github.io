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

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
