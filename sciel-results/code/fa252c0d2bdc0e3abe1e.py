"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    Ur = U.reshape(n, m, rank)
    Vr = V.reshape(m, p, rank)
    Wr = W.reshape(p, n, rank)
    # T[i,j,k] = sum_r Ur[i,j,r]*Vr[j,k,r]*Wr[k,i,r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += Ur[i, j, r] * Vr[j, k, r] * Wr[k, i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def construct():
    start = time.time()
    n, m, p = DIMENSIONS
    
    # Collect columns for U, V, W
    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(uvec, vvec, wvec):
        # uvec: dict {(i,j):coeff}
        # vvec: dict {(j,k):coeff}
        # wvec: dict {(k,i):coeff}  # Note: key is (k,i) corresponding to index k*n+i
        u_row = [uvec.get((i, j), 0) for i in range(n) for j in range(m)]
        v_row = [vvec.get((j, k), 0) for j in range(m) for k in range(p)]
        w_row = [wvec.get((k, i), 0) for k in range(p) for i in range(n)]
        U_cols.append(u_row)
        V_cols.append(v_row)
        W_cols.append(w_row)

    # 5 Strassen blocks for columns 0..9
    for b in range(5):
        c0 = 2 * b
        c1 = 2 * b + 1
        
        # M1 = (A00+A11)(B00+B11) -> C00+=M1, C11+=M1
        add_term(
            {(0,0):1, (1,1):1}, 
            {(0,c0):1, (1,c1):1}, 
            {(c0,0):1, (c1,1):1}
        )
        # M2 = (A10+A11)B00 -> C10+=M2, C11-=M2
        add_term(
            {(1,0):1, (1,1):1}, 
            {(0,c0):1}, 
            {(c0,1):1, (c1,1):-1}
        )
        # M3 = A00(B10-B11) -> C01+=M3, C11+=M3
        add_term(
            {(0,0):1}, 
            {(1,c0):1, (1,c1):-1}, 
            {(c1,0):1, (c1,1):1}
        )
        # M4 = A11(B01-B10) -> C00+=M4, C10+=M4
        add_term(
            {(1,1):1}, 
            {(0,c1):1, (1,c0):-1}, 
            {(c0,0):1, (c0,1):1}
        )
        # M5 = (A00+A01)B11 -> C00-=M5, C01+=M5
        add_term(
            {(0,0):1, (0,1):1}, 
            {(1,c1):1}, 
            {(c0,0):-1, (c1,0):1}
        )
        # M6 = (A10-A00)(B00+B10) -> C11+=M6
        add_term(
            {(1,0):1, (0,0):-1}, 
            {(0,c0):1, (1,c0):1}, 
            {(c1,1):1}
        )
        # M7 = (A01-A11)(B01+B11) -> C00+=M7
        add_term(
            {(0,1):1, (1,1):-1}, 
            {(0,c1):1, (1,c1):1}, 
            {(c0,0):1}
        )

    # tail: column 10, optimized 2 terms
    c = 10
    # Term 1: (A00+A10)B0c -> C0c += A00B0c, C1c += A10B0c
    add_term(
        {(0,0):1, (1,0):1}, 
        {(0,c):1}, 
        {(c,0):1, (c,1):1}
    )
    # Term 2: (A01+A11)B1c -> C0c += A01B1c, C1c += A11B1c
    add_term(
        {(0,1):1, (1,1):1}, 
        {(1,c):1}, 
        {(c,0):1, (c,1):1}
    )

    rank = len(U_cols)
    
    # Transpose columns to rows
    U = [[U_cols[r][i] for r in range(rank)] for i in range(n*m)]
    V = [[V_cols[r][i] for r in range(rank)] for i in range(m*p)]
    W = [[W_cols[r][i] for r in range(rank)] for i in range(p*n)]

    result = {"U": U, "V": V, "W": W}

    # Verify
    try:
        if not _verify(U, V, W, n, m, p):
            # Fallback to schoolbook if verification fails
            U_cols_fb = []
            V_cols_fb = []
            W_cols_fb = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        add_term_fb = lambda: None # dummy
                        # Manual append for fallback
                        u_row = [1 if (ii == i and jj == j) else 0 for ii in range(n) for jj in range(m)]
                        v_row = [1 if (jj == j and kk == k) else 0 for jj in range(m) for kk in range(p)]
                        w_row = [1 if (kk == k and ii == i) else 0 for kk in range(p) for ii in range(n)]
                        U_cols_fb.append(u_row)
                        V_cols_fb.append(v_row)
                        W_cols_fb.append(w_row)
            
            rank_fb = len(U_cols_fb)
            U_fb = [[U_cols_fb[r][i] for r in range(rank_fb)] for i in range(n*m)]
            V_fb = [[V_cols_fb[r][i] for r in range(rank_fb)] for i in range(m*p)]
            W_fb = [[W_cols_fb[r][i] for r in range(rank_fb)] for i in range(p*n)]
            result = {"U": U_fb, "V": V_fb, "W": W_fb}
    except Exception:
        pass

    return result
# EVOLVE-BLOCK-END
