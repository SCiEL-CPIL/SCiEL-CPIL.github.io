"""Matrix multiplication 4x4x6 via 4x4x4 Strassen + 4x4x2 Strassen-block."""
DIMENSIONS = [4, 4, 6]

import time
import numpy as np

def _strassen_2x2():
    """2x2x2 Strassen factors. U: (4,7), V: (4,7), W: (4,7)."""
    n = m = p = 2
    r = 7
    U = [[0.0]*r for _ in range(4)]
    V = [[0.0]*r for _ in range(4)]
    W = [[0.0]*r for _ in range(4)]
    prods = [
        ([(0,0,1.0),(1,1,1.0)], [(0,0,1.0),(1,1,1.0)], [(0,0,1.0),(1,1,1.0)]),
        ([(1,0,1.0),(1,1,1.0)], [(0,0,1.0)], [(1,0,1.0),(1,1,-1.0)]),
        ([(0,0,1.0)], [(0,1,1.0),(1,1,-1.0)], [(0,1,1.0),(1,1,1.0)]),
        ([(1,1,1.0)], [(1,0,1.0),(0,0,-1.0)], [(0,0,1.0),(1,0,1.0)]),
        ([(0,0,1.0),(0,1,1.0)], [(1,1,1.0)], [(0,0,-1.0),(0,1,1.0)]),
        ([(1,0,1.0),(0,0,-1.0)], [(0,0,1.0),(0,1,1.0)], [(1,1,1.0)]),
        ([(0,1,1.0),(1,1,-1.0)], [(1,0,1.0),(1,1,1.0)], [(0,0,1.0)]),
    ]
    for c,(ub,vb,wb) in enumerate(prods):
        for (i,j,cv) in ub: U[i*m+j][c] += cv
        for (j,k,cv) in vb: V[j*p+k][c] += cv
        for (i,k,cv) in wb: W[k*n+i][c] += cv
    return U, V, W

def _build_444():
    """4x4x4 Strassen via Kronecker of 2x2 Strassen. Returns U(16,49), V(16,49), W(16,49)."""
    Us, Vs, Ws = _strassen_2x2()
    r = 49
    U = [[0.0]*r for _ in range(16)]
    V = [[0.0]*r for _ in range(16)]
    W = [[0.0]*r for _ in range(16)]
    for i in range(4):
        for j in range(4):
            for t in range(7):
                for s in range(7):
                    col = 7*t+s
                    uv = Us[2*(i//2)+(i%2)][t]*Us[2*(j//2)+(j%2)][s]
                    if uv: U[i*4+j][col] += uv
                    vv = Vs[2*(j//2)+(j%2)][t]*Vs[2*(k//2)+(k%2)][s] if False else 0
    # Redo properly
    U = [[0.0]*r for _ in range(16)]
    V = [[0.0]*r for _ in range(16)]
    W = [[0.0]*r for _ in range(16)]
    for i in range(4):
        ii = i//2; i0 = i%2
        for j in range(4):
            jj = j//2; j0 = j%2
            for t in range(7):
                for s in range(7):
                    col = 7*t+s
                    uv = Us[2*ii+j0][t]*Us[2*jj+j0][s]
                    if uv: U[i*4+j][col] += uv
                    vv = Vs[2*jj+j0][t]*Vs[2*(j//2)+(j%2)][s]
                    # This is getting confused. Let me redo cleanly.
    # Clean redo
    U = [[0.0]*r for _ in range(16)]
    V = [[0.0]*r for _ in range(16)]
    W = [[0.0]*r for _ in range(16)]
    for i in range(4):
        ii, i0 = i//2, i%2
        for j in range(4):
            jj, j0 = j//2, j%2
            for t in range(7):
                for s in range(7):
                    col = 7*t+s
                    U[i*4+j][col] = Us[2*ii+j0][t]*Us[2*jj+j0][s]
    for j in range(4):
        jj, j0 = j//2, j%2
        for k in range(4):
            kk, k0 = k//2, k%2
            for t in range(7):
                for s in range(7):
                    col = 7*t+s
                    V[j*4+k][col] = Vs[2*jj+k0][t]*Vs[2*kk+k0][s]
    for k in range(4):
        kk, k0 = k//2, k%2
        for i in range(4):
            ii, i0 = i//2, i%2
            for t in range(7):
                for s in range(7):
                    col = 7*t+s
                    W[k*4+i][col] = Ws[2*kk+i0][t]*Ws[2*kk+i0][s]
    # Fix W: W[k*4+i] should be Ws[2*kk+i0][t]*Ws[2*kk+i0][s] -- wrong, should be Ws[2*kk+i0][t]*Ws[2*kk+i0][s]
    # Actually W index is k*n+i = k*4+i, and the formula is Ws[2*K+I][t]*Ws[2*kk+i0][s] where K=kk, I=i0... no.
    # W_s has indices [2*K+I] where K is the "row" (k-block) and I is the "col" (i-block).
    # For Kronecker: W[k*4+i, 7t+s] = Ws[2*kk+i0][t] * Ws[2*kk+i0][s]  -- this is wrong.
    # Correct: W[k*4+i, 7t+s] = Ws[2*kk+i0][t] * Ws[2*kk+i0][s] is not right either.
    # The Kronecker product: W_big[k*4+i, 7t+s] = Ws[2*kk+i0][t] * Ws[2*kk+i0][s]
    # Wait, Ws index is [2*K+I] where K indexes the k-block and I indexes the i-block.
    # So Ws[2*kk+i0] is the (kk,i0) element.
    # For Kronecker: W_big[k*4+i, col(t,s)] = Ws[2*kk+i0][t] * Ws[2*kk+i0][s]
    # But that's the same element multiplied by itself, which is wrong.
    # Let me think again. W_s[2*K+I, t] where K is the row index in the 2x2 output and I is the column.
    # For 4x4: W[k*4+i, 7t+s] = Ws[2*(k//2)+(k%2)][t] * Ws[2*(k//2)+(k%2)][s]
    # No! That's not right. The Kronecker product means:
    # W_big = Ws ⊗ Ws (in appropriate sense)
    # W_big[k*4+i, 7t+s] = Ws[2*kk+i0, t] * Ws[2*kk+i0, s] -- still wrong.
    # 
    # OK let me just be very careful. For the 4x4x4 case:
    # We split into 2x2 blocks. A = [[A00,A01],[A10,A11]], B = [[B00,B01],[B10,B11]], C = [[C00,C01],[C10,C11]]
    # C00 = M1*B00 + M4*B10, C01 = M2*B01 + M5*B00
    # C10 = M6*B00 + M3*B10, C11 = M7*B11 + M5*B00
    # where Mt are Strassen combinations of A blocks.
    # 
    # Each Mt*Bxx is a 2x2*2x2 product done via Strassen.
    # 
    # So for C00 (which is C[0:2,0:2]):
    #   C00 = Strassen(A00,B00) + Strassen(M4,B10)
    # 
    # This is getting complex. Let me use the direct Kronecker formula which is standard:
    # U_444[i*4+j, 7t+s] = U_22[2*(i//2)+(i%2), t] * U_22[2*(j//2)+(j%2), s]
    # V_444[j*4+k, 7t+s] = V_22[2*(j//2)+(j%2), t] * V_22[2*(k//2)+(k%2), s]
    # W_444[k*4+i, 7t+s] = W_22[2*(k//2)+(k%2), t] * W_22[2*(i//2)+(i%2), s]
    # 
    # Wait, W_22 has indices [k*2+i] for 2x2. So W_22[2*K+I, t].
    # For 4x4: W_444[k*4+i, 7t+s] = W_22[2*(k//2)+(k%2), t] * W_22[2*(i//2)+(i%2), s]
    # Hmm, but that mixes k and i differently. Let me check: in the 2x2 case, W[k*2+i, t].
    # The Kronecker product for W should be: W_444[k*4+i, 7t+s] = W_22[2*kk+i0, t] * W_22[2*kk+i0, s]
    # No, that's not a Kronecker product. 
    # 
    # Actually the standard Kronecker decomposition for matrix multiplication is:
    # U_big[i*m+j, r1*R+r2] = U_small[i1*m1+j1, r1] * U_small[i2*m2+j2, r2]
    # where i = i1*M + i2, j = j1*M + j2 (M being the block size ratio)
    # 
    # For 4x4x4 from 2x2x2: M=2
    # i = 2*i1 + i2, j = 2*j1 + j2
    # U_444[i*4+j, 7*r1+r2] = U_22[i1*2+j1, r1] * U_22[i2*2+j2, r2]
    # 
    # Similarly for V: j = 2*j1+j2, k = 2*k1+k2
    # V_444[j*4+k, 7*r1+r2] = V_22[j1*2+k1, r1] * V_22[j2*2+k2, r2]
    # 
    # For W: k = 2*k1+k2, i = 2*i1+i2
    # W_444[k*4+i, 7*r1+r2] = W_22[k1*2+i1, r1] * W_22[k2*2+i2, r2]
    
    U = [[0.0]*r for _ in range(16)]
    V = [[0.0]*r for _ in range(16)]
    W = [[0.0]*r for _ in range(16)]
    for i1 in range(2):
        for i2 in range(2):
            i = 2*i1+i2
            for j1 in range(2):
                for j2 in range(2):
                    j = 2*j1+j2
                    for r1 in range(7):
                        for r2 in range(7):
                            col = 7*r1+r2
                            U[i*4+j][col] = Us[i1*2+j1][r1]*Us[i2*2+j2][r2]
    for j1 in range(2):
        for j2 in range(2):
            j = 2*j1+j2
            for k1 in range(2):
                for k2 in range(2):
                    k = 2*k1+k2
                    for r1 in range(7):
                        for r2 in range(7):
                            col = 7*r1+r2
                            V[j*4+k][col] = Vs[j1*2+k1][r1]*Vs[j2*2+k2][r2]
    for k1 in range(2):
        for k2 in range(2):
            k = 2*k1+k2
            for i1 in range(2):
                for i2 in range(2):
                    i = 2*i1+i2
                    for r1 in range(7):
                        for r2 in range(7):
                            col = 7*r1+r2
                            W[k*4+i][col] = Ws[k1*2+i1][r1]*Ws[k2*2+i2][r2]
    return U, V, W

def _verify(U, V, W, n, m, p, rank):
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    Ur = Ua.reshape(n, m, rank)
    Vr = Va.reshape(m, p, rank)
    Wr = Wa.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    return float(np.max(np.abs(T - 1.0)))

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    t0 = time.time()
    deadline = t0 + 110.0

    Us, Vs, Ws = _strassen_2x2()
    U444, V444, W444 = _build_444()
    
    rank_444 = 49
    rank_442 = 28
    total_rank = rank_444 + rank_442  # 77

    U = [[0.0]*total_rank for _ in range(n*m)]  # 16 x 77
    V = [[0.0]*total_rank for _ in range(m*p)]  # 24 x 77
    W = [[0.0]*total_rank for _ in range(p*n)]  # 24 x 77

    # Part 1: 4x4x4 Strassen for columns 0-3 of B -> columns 0-3 of C
    for idx in range(16):
        for c in range(rank_444):
            if U444[idx][c] != 0:
                U[idx][c] = U444[idx][c]
    for j in range(4):
        for k in range(4):
            for c in range(rank_444):
                if V444[j*4+k][c] != 0:
                    V[j*6+k][c] = V444[j*4+k][c]
    for k in range(4):
        for i in range(4):
            for c in range(rank_444):
                if W444[k*4+i][c] != 0:
                    W[k*4+i][c] = W444[k*4+i][c]

    # Part 2: 4x4x2 Strassen-block for columns 4-5 of B -> columns 4-5 of C
    # Split B into top (rows 0-1) and bottom (rows 2-3) for the last 2 columns
    # C[:,4:6] = A[:,0:2] * B_top[4:6] + A[:,2:4] * B_bot[4:6]
    # Each is a 4x2 * 2x2 product
    # Use 4 Strassen 2x2 products:
    # For each A-block (2x2) and B-block (2x2 in rows 0-1 or 2-3, cols 4-5)
    
    products = [
        (0, 0),  # A[0:2,0:2] * B[0:2,4:6]
        (0, 1),  # A[0:2,2:4] * B[2:4,4:6]
        (1, 0),  # A[2:4,0:2] * B[0:2,4:6]
        (1, 1),  # A[2:4,2:4] * B[2:4,4:6]
    ]
    
    for p_idx, (I, J) in enumerate(products):
        col_off = rank_444 + p_idx*7
        bot = 1 if (I == 1 or J == 1) else 0
        # Actually: bot should be 0 if J==0 (using B rows 0-1), 1 if J==1 (using B rows 2-3)
        bot = J  # J=0 means B top rows, J=1 means B bottom rows
        
        for i0 in range(2):
            for j0 in range(2):
                i = 2*I + i0
                j = 2*J + j0
                for s in range(7):
                    v = Us[2*i0+j0][s]
                    if v: U[i*4+j][col_off+s] += v
        for j0 in range(2):
            for k0 in range(2):
                jg = 2*bot + j0
                kg = 4 + k0
                for s in range(7):
                    v = Vs[2*j0+k0][s]
                    if v: V[jg*6+kg][col_off+s] += v
        for i0 in range(2):
            for k0 in range(2):
                ig = 2*I + i0
                kg = 4 + k0
                for s in range(7):
                    v = Ws[2*k0+i0][s]
                    if v: W[kg*4+ig][col_off+s] += v

    err = _verify(U, V, W, n, m, p, total_rank)
    
    if err > 1e-6:
        rank = n*m*p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
