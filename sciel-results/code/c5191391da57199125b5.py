import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # We will construct a decomposition of rank 50.
    # Base: Strassen 2x2x2 (rank 7)
    # U, V, W shapes: (4, 7), (4, 7), (4, 7)
    
    # Strassen 2x2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(f-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C00 = M1 - M4 - M6 + M7
    # C01 = M3 + M4 + M5
    # C10 = M2 + M6
    # C11 = M1 - M3 - M2 + M5
    
    # Indices for 2x2x2:
    # U index: i*2+j. (0,1,2,3)
    # V index: j*2+k. (0,1,2,3)
    # W index: k*2+i. (0,1,2,3)
    
    # We need to map the local 2x2x2 decomposition to the global indices.
    
    # Helper to create a 2x2x2 block decomposition
    def get_strassen_2x2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1: (a+d)(e+h)
        # a: U[0], d: U[3]
        # e: V[0], h: V[3]
        # C00: W[0], C11: W[3]
        # C00 coeff: 1, C11 coeff: 1
        # W[0] += 1, W[3] += 1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2: (c+d)e
        # c: U[2], d: U[3]
        # e: V[0]
        # C10: W[1] (k=0, i=1 -> 0*2+1=1)
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1
        
        # M3: a(f-h)
        # a: U[0]
        # f: V[1], h: V[3]
        # C01: W[2] (k=1, i=0 -> 1*2+0=2)
        # C01 coeff: 1
        # Wait, M3 is added to C01.
        # C01 = M3 + M4 + M5
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[2, 2] = 1
        
        # M4: d(f-e)
        # d: U[3]
        # f: V[1], e: V[0]
        # C00: W[0], C01: W[2]
        # C00 coeff: -1, C01 coeff: 1
        U[3, 3] = 1
        V[1, 3] = 1; V[0, 3] = -1
        W[0, 3] = -1; W[2, 3] = 1
        
        # M5: (a+b)h
        # a: U[0], b: U[1]
        # h: V[3]
        # C01: W[2], C11: W[3]
        # C01 coeff: 1, C11 coeff: 1
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[2, 4] = 1; W[3, 4] = 1
        
        # M6: (c-a)(e+f)
        # c: U[2], a: U[0]
        # e: V[0], f: V[1]
        # C00: W[0], C10: W[1]
        # C00 coeff: -1, C10 coeff: 1
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[0, 5] = -1; W[1, 5] = 1
        
        # M7: (b-d)(g+h)
        # b: U[1], d: U[3]
        # g: V[2], h: V[3]
        # C00: W[0], C11: W[3]
        # C00 coeff: 1, C11 coeff: 1
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1; W[3, 6] = 1
        
        return U, V, W

    # Helper to create 2x2x1 decomposition (schoolbook)
    def get_2x2x1():
        U = np.zeros((4, 4))
        V = np.zeros((4, 4))
        W = np.zeros((2, 4)) # p=1, n=2 -> 1*2=2 rows
        
        # C0 = a*e + b*f (B is 2x1, so B[0]=e, B[1]=f)
        # V indices: j*1+k. j=0,1. k=0.
        # V[0] is B[0], V[1] is B[1].
        # W indices: k*n+i. k=0. i=0,1.
        # W[0] is C0, W[1] is C1.
        
        # M1: a*e
        U[0, 0] = 1
        V[0, 0] = 1
        W[0, 0] = 1
        
        # M2: b*f
        U[1, 1] = 1
        V[1, 1] = 1
        W[0, 1] = 1
        
        # M3: c*e
        U[2, 2] = 1
        V[0, 2] = 1
        W[1, 2] = 1
        
        # M4: d*f
        U[3, 3] = 1
        V[1, 3] = 1
        W[1, 3] = 1
        
        return U, V, W

    # We need to assemble the full 2x4x7 decomposition.
    # Total rank 50.
    # U: 8x50, V: 28x50, W: 14x50
    
    U_total = np.zeros((8, 50))
    V_total = np.zeros((28, 50))
    W_total = np.zeros((14, 50))
    
    rank_offset = 0
    
    # Add 2x2x2 block
    def add_block(U_block, V_block, W_block, i_off, j_off, k_off):
        # i_off: row offset in A (0 or 2)
        # j_off: col offset in A / row offset in B (0, 2)
        # k_off: col offset in B (0, 2, 4, 6)
        
        nonlocal rank_offset
        r = U_block.shape[1]
        
        # Map U indices: local (0..3) to global (i_off*2 + local_i)*2 + local_j?
        # A is 2x4. Index is i*4+j.
        # Local A is 2x2. i in 0,1. j in 0,1.
        # Global i = i_off + local_i. Global j = j_off + local_j.
        # Global U index = (i_off + local_i)*4 + (j_off + local_j).
        
        # Map V indices: local (0..3) to global.
        # Local B is 2x2. j in 0,1. k in 0,1.
        # Global j = j_off + local_j. Global k = k_off + local_k.
        # V index is j*7+k? No, V is m x p. m=4, p=7.
        # V index = j*7 + k.
        # Global V index = (j_off + local_j)*7 + (k_off + local_k).
        
        # Map W indices: local (0..3) to global.
        # Local C is 2x2. k in 0,1. i in 0,1.
        # Global k = k_off + local_k. Global i = i_off + local_i.
        # W index is k*n+i. n=2.
        # W index = k*2 + i.
        # Global W index = (k_off + local_k)*2 + (i_off + local_i).
        
        for c in range(r):
            for li in range(2):
                for lj in range(2):
                    lu = li*2 + lj
                    if U_block[lu, c] != 0:
                        gu = (i_off + li)*4 + (j_off + lj)
                        U_total[gu, rank_offset+c] += U_block[lu, c]
                        
            for lj in range(2):
                for lk in range(2):
                    lv = lj*2 + lk
                    if V_block[lv, c] != 0:
                        gv = (j_off + lj)*7 + (k_off + lk)
                        V_total[gv, rank_offset+c] += V_block[lv, c]
                        
            for lk in range(2):
                for li in range(2):
                    lw = lk*2 + li
                    if W_block[lw, c] != 0:
                        gw = (k_off + lk)*2 + (i_off + li)
                        W_total[gw, rank_offset+c] += W_block[lw, c]
                        
        rank_offset += r

    # Add 2x2x1 block
    def add_block_2x2x1(U_block, V_block, W_block, i_off, j_off, k_off):
        nonlocal rank_offset
        r = U_block.shape[1]
        
        for c in range(r):
            for li in range(2):
                for lj in range(2):
                    lu = li*2 + lj
                    if U_block[lu, c] != 0:
                        gu = (i_off + li)*4 + (j_off + lj)
                        U_total[gu, rank_offset+c] += U_block[lu, c]
                        
            for lj in range(2):
                lv = lj # k=0
                if V_block[lv, c] != 0:
                    gv = (j_off + lj)*7 + k_off
                    V_total[gv, rank_offset+c] += V_block[lv, c]
                        
            for li in range(2):
                lw = li # k=0
                if W_block[lw, c] != 0:
                    gw = k_off*2 + (i_off + li)
                    W_total[gw, rank_offset+c] += W_block[lw, c]
                        
        rank_offset += r

    S222_U, S222_V, S222_W = get_strassen_2x2x2()
    S221_U, S221_V, S221_W = get_2x2x1()

    # 2x4x4 part (i_off 0,2; j_off 0,2; k_off 0,2)
    # 4 blocks of 2x2x2
    add_block(S222_U, S222_V, S222_W, 0, 0, 0)
    add_block(S222_U, S222_V, S222_W, 0, 2, 0)
    add_block(S222_U, S222_V, S222_W, 2, 0, 0)
    add_block(S222_U, S222_V, S222_W, 2, 2, 0)
    
    # 2x4x3 part (i_off 0,2; j_off 0,2; k_off 4)
    # 2 blocks of 2x2x2 (k_off 4, 6) and 2 blocks of 2x2x1 (k_off 6? No)
    # Wait, 3 columns.
    # k=4,5,6.
    # Split into 2x2 (k=4,5) and 2x1 (k=6).
    # So for each i_off, j_off:
    # add 2x2x2 at k_off=4
    # add 2x2x1 at k_off=6
    
    add_block(S222_U, S222_V, S222_W, 0, 0, 4)
    add_block_2x2x1(S221_U, S221_V, S221_W, 0, 0, 6)
    add_block(S222_U, S222_V, S222_W, 0, 2, 4)
    add_block_2x2x1(S221_U, S221_V, S221_W, 0, 2, 6)
    add_block(S222_U, S222_V, S222_W, 2, 0, 4)
    add_block_2x2x1(S221_U, S221_V, S221_W, 2, 0, 6)
    add_block(S222_U, S222_V, S222_W, 2, 2, 4)
    add_block_2x2x1(S221_U, S221_V, S221_W, 2, 2, 6)

    return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}

# EVOLVE-BLOCK-END
