"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen decomposition for 2x2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-f)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # P1 = M1 + M4 - M5 + M7
    # P2 = M3 + M5
    # P3 = M1 + M2 - M4 + M6
    # P4 = M2 + M3 - M1 + M7
    
    # Coefficients for A (U) and B (V) for each M
    # Indices: a=0, b=1, c=2, d=3
    # Indices: e=0, f=1, g=2, h=3
    
    U_strassen = [
        [1, 0, 1, 0, 1, 1, 0], # a
        [0, 0, 0, 0, 1, -1, 1], # b
        [0, 1, 0, 0, 0, 1, 0], # c
        [1, 1, 0, 1, 0, -1, -1] # d
    ]
    
    V_strassen = [
        [1, 1, 0, 0, 0, 1, 0], # e
        [0, 0, 1, -1, 0, 1, 0], # f
        [0, 0, 0, 1, 0, 0, 1], # g
        [1, 0, -1, 0, 1, 0, 1] # h
    ]
    
    W_strassen = [
        [1, 0, 1, 0], # P1
        [0, 1, 0, 0], # P2
        [1, 0, 0, 0], # P3
        [0, 1, 0, 1]  # P4
    ]
    # Note: W_strassen maps M_k to P_l.
    # P1 = M1 + M4 - M5 + M7
    # P2 = M3 + M5
    # P3 = M1 + M2 - M4 + M6
    # P4 = M2 + M3 - M1 + M7
    
    # Correct W_strassen based on linear combo of M's
    # Rows are P's, Cols are M's
    W_strassen_correct = np.array([
        [1, 0, 0, 1, -1, 0, 1], # P1
        [0, 0, 1, 0, 1, 0, 0],  # P2
        [1, 1, 0, -1, 0, 1, 0], # P3
        [-1, 1, 1, 0, 0, 0, 1]  # P4
    ])
    
    # Rank 1 decomposition for 2x1x2 (or 2x1x6 block)
    # A = [a0, a1], B = [b0, b1]
    # C = [[a0*b0, a0*b1], [a1*b0, a1*b1]]
    # U = [1, 1]
    # V = [1, 1]
    # W = [1, 1, 1, 1]
    
    U_rank1 = [1, 1]
    V_rank1 = [1, 1]
    W_rank1 = [1, 1, 1, 1]
    
    # Total Rank
    # J1: 3 Strassen blocks (21)
    # J2: 3 Strassen blocks (21)
    # J3: 1 Rank-1 block (1)
    total_rank = 43
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    col = 0
    
    # Helper to add Strassen block
    def add_strassen(i_offset, j_offset, k_offset):
        nonlocal col
        # A indices: i in {0,1}, j in {j_offset, j_offset+1}
        # U index: i*m + j
        for i in range(2):
            for j_local in range(2):
                u_idx = i * m + (j_offset + j_local)
                U[u_idx, col:col+7] = U_strassen[j_local] # Wait, U_strassen is indexed by a,b,c,d
                # a=0, b=1, c=2, d=3
                # j_local 0 -> j_offset, j_local 1 -> j_offset+1
                # i=0 -> top row (a,b), i=1 -> bottom row (c,d)
                # So for i=0, we use rows 0,1 of U_strassen?
                # U_strassen[0] is coeffs for a.
                # a is at i=0, j=0.
                # So U[i*m + j_offset + j_local, col+k] = U_strassen[2*i + j_local][k]
        
        # Let's redo the mapping carefully
        # U_strassen is 4x7. Rows: a, b, c, d.
        # a: i=0, j=0. b: i=0, j=1. c: i=1, j=0. d: i=1, j=1.
        for i in range(2):
            for j_local in range(2):
                row_idx = 2 * i + j_local
                u_idx = i * m + (j_offset + j_local)
                U[u_idx, col:col+7] = U_strassen[row_idx]
        
        # V indices: j in {j_offset, j_offset+1}, k in {k_offset, k_offset+1}
        # V index: j*p + k
        # e=0, f=1, g=2, h=3
        # e: j=0, k=0. f: j=0, k=1. g: j=1, k=0. h: j=1, k=1.
        for j_local in range(2):
            for k_local in range(2):
                row_idx = 2 * j_local + k_local
                v_idx = (j_offset + j_local) * p + (k_offset + k_local)
                V[v_idx, col:col+7] = V_strassen[row_idx]
        
        # W indices: k in {k_offset, k_offset+1}, i in {0,1}
        # W index: k*n + i
        # P1, P2, P3, P4
        # P1: k=0, i=0? No.
        # C = [[P1, P2], [P3, P4]]
        # P1 is C[0,0]. P2 is C[0,1]. P3 is C[1,0]. P4 is C[1,1].
        # W index for C[i,k] is k*n + i.
        # P1 (i=0, k=0): idx 0*2+0 = 0
        # P2 (i=0, k=1): idx 1*2+0 = 2
        # P3 (i=1, k=0): idx 0*2+1 = 1
        # P4 (i=1, k=1): idx 1*2+1 = 3
        
        # W_strassen_correct is 4x7. Rows P1..P4.
        # We need to map P1->idx0, P2->idx2, P3->idx1, P4->idx3.
        # And add k_offset.
        # k=0 -> k_offset. k=1 -> k_offset+1.
        
        # P1 (k=0, i=0): W[k_offset*2 + 0, col:col+7] += W_strassen_correct[0]
        # P2 (k=1, i=0): W[(k_offset+1)*2 + 0, col:col+7] += W_strassen_correct[1]
        # P3 (k=0, i=1): W[k_offset*2 + 1, col:col+7] += W_strassen_correct[2]
        # P4 (k=1, i=1): W[(k_offset+1)*2 + 1, col:col+7] += W_strassen_correct[3]
        
        W[k_offset * n + 0, col:col+7] += W_strassen_correct[0]
        W[(k_offset + 1) * n + 0, col:col+7] += W_strassen_correct[1]
        W[k_offset * n + 1, col:col+7] += W_strassen_correct[2]
        W[(k_offset + 1) * n + 1, col:col+7] += W_strassen_correct[3]
        
        col += 7

    # Helper to add Rank-1 block (2x1x2)
    def add_rank1(i_offset, j_offset, k_offset):
        nonlocal col
        # A: i in {0,1}, j=j_offset
        # U: [1, 1]
        U[i_offset * m + j_offset, col] = 1
        U[(i_offset + 1) * m + j_offset, col] = 1
        
        # B: j=j_offset, k in {k_offset, k_offset+1}
        # V: [1, 1]
        V[j_offset * p + k_offset, col] = 1
        V[j_offset * p + k_offset + 1, col] = 1
        
        # C: i in {0,1}, k in {k_offset, k_offset+1}
        # W: [1, 1, 1, 1]
        # C[0, k_offset]: idx k_offset*2 + 0
        # C[1, k_offset]: idx k_offset*2 + 1
        # C[0, k_offset+1]: idx (k_offset+1)*2 + 0
        # C[1, k_offset+1]: idx (k_offset+1)*2 + 1
        
        W[k_offset * n + 0, col] = 1
        W[k_offset * n + 1, col] = 1
        W[(k_offset + 1) * n + 0, col] = 1
        W[(k_offset + 1) * n + 1, col] = 1
        
        col += 1

    # J1: j in {0, 1}
    # K1: k in {0, 1}
    add_strassen(0, 0, 0)
    # K2: k in {2, 3}
    add_strassen(0, 0, 2)
    # K3: k in {4, 5}
    add_strassen(0, 0, 4)
    
    # J2: j in {2, 3}
    # K1: k in {0, 1}
    add_strassen(0, 2, 0)
    # K2: k in {2, 3}
    add_strassen(0, 2, 2)
    # K3: k in {4, 5}
    add_strassen(0, 2, 4)
    
    # J3: j in {4}
    # K1: k in {0, 1}
    add_rank1(0, 4, 0)
    # K2: k in {2, 3}
    add_rank1(0, 4, 2)
    # K3: k in {4, 5}
    add_rank1(0, 4, 4)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
