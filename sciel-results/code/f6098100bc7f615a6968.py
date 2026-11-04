import time
import sys
import numpy as np

def construct():
    n, m, p = 2, 3, 8
    rank = 30
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    def add_strassen_block(u_map, v_map, w_map):
        nonlocal col
        # M1: (A00+A11)(B00+B11)
        U[u_map[0]][col] += 1
        U[u_map[3]][col] += 1
        V[v_map[0]][col] += 1
        V[v_map[3]][col] += 1
        W[w_map[0]][col] += 1
        W[w_map[3]][col] += 1
        col += 1
        
        # M2: (A10+A11)B00
        U[u_map[2]][col] += 1
        U[u_map[3]][col] += 1
        V[v_map[0]][col] += 1
        W[w_map[2]][col] += 1
        W[w_map[3]][col] += -1
        col += 1
        
        # M3: A00(B01-B11)
        U[u_map[0]][col] += 1
        V[v_map[1]][col] += 1
        V[v_map[3]][col] += -1
        W[w_map[1]][col] += 1
        W[w_map[3]][col] += 1
        col += 1
        
        # M4: A11(B10-B00)
        U[u_map[3]][col] += 1
        V[v_map[2]][col] += 1
        V[v_map[0]][col] += -1
        W[w_map[0]][col] += 1
        W[w_map[2]][col] += 1
        col += 1
        
        # M5: (A00+A01)B11
        U[u_map[0]][col] += 1
        U[u_map[1]][col] += 1
        V[v_map[3]][col] += 1
        W[w_map[0]][col] += -1
        W[w_map[1]][col] += 1
        col += 1
        
        # M6: (A10-A00)(B00+B01)
        U[u_map[2]][col] += 1
        U[u_map[0]][col] += -1
        V[v_map[0]][col] += 1
        V[v_map[1]][col] += 1
        W[w_map[3]][col] += 1
        col += 1
        
        # M7: (A01-A11)(B10+B11)
        U[u_map[1]][col] += 1
        U[u_map[3]][col] += -1
        V[v_map[2]][col] += 1
        V[v_map[3]][col] += 1
        W[w_map[0]][col] += 1
        col += 1

    def add_rank1_block(u_idx, v_idx, w_idx):
        nonlocal col
        U[u_idx][col] = 1
        V[v_idx][col] = 1
        W[w_idx][col] = 1
        col += 1

    # Split p=8 into p_L=4, p_R=4
    for half in range(2):
        # p_offset is 0 or 4
        p_off = half * 4
        
        # Split m=3 into m_0=1, m_1=2
        # A_0 is col 0 of A (indices 0, 3)
        # A_1 is cols 1,2 of A (indices 1,2,4,5)
        
        # B_0 is row 0 of B (indices 0..3 + p_off)
        # B_1 is rows 1,2 of B (indices 4..11 + p_off)
        
        # C_0 is cols 0..3 of C (indices 0..7 + p_off*n)
        # C_1 is cols 0..3 of C (indices 0..7 + p_off*n)
        
        # 1. Rank 1 block: A_0 * B_0
        # A_0 indices: 0, 3
        # B_0 indices: 0+p_off, 1+p_off, 2+p_off, 3+p_off
        # C indices: 0+p_off*2, 1+p_off*2, 2+p_off*2, 3+p_off*2, 4+p_off*2, 5+p_off*2, 6+p_off*2, 7+p_off*2
        # Wait, C is 2x4. Indices k*n + i.
        # k=0..3, i=0..1.
        # C indices: (0)*2+0, (0)*2+1, (1)*2+0, (1)*2+1, (2)*2+0, (2)*2+1, (3)*2+0, (3)*2+1
        # Plus offset p_off * n? No, W index is k*n + i.
        # The columns of C are 0..3 for this block.
        # So W indices are 0..7.
        
        # But A_0 * B_0 produces a 2x4 matrix.
        # C_{ik} = A_{i0} B_{0k}.
        # This is rank 1.
        # U indices: 0, 3.
        # V indices: 0+p_off, 1+p_off, 2+p_off, 3+p_off.
        # W indices: 0, 1, 2, 3, 4, 5, 6, 7 (relative to block start).
        
        # We need to add this as a single term?
        # No, rank 1 means 1 term.
        # U[0][col] = 1, U[3][col] = 1
        # V[0+p_off][col] = 1, V[1+p_off][col] = 1, V[2+p_off][col] = 1, V[3+p_off][col] = 1
        # W[0][col] = 1, W[1][col] = 1, W[2][col] = 1, W[3][col] = 1, W[4][col] = 1, W[5][col] = 1, W[6][col] = 1, W[7][col] = 1
        
        # Wait, W indices are global.
        # For left half (p_off=0), W indices are 0..7.
        # For right half (p_off=4), W indices are 8..15.
        
        u_idx = [0, 3]
        v_idx = [0+p_off, 1+p_off, 2+p_off, 3+p_off]
        w_idx = [k*2 + i for k in range(4) for i in range(2)]
        w_idx = [w + p_off*2 for w in w_idx]
        
        for u in u_idx:
            U[u][col] = 1
        for v in v_idx:
            V[v][col] = 1
        for w in w_idx:
            W[w][col] = 1
        col += 1
        
        # 2. Rank 14 block: A_1 * B_1
        # A_1 is 2x2. Indices:
        # A_1[0,0] = 1, A_1[0,1] = 2
        # A_1[1,0] = 4, A_1[1,1] = 5
        u_map = {0: 1, 1: 2, 2: 4, 3: 5}
        
        # B_1 is 2x4. Indices:
        # Row 1: 4+p_off, 5+p_off, 6+p_off, 7+p_off
        # Row 2: 8+p_off, 9+p_off, 10+p_off, 11+p_off
        # We split B_1 into two 2x2 blocks: B_L and B_R.
        
        # B_L: cols 0,1 of B_1.
        # v_map_L:
        # 0: 4+p_off
        # 1: 5+p_off
        # 2: 8+p_off
        # 3: 9+p_off
        v_map_L = {0: 4+p_off, 1: 5+p_off, 2: 8+p_off, 3: 9+p_off}
        
        # W for C_L: cols 0,1 of C.
        # w_map_L:
        # 0: 0 + p_off*2
        # 1: 1 + p_off*2
        # 2: 2 + p_off*2
        # 3: 3 + p_off*2
        w_map_L = {0: 0 + p_off*2, 1: 1 + p_off*2, 2: 2 + p_off*2, 3: 3 + p_off*2}
        
        add_strassen_block(u_map, v_map_L, w_map_L)
        
        # B_R: cols 2,3 of B_1.
        # v_map_R:
        # 0: 6+p_off
        # 1: 7+p_off
        # 2: 10+p_off
        # 3: 11+p_off
        v_map_R = {0: 6+p_off, 1: 7+p_off, 2: 10+p_off, 3: 11+p_off}
        
        # W for C_R: cols 2,3 of C.
        # w_map_R:
        # 0: 4 + p_off*2
        # 1: 5 + p_off*2
        # 2: 6 + p_off*2
        # 3: 7 + p_off*2
        w_map_R = {0: 4 + p_off*2, 1: 5 + p_off*2, 2: 6 + p_off*2, 3: 7 + p_off*2}
        
        add_strassen_block(u_map, v_map_R, w_map_R)

    return {"U": U, "V": V, "W": W}
