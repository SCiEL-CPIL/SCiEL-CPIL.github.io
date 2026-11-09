import time
import sys

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np

DEADLINE = time.time() + 145  # Safety margin before 150s limit

def construct():
    """
    Constructs a rank-65 tensor decomposition for 4x4 * 4x5 matrix multiplication.
    This is the minimal known rank, derived from the direct sum of:
    1. Rank 49 decomposition for 4x4 * 4x4 (Strassen)
    2. Rank 16 decomposition for 4x4 * 4x1 (Schoolbook)
    """
    if time.time() > DEADLINE:
        # Fallback: return a valid but potentially higher-rank solution
        return _construct_fallback()
    
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    rank = 65
    # U: (n*m, rank) -> (16, 65)
    # V: (m*p, rank) -> (20, 65)
    # W: (p*n, rank) -> (20, 65)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(col, u_list, v_list, w_list):
        for i, j, c in u_list:
            if c != 0:
                U[i * m + j][col] += c
        for j, k, c in v_list:
            if c != 0:
                V[j * p + k][col] += c
        for k, i, c in w_list:
            if c != 0:
                W[k * n + i][col] += c

    col = 0
    
    # --- Part 1: 4x4 * 4x4 using Strassen (49 terms) ---
    
    def get_indices(super_idx, inner_idx):
        u_list = []
        v_list = []
        w_list = []
        
        # Coefficients for A blocks in P (super term)
        # p_blocks: [coeff_A00, coeff_A01, coeff_A10, coeff_A11]
        if super_idx == 0: p_blocks = [1, 0, 0, 1]
        elif super_idx == 1: p_blocks = [0, 0, 1, 1]
        elif super_idx == 2: p_blocks = [1, 0, 0, 0]
        elif super_idx == 3: p_blocks = [0, 0, 0, 1]
        elif super_idx == 4: p_blocks = [1, 1, 0, 0]
        elif super_idx == 5: p_blocks = [-1, 0, 1, 0]
        elif super_idx == 6: p_blocks = [0, 1, 0, -1]
        
        # Coefficients for B blocks in Q (super term)
        # q_blocks: [coeff_B00, coeff_B01, coeff_B10, coeff_B11]
        if super_idx == 0: q_blocks = [1, 0, 0, 1]
        elif super_idx == 1: q_blocks = [1, 0, 0, 0]
        elif super_idx == 2: q_blocks = [0, 1, 0, -1]
        elif super_idx == 3: q_blocks = [-1, 0, 1, 0]
        elif super_idx == 4: q_blocks = [0, 0, 0, 1]
        elif super_idx == 5: q_blocks = [1, 1, 0, 0]
        elif super_idx == 6: q_blocks = [0, 0, 1, 1]
        
        # Inner Strassen terms for 2x2 P, Q
        # P = [[P00, P01], [P10, P11]]
        # Q = [[Q00, Q01], [Q10, Q11]]
        
        # Inner term definitions (p_elems, q_elems, s_contrib)
        # p_elems: list of (elem_idx, coeff)
        # elem_idx: 0:00, 1:01, 2:10, 3:11
        
        if inner_idx == 0: # (P00+P11)(Q00+Q11)
            p_elems = [(0, 1), (3, 1)]
            q_elems = [(0, 1), (3, 1)]
            s_contrib = [1, 0, 0, 1] # S00, S01, S10, S11
        elif inner_idx == 1: # (P10+P11)Q00
            p_elems = [(2, 1), (3, 1)]
            q_elems = [(0, 1)]
            s_contrib = [0, 0, 1, -1]
        elif inner_idx == 2: # P00(Q01-Q11)
            p_elems = [(0, 1)]
            q_elems = [(1, 1), (3, -1)]
            s_contrib = [0, 1, 0, 1]
        elif inner_idx == 3: # P11(Q10-Q00)
            p_elems = [(3, 1)]
            q_elems = [(2, 1), (0, -1)]
            s_contrib = [1, 0, 1, 0]
        elif inner_idx == 4: # (P00+P01)Q11
            p_elems = [(0, 1), (1, 1)]
            q_elems = [(3, 1)]
            s_contrib = [-1, 1, 0, 0]
        elif inner_idx == 5: # (P10-P00)(Q00+Q01)
            p_elems = [(2, 1), (0, -1)]
            q_elems = [(0, 1), (1, 1)]
            s_contrib = [0, 0, 0, 1]
        elif inner_idx == 6: # (P01-P11)(Q10+Q11)
            p_elems = [(1, 1), (3, -1)]
            q_elems = [(2, 1), (3, 1)]
            s_contrib = [1, 0, 0, 0]
        
        # Build u_list
        for p_idx, p_coeff in p_elems:
            r = p_idx // 2
            c = p_idx % 2
            for b_idx, coeff in enumerate(p_blocks):
                if coeff == 0: continue
                if b_idx == 0: gi, gj = r, c
                elif b_idx == 1: gi, gj = r, 2+c
                elif b_idx == 2: gi, gj = 2+r, c
                elif b_idx == 3: gi, gj = 2+r, 2+c
                u_list.append((gi, gj, coeff * p_coeff))
        
        # Build v_list
        for q_idx, q_coeff in q_elems:
            r = q_idx // 2
            c = q_idx % 2
            for b_idx, coeff in enumerate(q_blocks):
                if coeff == 0: continue
                if b_idx == 0: gj, gk = r, c
                elif b_idx == 1: gj, gk = r, 2+c
                elif b_idx == 2: gj, gk = 2+r, c
                elif b_idx == 3: gj, gk = 2+r, 2+c
                v_list.append((gj, gk, coeff * q_coeff))
        
        # Build w_list
        # Map S contributions to C blocks
        if super_idx == 0: c_block_contrib = [1, 0, 0, 1]
        elif super_idx == 1: c_block_contrib = [0, 0, 1, -1]
        elif super_idx == 2: c_block_contrib = [0, 1, 0, 1]
        elif super_idx == 3: c_block_contrib = [1, 0, 1, 0]
        elif super_idx == 4: c_block_contrib = [-1, 1, 0, 0]
        elif super_idx == 5: c_block_contrib = [0, 0, 0, 1]
        elif super_idx == 6: c_block_contrib = [1, 0, 0, 0]
        
        for s_pos in range(4):
            s_coeff = s_contrib[s_pos]
            if s_coeff == 0: continue
            s_r = s_pos // 2
            s_c = s_pos % 2
            
            for c_b in range(4):
                c_coeff = c_block_contrib[c_b]
                if c_coeff == 0: continue
                
                total_coeff = s_coeff * c_coeff
                if total_coeff == 0: continue
                
                if c_b == 0: gi, gk = s_r, s_c
                elif c_b == 1: gi, gk = s_r, 2+s_c
                elif c_b == 2: gi, gk = 2+s_r, s_c
                elif c_b == 3: gi, gk = 2+s_r, 2+s_c
                
                w_list.append((gk, gi, total_coeff))
                
        return u_list, v_list, w_list

    for s in range(7):
        for t in range(7):
            u_list, v_list, w_list = get_indices(s, t)
            add_term(col, u_list, v_list, w_list)
            col += 1

    # --- Part 2: 4x4 * 4x1 using Schoolbook (16 terms) ---
    # A (4x4) * B_col (4x1)
    # C[i,4] = sum_j A[i,j] B[j,4]
    # Note: In 0-indexed, this is column 4.
    # B has columns 0,1,2,3,4.
    # B[j,4] is the 5th column.
    
    for i in range(4):
        for j in range(4):
            u_list = [(i, j, 1)]
            v_list = [(j, 4, 1)]
            w_list = [(4, i, 1)]
            add_term(col, u_list, v_list, w_list)
            col += 1

    # Check deadline before returning
    if time.time() > DEADLINE:
        return _construct_fallback()
    
    return {"U": U, "V": V, "W": W}

def _construct_fallback():
    """Fallback schoolbook construction with rank n*m*p = 80."""
    n, m, p = DIMENSIONS  # 4, 4, 5
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
# EVOLVE-BLOCK-END
