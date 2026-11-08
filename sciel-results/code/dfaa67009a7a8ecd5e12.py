import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

# Precompute the decomposition at module level
_DECOMP = None

def _build_decomposition():
    """
    Build the rank-84 decomposition:
      - 4x4x4 part via 3x3x3 rank-23 blocks (23) + 3 schoolbook (9) = 32
      - 4x4x3 part via schoolbook = 48
    Total rank = 32 + 48 = 80... let me recount.
    
    Actually for 4x4x4: we split 4x4 into 3x3 + 1, and 4x4 output into 3x3 + 1.
    The 4x4x4 tensor decomposes as:
      (3x3x3) ⊗ (3x3x3) for the top-left 3x3 block of A and C with B[0:3,0:3]
    ... this gets complicated. Let me use a simpler approach.
    
    For 4x4x4: Use the known rank-32 decomposition.
    Actually, the simplest improvement: use rank-23 for 3x3x3 (Larsen) 
    and handle the 4th row/col with schoolbook.
    
    4x4x4 = 3x3x3 (rank 23) + terms for row/col 3
    = 23 + 3*4 + 3*4 + 4 = 23 + 12 + 12 + 4 = 51? No that's worse.
    
    Let me use the direct rank-32 for 4x4x4 which is known.
    Or better: just use the 7-term Strassen for 4x4x4 which gives rank 49,
    but that's what we already have.
    
    Let me try a different approach: use the 3x3x3 rank-23 decomposition
    embedded in 4x4x4 with padding, plus schoolbook for the 4th indices.
    
    For 4x4x4 with 3x3x3 core:
    - Core: 3x3x3 using rank-23 (Larsen) decomposition
    - A[3,:], B[:,3], C[3,3] need schoolbook terms
    - Specifically: sum over i,j,k for (i,j,k) where at least one index = 3
      that's 4*4*4 - 3*3*3 = 64 - 27 = 37 schoolbook terms
    Total: 23 + 37 = 60. Worse than 49.
    
    OK let me just keep rank 49 for 4x4x4 and focus on the 4x4x3 part.
    The 4x4x3 part is currently schoolbook (48). Can we do better?
    
    Actually, let me reconsider. The current score is 1.1546 = 112/97.
    If I can get to rank 96, score = 112/96 = 1.1667.
    If I can get to rank 92, score = 112/92 = 1.2174.
    
    For the 4x4x3 part: A is 4x4, B is 4x3, C is 4x3.
    This is equivalent to multiplying 4x4 by 4x3.
    Schoolbook: 4*4*3 = 48.
    
    Can we use Strassen-like for 4x4x3? 
    Split 4x4 into 2x2 blocks, 4x3 into 2x2 and 2x1 blocks.
    This is more complex. Let me try a simpler optimization.
    
    Actually, let me just add a deadline guard and keep the same decomposition.
    The main improvement would be finding a lower-rank decomposition, but
    that's a research problem. Let me at least add the deadline guard.
    """
    n, m, p = 4, 4, 7
    
    # 7-term Strassen-like for 4x4x4
    u_in = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
        [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]
    ]
    v_in = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0],
        [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
    ]
    w_out = [
        [1, 0, 0, 1, -1, 0, 1], [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0], [1, -1, 1, 0, 0, 1, 0]
    ]
    u_out = u_in[:]
    v_out = v_in[:]
    w_outer = w_out[:]
    
    rank_strassen = 49
    rank_schoolbook = 4 * 4 * 3
    total_rank = rank_strassen + rank_schoolbook
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for k_outer in range(7):
        for l_inner in range(7):
            u_vec = [0] * 16
            for i_inner in range(2):
                for j_inner in range(2):
                    ci = u_in[l_inner][i_inner * 2 + j_inner]
                    if ci == 0:
                        continue
                    for b in range(4):
                        co = u_out[k_outer][b]
                        if co == 0:
                            continue
                        br, bc = b // 2, b % 2
                        ig = 2 * br + i_inner
                        jg = 2 * bc + j_inner
                        u_vec[ig * 4 + jg] += ci * co
            
            v_vec = [0] * 28
            for i_inner in range(2):
                for j_inner in range(2):
                    ci = v_in[l_inner][i_inner * 2 + j_inner]
                    if ci == 0:
                        continue
                    for b in range(4):
                        co = v_out[k_outer][b]
                        if co == 0:
                            continue
                        br, bc = b // 2, b % 2
                        jg = 2 * br + i_inner
                        kg = 2 * bc + j_inner
                        v_vec[jg * 7 + kg] += ci * co
            
            w_vec = [0] * 28
            for cb in range(4):
                cow = w_outer[cb][k_outer]
                if cow == 0:
                    continue
                for ci2 in range(4):
                    ciw = w_out[ci2][l_inner]
                    if ciw == 0:
                        continue
                    tc = cow * ciw
                    if tc == 0:
                        continue
                    cbr, cbc = cb // 2, cb % 2
                    cir, cic = ci2 // 2, ci2 % 2
                    ig = 2 * cbr + cir
                    kg = 2 * cbc + cic
                    w_vec[kg * 4 + ig] += tc
            
            for r in range(16):
                U[r][col] = u_vec[r]
            for r in range(28):
                V[r][col] = v_vec[r]
            for r in range(28):
                W[r][col] = w_vec[r]
            col += 1
    
    for i in range(4):
        for j in range(4):
            for k_prime in range(3):
                k_global = 4 + k_prime
                U[i * 4 + j][col] = 1
                V[j * 7 + k_global][col] = 1
                W[k_global * 4 + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

_DECOMP = _build_decomposition()

def construct():
    """
    Construct a rank-97 exact tensor decomposition for multiplying
    a 4x4 matrix by a 4x7 matrix.
    
    Interface: construct() -> {'U', 'V', 'W'}
    Shapes: U (n*m, rank), V (m*p, rank), W (p*n, rank)
    Convention: W[k*n+i, r]
    """
    return _DECOMP
    

