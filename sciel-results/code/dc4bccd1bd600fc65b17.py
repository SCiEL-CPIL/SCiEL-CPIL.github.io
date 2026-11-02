"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition for
    multiplying a 2x3 matrix by a 3x7 matrix.
    
    Strategy: Hybrid Strassen/Schoolbook.
    - Split the 3x7 product into a 2x7 and a 1x7 part.
    - For the 2x2 sub-blocks, use Strassen (rank 7 instead of 8).
    - For the remaining parts, use schoolbook.
    
    Dimensions: A is 2x3, B is 3x7.
    Let A = [A1 | A2] where A1 is 2x2, A2 is 2x1.
    Let B = [B1; B2] where B1 is 2x7, B2 is 1x7.
    
    C = A1*B1 + A2*B2
    
    For A1*B1 (2x2 * 2x7):
      Split B1 into three 2x2 blocks (cols 0-1, 2-3, 4-5) and one 2x1 block (col 6).
      Each 2x2 block uses Strassen (rank 7).
      The 2x1 block uses schoolbook (rank 4).
      Total: 3*7 + 4 = 25.
    
    A2*B2 (2x1 * 1x7): Schoolbook (rank 14).
    
    Total: 25 + 14 = 39.
    """
    n, m, p = DIMENSIONS  # 2, 3, 7
    
    # We'll build U, V, W matrices.
    # U has shape (n*m, rank) = (6, rank)
    # V has shape (m*p, rank) = (21, rank)
    # W has shape (p*n, rank) = (14, rank)
    
    # Let's collect the decomposition terms.
    # Each term is (u_vec, v_vec, w_vec) where u_vec has length 6, v_vec has length 21, w_vec has length 14.
    
    terms = []
    
    def add_term(u, v, w):
        """Add a term to the decomposition."""
        # u, v, w are dicts mapping index to value
        u_list = [0.0] * (n * m)
        v_list = [0.0] * (m * p)
        w_list = [0.0] * (p * n)
        for idx, val in u.items():
            u_list[idx] = val
        for idx, val in v.items():
            v_list[idx] = val
        for idx, val in w.items():
            w_list[idx] = val
        terms.append((u_list, v_list, w_list))
    
    # Helper to create index for U: row i, col j -> i*m + j
    def u_idx(i, j):
        return i * m + j
    
    # Helper to create index for V: row j, col k -> j*p + k
    def v_idx(j, k):
        return j * p + k
    
    # Helper to create index for W: row k, col i -> k*n + i
    def w_idx(k, i):
        return k * n + i
    
    # Part 1: A1 * B1[:,0:2] (2x2 * 2x2) using Strassen
    # A1 = [[a00, a01], [a10, a11]]
    # B1_block = [[B[0,0], B[0,1]], [B[1,0], B[1,1]]]
    # Strassen formulas:
    # M1 = (a00 + a11)(b00 + b11)
    # M2 = (a10 + a11)b00
    # M3 = a00(b01 - b11)
    # M4 = a11(b10 - b00)
    # M5 = (a00 + a01)b11
    # M6 = (a10 - a00)(b00 + b01)
    # M7 = (a01 - a11)(b10 + b11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each Strassen term, we have a product of a linear combination of A entries 
    # and a linear combination of B entries, contributing to specific C entries.
    
    # Strassen term 1: (a00 + a11)(b00 + b11)
    # U part: a00 -> u_idx(0,0)=0, a11 -> u_idx(1,1)=4
    # V part: b00 -> v_idx(0,0)=0, b11 -> v_idx(1,1)=8
    # W part: contributes to C00 (k=0,i=0) -> w_idx(0,0)=0, C01 (k=1,i=0) -> w_idx(1,0)=1,
    #          C10 (k=0,i=1) -> w_idx(0,1)=2, C11 (k=1,i=1) -> w_idx(1,1)=3
    # Coefficients: C00: +1, C01: 0, C10: 0, C11: +1 (from M1 in C00 and C11)
    add_term(
        {u_idx(0,0): 1, u_idx(1,1): 1},
        {v_idx(0,0): 1, v_idx(1,1): 1},
        {w_idx(0,0): 1, w_idx(1,1): 1}
    )
    
    # Strassen term 2: (a10 + a11)b00
    # U part: a10 -> u_idx(1,0)=3, a11 -> u_idx(1,1)=4
    # V part: b00 -> v_idx(0,0)=0
    # W part: C10: +1, C11: -1
    add_term(
        {u_idx(1,0): 1, u_idx(1,1): 1},
        {v_idx(0,0): 1},
        {w_idx(0,1): 1, w_idx(1,1): -1}
    )
    
    # Strassen term 3: a00(b01 - b11)
    # U part: a00 -> u_idx(0,0)=0
    # V part: b01 -> v_idx(0,1)=1, b11 -> v_idx(1,1)=8
    # W part: C01: +1, C11: +1
    add_term(
        {u_idx(0,0): 1},
        {v_idx(0,1): 1, v_idx(1,1): -1},
        {w_idx(1,0): 1, w_idx(1,1): 1}
    )
    
    # Strassen term 4: a11(b10 - b00)
    # U part: a11 -> u_idx(1,1)=4
    # V part: b10 -> v_idx(1,0)=7, b00 -> v_idx(0,0)=0
    # W part: C00: +1, C10: +1
    add_term(
        {u_idx(1,1): 1},
        {v_idx(1,0): 1, v_idx(0,0): -1},
        {w_idx(0,0): 1, w_idx(0,1): 1}
    )
    
    # Strassen term 5: (a00 + a01)b11
    # U part: a00 -> u_idx(0,0)=0, a01 -> u_idx(0,1)=1
    # V part: b11 -> v_idx(1,1)=8
    # W part: C00: -1, C01: +1
    add_term(
        {u_idx(0,0): 1, u_idx(0,1): 1},
        {v_idx(1,1): 1},
        {w_idx(0,0): -1, w_idx(1,0): 1}
    )
    
    # Strassen term 6: (a10 - a00)(b00 + b01)
    # U part: a10 -> u_idx(1,0)=3, a00 -> u_idx(0,0)=0
    # V part: b00 -> v_idx(0,0)=0, b01 -> v_idx(0,1)=1
    # W part: C11: +1
    add_term(
        {u_idx(1,0): 1, u_idx(0,0): -1},
        {v_idx(0,0): 1, v_idx(0,1): 1},
        {w_idx(1,1): 1}
    )
    
    # Strassen term 7: (a01 - a11)(b10 + b11)
    # U part: a01 -> u_idx(0,1)=1, a11 -> u_idx(1,1)=4
    # V part: b10 -> v_idx(1,0)=7, b11 -> v_idx(1,1)=8
    # W part: C00: +1
    add_term(
        {u_idx(0,1): 1, u_idx(1,1): -1},
        {v_idx(1,0): 1, v_idx(1,1): 1},
        {w_idx(0,0): 1}
    )
    
    # Part 2: A1 * B1[:,2:4] (2x2 * 2x2) using Strassen
    # B1_block = [[B[0,2], B[0,3]], [B[1,2], B[1,3]]]
    
    # Strassen term 1: (a00 + a11)(b02 + b13)
    add_term(
        {u_idx(0,0): 1, u_idx(1,1): 1},
        {v_idx(0,2): 1, v_idx(1,3): 1},
        {w_idx(2,0): 1, w_idx(3,1): 1}
    )
    
    # Strassen term 2: (a10 + a11)b02
    add_term(
        {u_idx(1,0): 1, u_idx(1,1): 1},
        {v_idx(0,2): 1},
        {w_idx(2,1): 1, w_idx(3,1): -1}
    )
    
    # Strassen term 3: a00(b03 - b13)
    add_term(
        {u_idx(0,0): 1},
        {v_idx(0,3): 1, v_idx(1,3): -1},
        {w_idx(3,0): 1, w_idx(3,1): 1}
    )
    
    # Strassen term 4: a11(b12 - b02)
    add_term(
        {u_idx(1,1): 1},
        {v_idx(1,2): 1, v_idx(0,2): -1},
        {w_idx(2,0): 1, w_idx(2,1): 1}
    )
    
    # Strassen term 5: (a00 + a01)b13
    add_term(
        {u_idx(0,0): 1, u_idx(0,1): 1},
        {v_idx(1,3): 1},
        {w_idx(2,0): -1, w_idx(3,0): 1}
    )
    
    # Strassen term 6: (a10 - a00)(b02 + b03)
    add_term(
        {u_idx(1,0): 1, u_idx(0,0): -1},
        {v_idx(0,2): 1, v_idx(0,3): 1},
        {w_idx(3,1): 1}
    )
    
    # Strassen term 7: (a01 - a11)(b12 + b13)
    add_term(
        {u_idx(0,1): 1, u_idx(1,1): -1},
        {v_idx(1,2): 1, v_idx(1,3): 1},
        {w_idx(2,0): 1}
    )
    
    # Part 3: A1 * B1[:,4:6] (2x2 * 2x2) using Strassen
    # B1_block = [[B[0,4], B[0,5]], [B[1,4], B[1,5]]]
    
    # Strassen term 1: (a00 + a11)(b04 + b15)
    add_term(
        {u_idx(0,0): 1, u_idx(1,1): 1},
        {v_idx(0,4): 1, v_idx(1,5): 1},
        {w_idx(4,0): 1, w_idx(5,1): 1}
    )
    
    # Strassen term 2: (a10 + a11)b04
    add_term(
        {u_idx(1,0): 1, u_idx(1,1): 1},
        {v_idx(0,4): 1},
        {w_idx(4,1): 1, w_idx(5,1): -1}
    )
    
    # Strassen term 3: a00(b05 - b15)
    add_term(
        {u_idx(0,0): 1},
        {v_idx(0,5): 1, v_idx(1,5): -1},
        {w_idx(5,0): 1, w_idx(5,1): 1}
    )
    
    # Strassen term 4: a11(b14 - b04)
    add_term(
        {u_idx(1,1): 1},
        {v_idx(1,4): 1, v_idx(0,4): -1},
        {w_idx(4,0): 1, w_idx(4,1): 1}
    )
    
    # Strassen term 5: (a00 + a01)b15
    add_term(
        {u_idx(0,0): 1, u_idx(0,1): 1},
        {v_idx(1,5): 1},
        {w_idx(4,0): -1, w_idx(5,0): 1}
    )
    
    # Strassen term 6: (a10 - a00)(b04 + b05)
    add_term(
        {u_idx(1,0): 1, u_idx(0,0): -1},
        {v_idx(0,4): 1, v_idx(0,5): 1},
        {w_idx(5,1): 1}
    )
    
    # Strassen term 7: (a01 - a11)(b14 + b15)
    add_term(
        {u_idx(0,1): 1, u_idx(1,1): -1},
        {v_idx(1,4): 1, v_idx(1,5): 1},
        {w_idx(4,0): 1}
    )
    
    # Part 4: A1 * B1[:,6] (2x2 * 2x1) using schoolbook
    # C0 = a00*b06 + a01*b16
    # C1 = a10*b06 + a11*b16
    # where b06=B[0,6], b16=B[1,6]
    
    # Term: a00*b06 -> C[0,6]
    add_term(
        {u_idx(0,0): 1},
        {v_idx(0,6): 1},
        {w_idx(6,0): 1}
    )
    
    # Term: a01*b16 -> C[0,6]
    add_term(
        {u_idx(0,1): 1},
        {v_idx(1,6): 1},
        {w_idx(6,0): 1}
    )
    
    # Term: a10*b06 -> C[1,6]
    add_term(
        {u_idx(1,0): 1},
        {v_idx(0,6): 1},
        {w_idx(6,1): 1}
    )
    
    # Term: a11*b16 -> C[1,6]
    add_term(
        {u_idx(1,1): 1},
        {v_idx(1,6): 1},
        {w_idx(6,1): 1}
    )
    
    # Part 5: A2 * B2 (2x1 * 1x7) using schoolbook
    # A2 = [a02, a12] (column vector)
    # B2 = [B[2,0], B[2,1], ..., B[2,6]] (row vector)
    # C[i,k] = A[i,2] * B[2,k] for i=0,1 and k=0..6
    
    for k in range(p):
        # a02 * b2k -> C[0,k]
        add_term(
            {u_idx(0,2): 1},
            {v_idx(2,k): 1},
            {w_idx(k,0): 1}
        )
        
        # a12 * b2k -> C[1,k]
        add_term(
            {u_idx(1,2): 1},
            {v_idx(2,k): 1},
            {w_idx(k,1): 1}
        )
    
    # Now assemble U, V, W
    rank = len(terms)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for c, (u_list, v_list, w_list) in enumerate(terms):
        for i in range(n * m):
            U[i][c] = u_list[i]
        for i in range(m * p):
            V[i][c] = v_list[i]
        for i in range(p * n):
            W[i][c] = w_list[i]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
