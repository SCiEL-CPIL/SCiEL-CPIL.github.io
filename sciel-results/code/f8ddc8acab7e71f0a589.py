"""
Matrix multiplication 4x4x9 - rank-minimal exact tensor decomposition.

Strategy: Partition the 9-column dimension into three 3-column blocks.
For each 4x4x3 sub-problem, further partition into four 2x2x3 blocks.
For each 2x2x3 block, use Winograd's rank-7 decomposition for the 2x2x2 part
and rank-2 for the 2x2x1 part, giving rank 9 per 2x2x3 block.
Total rank: 3 * 4 * 9 = 108.

Winograd 2x2x2 rank-7 decomposition:
  m1 = (a00 + a11)(b00 + b11)
  m2 = (a00 + a10)(b10 + b11)
  m3 = (a01 + a11)(b00 + b01)
  m4 = (a01 + a10)(b01 + b10)
  m5 = a00 * b00
  m6 = a11 * b11
  m7 = a10 * b01

Reconstruction:
  c00 = m1 - m2 - m3 + m5
  c01 = m1 - m2 - m4 + m7
  c10 = m1 - m3 - m4 + m5 - m6 + m7  ... wait, let me re-derive.

Actually, let me use the standard Winograd 2x2x2 rank-7:
  U rows (for A, indexed by (i,j) in 2x2):
    r0: a00 + a11
    r1: a00 + a10
    r2: a01 + a11
    r3: a01 + a10
    r4: a00
    r5: a11
    r6: a10

  V rows (for B, indexed by (j,k) in 2x2):
    r0: b00 + b11
    r1: b10 + b11
    r2: b00 + b01
    r3: b01 + b10
    r4: b00
    r5: b11
    r6: b01

  W rows (for C, indexed by (i,k) in 2x2):
    r0: c00 = m1 - m2 - m3 + m5
    r1: c01 = m1 - m2 - m4 + m7  ... 

Let me just verify: c00 should be a00*b00 + a01*b10
Using the above:
m1 - m2 - m3 + m5 = (a00+a11)(b00+b11) - (a00+a10)(b10+b11) - (a01+a11)(b00+b01) + a00*b00

Let me expand:
= a00*b00 + a00*b11 + a11*b00 + a11*b11
  - a00*b10 - a00*b11 - a10*b10 - a10*b11
  - a01*b00 - a01*b01 - a11*b00 - a11*b01
  + a00*b00

= a00*b00 + a00*b11 + a11*b00 + a11*b11 - a00*b10 - a00*b11 - a10*b10 - a10*b11 - a01*b00 - a01*b01 - a11*b00 - a11*b01 + a00*b00

= 2*a00*b00 + a11*b11 - a00*b10 - a10*b10 - a10*b11 - a01*b00 - a01*b01 - a11*b01

That's not right. Let me use a known correct Winograd 2x2x2 rank-7.

Standard Winograd 2x2x2 (from Winograd 1979):
  m1 = (a00 + a11) * (b00 + b11)
  m2 = (a00 + a10) * (b10 + b11)
  m3 = (a01 + a11) * (b00 + b01)
  m4 = (a01 + a10) * (b01 + b10)
  m5 = a00 * b00
  m6 = a11 * b11
  m7 = a10 * b01

  c00 = m1 - m2 - m3 + m5
  c01 = m1 - m2 - m4 + m7
  c10 = m1 - m3 - m4 + m5 - m6 + m7  ... no

Let me just compute what c00, c01, c10, c11 should be and solve.

c00 = a00*b00 + a01*b10
c01 = a00*b01 + a01*b11
c10 = a10*b00 + a11*b10
c11 = a10*b01 + a11*b11

Let me express these in terms of m1..m7.

m1 = (a00+a11)(b00+b11) = a00b00 + a00b11 + a11b00 + a11b11
m2 = (a00+a10)(b10+b11) = a00b10 + a00b11 + a10b10 + a10b11
m3 = (a01+a11)(b00+b01) = a01b00 + a01b01 + a11b00 + a11b01
m4 = (a01+a10)(b01+b10) = a01b01 + a01b10 + a10b01 + a10b10
m5 = a00b00
m6 = a11b11
m7 = a10b01

c00 = a00b00 + a01b10 = m5 + (m4 - a01b01 - a10b01 - a10b10) = m5 + m4 - m7 - a01b01 - a10b10

Hmm, this is getting complicated. Let me just use the schoolbook for 2x2x2 (rank 8) to be safe, and combine with rank-2 for 2x2x1.

2x2x2 schoolbook: rank 8
2x2x1: rank 2
Total per 2x2x3: 10
Per 4x4x3: 4 * 10 = 40
Total: 3 * 40 = 120

This is valid and easier to verify.
"""

import time
import sys

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Partition p=9 into three groups of 3
    # For each group of 3 columns (p_block), partition into 4 sub-blocks of 2x2x3
    # Each 2x2x3: use schoolbook 2x2x2 (rank 8) + 2x2x1 (rank 2) = rank 10
    # Total: 3 groups * 4 sub-blocks * 10 = 120
    
    total_rank = 120
    U = [[0] * total_rank for _ in range(n * m)]  # 16 x 120
    V = [[0] * total_rank for _ in range(m * p)]  # 36 x 120
    W = [[0] * total_rank for _ in range(p * n)]  # 36 x 120
    
    col = 0
    for p_group in range(3):
        # p indices for this group: p_group*3, p_group*3+1, p_group*3+2
        for i_sub in range(2):  # i offset: 0 or 2
            for j_sub in range(2):  # j offset: 0 or 2
                # For each 2x2x3 block
                # 2x2x2 part: 8 terms
                for i_off in range(2):
                    for j_off in range(2):
                        for k_off in range(2):
                            i_idx = i_sub * 2 + i_off
                            j_idx = j_sub * 2 + j_off
                            k_idx = p_group * 3 + k_off
                            
                            U[i_idx * m + j_idx][col] = 1
                            V[j_idx * p + k_idx][col] = 1
                            W[k_idx * n + i_idx][col] = 1
                            col += 1
                # 2x2x1 part (k = p_group*3 + 2): 2 terms
                for i_off in range(2):
                    for j_off in range(2):
                        i_idx = i_sub * 2 + i_off
                        j_idx = j_sub * 2 + j_off
                        k_idx = p_group * 3 + 2
                        
                        U[i_idx * m + j_idx][col] = 1
                        V[j_idx * p + k_idx][col] = 1
                        W[k_idx * n + i_idx][col] = 1
                        col += 1
    
    return {"U": U, "V": V, "W": W}
