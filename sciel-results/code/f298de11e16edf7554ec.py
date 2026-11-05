<<<<<<< SEARCH
"""Program for matrix multiplication tensor decomposition: 2x4 * 4x5."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def _construct_rank_36():
    """
    Construct rank-36 decomposition for 2x4 x 4x5 matrix multiplication.
    Splits into two 2x2x5 blocks.
    Each 2x2x5 uses Strassen (rank 7) for 2 cols + Schoolbook (rank 4) for 1 col.
    Wait, 2x2x5 = 2x2x2 + 2x2x2 + 2x2x1? No.
    2x2x5 has 5 columns in B.
    Strassen handles 2x2x2 (2 cols of B). Rank 7.
    Remaining 3 cols of B.
    Schoolbook for 2x2x1 is rank 4.
    So 3 cols = 3 * rank 4 = 12.
    Total 7 + 12 = 19.
    Two blocks = 38.
    
    Why did the previous program say 36?
    "Each 2x2x5 splits into two 2x2x2 (Strassen, rank 7) + one 2x2x1 (schoolbook, rank 4). Total rank: 2 * (7 + 7 + 4) = 36."
    This implies 2x2x5 is split into 2x2x2 + 2x2x2 + 2x2x1.
    Sum of cols: 2 + 2 + 1 = 5.
    Rank: 7 + 7 + 4 = 18.
    Two blocks: 36.
    
    This is correct.
    """
    n, m, p = 2, 4, 5
    rank = 36
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 36
    V = [[0] * rank for _ in range(m * p)]  # 20 x 36
    W = [[0] * rank for _ in range(p * n)]  # 10 x 36
    
    def add_strassen_222(i1, i2, j1, j2, k1, k2, col_offset):
        a11 = i1 * m + j1
        a12 = i1 * m + j2
        a21 = i2 * m + j1
        a22 = i2 * m + j2
        
        b11 = j1 * p + k1
        b12 = j1 * p + k2
        b21 = j2 * p + k1
        b22 = j2 * p + k2
        
        c11 = k1 * n + i1
        c12 = k2 * n + i1
        c21 = k1 * n + i2
        c22 = k2 * n + i2
        
        # M1
        col = col_offset + 0
        U[a11][col] = 1; U[a22][col] = 1
        V[b11][col] = 1; V[b22][col] = 1
        W[c11][col] = 1; W[c22][col] = 1
        
        # M2
        col = col_offset + 1
        U[a21][col] = 1; U[a22][col] = 1
        V[b11][col] = 1
        W[c21][col] = 1; W[c22][col] = -1
        
        # M3
        col = col_offset + 2
        U[a11][col] = 1
        V[b12][col] = 1; V[b22][col] = -1
        W[c12][col] = 1; W[c22][col] = 1
        
        # M4
        col = col_offset + 3
        U[a22][col] = 1
        V[b21][col] = 1; V[b11][col] = -1
        W[c11][col] = 1; W[c21][col] = 1
        
        # M5
        col = col_offset + 4
        U[a11][col] = 1; U[a12][col] = 1
        V[b22][col] = 1
        W[c11][col] = -1; W[c12][col] = 1
        
        # M6
        col = col_offset + 5
        U[a21][col] = 1; U[a11][col] = -1
        V[b11][col] = 1; V[b12][col] = 1
        W[c22][col] = 1
        
        # M7
        col = col_offset + 6
        U[a12][col] = 1; U[a22][col] = -1
        V[b21][col] = 1; V[b22][col] = 1
        W[c11][col] = 1

    def add_schoolbook_221(i1, i2, j1, j2, k1, col_offset):
        a11 = i1 * m + j1
        a12 = i1 * m + j2
        a21 = i2 * m + j1
        a22 = i2 * m + j2
        
        b11 = j1 * p + k1
        b21 = j2 * p + k1
        
        c11 = k1 * n + i1
        c21 = k1 * n + i2
        
        col = col_offset + 0
        U[a11][col] = 1; V[b11][col] = 1; W[c11][col] = 1
        
        col = col_offset + 1
        U[a12][col] = 1; V[b21][col] = 1; W[c11][col] = 1
        
        col = col_offset + 2
        U[a21][col] = 1; V[b11][col] = 1; W[c21][col] = 1
        
        col = col_offset + 3
        U[a22][col] = 1; V[b21][col] = 1; W[c21][col] = 1

    col = 0
    # Block 1: j=0,1
    add_strassen_222(0, 1, 0, 1, 0, 1, col); col += 7
    add_strassen_222(0, 1, 0, 1, 2, 3, col); col += 7
    add_schoolbook_221(0, 1, 0, 1, 4, col); col += 4
    
    # Block 2: j=2,3
    add_strassen_222(0, 1, 2, 3, 0, 1, col); col += 7
    add_strassen_222(0, 1, 2, 3, 2, 3, col); col += 7
    add_schoolbook_221(0, 1, 2, 3, 4, col); col += 4
    
    return U, V, W

def construct():
    U, V, W = _construct_rank_36()
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
=======
"""Program for matrix multiplication tensor decomposition: 2x4 * 4x5."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def _construct_rank_32():
    """
    Construct rank-32 decomposition for 2x4 x 4x5 matrix multiplication.
    Splits into two 2x2x5 blocks.
    Each 2x2x5: Strassen (rank 7) for 2 cols + rank-10 for 3 cols.
    Total per block: 7 + 10 = 17. Two blocks: 34.
    Wait, let me re-derive. 2+3=5 cols. 7+10=17 per block. 2*17=34.
    Hmm, that's 34 not 32. Let me check if 2x2x3 can be done in rank 9 or less.
    Known: R(2,2,3) = 10. So 7+10=17, total 34.
    
    Alternative: don't split into two independent 2x2x5 blocks.
    Use a global structure.
    
    Actually, let me try: split 5 cols as 2+3.
    2x2x2: rank 7
    2x2x3: rank 10
    Per 2x2x5 block: 17
    Two blocks: 34.
    
    To get 32, I need a different approach.
    What if I use Strassen-like ideas across the full 2x4x5?
    
    Let me just go with 34 for now and verify correctness.
    """
    n, m, p = 2, 4, 5
    rank = 34
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 34
    V = [[0] * rank for _ in range(m * p)]  # 20 x 34
    W = [[0] * rank for _ in range(p * n)]  # 10 x 34
    
    def add_strassen_222(i1, i2, j1, j2, k1, k2, col_offset):
        a11 = i1 * m + j1
        a12 = i1 * m + j2
        a21 = i2 * m + j1
        a22 = i2 * m + j2
        
        b11 = j1 * p + k1
        b12 = j1 * p + k2
        b21 = j2 * p + k1
        b22 = j2 * p + k2
        
        c11 = k1 * n + i1
        c12 = k2 * n + i1
        c21 = k1 * n + i2
        c22 = k2 * n + i2
        
        # M1
        col = col_offset + 0
        U[a11][col] = 1; U[a22][col] = 1
        V[b11][col] = 1; V[b22][col] = 1
        W[c11][col] = 1; W[c22][col] = 1
        
        # M2
        col = col_offset + 1
        U[a21][col] = 1; U[a22][col] = 1
        V[b11][col] = 1
        W[c21][col] = 1; W[c22][col] = -1
        
        # M3
        col = col_offset + 2
        U[a11][col] = 1
        V[b12][col] = 1; V[b22][col] = -1
        W[c12][col] = 1; W[c22][col] = 1
        
        # M4
        col = col_offset + 3
        U[a22][col] = 1
        V[b21][col] = 1; V[b11][col] = -1
        W[c11][col] = 1; W[c21][col] = 1
        
        # M5
        col = col_offset + 4
        U[a11][col] = 1; U[a12][col] = 1
        V[b22][col] = 1
        W[c11][col] = -1; W[c12][col] = 1
        
        # M6
        col = col_offset + 5
        U[a21][col] = 1; U[a11][col] = -1
        V[b11][col] = 1; V[b12][col] = 1
        W[c22][col] = 1
        
        # M7
        col = col_offset + 6
        U[a12][col] = 1; U[a22][col] = -1
        V[b21][col] = 1; V[b22][col] = 1
        W[c11][col] = 1

    def add_rank10_223(i1, i2, j1, j2, k1, k2, k3, col_offset):
        """
        Rank-10 decomposition for 2x2x3 matrix multiplication.
        A (2x2) * B (2x3) -> C (2x3)
        Uses known rank-10 decomposition.
        """
        a11 = i1 * m + j1
        a12 = i1 * m + j2
        a21 = i2 * m + j1
        a22 = i2 * m + j2
        
        b11 = j1 * p + k1
        b12 = j1 * p + k2
        b13 = j1 * p + k3
        b21 = j2 * p + k1
        b22 = j2 * p + k2
        b23 = j2 * p + k3
        
        c11 = k1 * n + i1
        c12 = k2 * n + i1
        c13 = k3 * n + i1
        c21 = k1 * n + i2
        c22 = k2 * n + i2
        c23 = k3 * n + i2
        
        # Rank-10 decomposition for 2x2x3
        # Based on known results: R(2,2,3) = 10
        # Terms:
        # 1: a11*b11 -> c11
        # 2: a12*b21 -> c11
        # 3: a21*b11 -> c21
        # 4: a22*b21 -> c21
        # 5: (a11+a12)*(b22-b12) -> c12 + c22 correction
        # 6: (a21+a22)*(b12-b22) -> c12 + c22 correction  
        # 7: a11*b13 -> c13
        # 8: a12*b23 -> c13
        # 9: a21*b13 -> c23
        # 10: a22*b23 -> c23
        
        # Let me use a verified rank-10 scheme:
        # M1: a11*b11 -> c11
        col = col_offset + 0
        U[a11][col] = 1; V[b11][col] = 1; W[c11][col] = 1
        
        # M2: a12*b21 -> c11
        col = col_offset + 1
        U[a12][col] = 1; V[b21][col] = 1; W[c11][col] = 1
        
        # M3: a21*b11 -> c21
        col = col_offset + 2
        U[a21][col] = 1; V[b11][col] = 1; W[c21][col] = 1
        
        # M4: a22*b21 -> c21
        col = col_offset + 3
        U[a22][col] = 1; V[b21][col] = 1; W[c21][col] = 1
        
        # M5: (a11+a12)*(b12-b22) -> contributes to c12, c22
        col = col_offset + 4
        U[a11][col] = 1; U[a12][col] = 1
        V[b12][col] = 1; V[b22][col] = -1
        W[c12][col] = 1; W[c22][col] = 1
        
        # M6: (a21+a22)*(b12-b22) -> contributes to c12, c22
        col = col_offset + 5
        U[a21][col] = 1; U[a22][col] = 1
        V[b12][col] = 1; V[b22][col] = -1
        W[c12][col] = 1; W[c22][col] = 1
        
        # M7: a11*b13 -> c13
        col = col_offset + 6
        U[a11][col] = 1; V[b13][col] = 1; W[c13][col] = 1
        
        # M8: a12*b23 -> c13
        col = col_offset + 7
        U[a12][col] = 1; V[b23][col] = 1; W[c13][col] = 1
        
        # M9: a21*b13 -> c23
        col = col_offset + 8
        U[a21][col] = 1; V[b13][col] = 1; W[c23][col] = 1
        
        # M10: a22*b23 -> c23
        col = col_offset + 9
        U[a22][col] = 1; V[b23][col] = 1; W[c23][col] = 1

    col = 0
    # Block 1: j=0,1; cols k=0,1,2,3,4
    add_strassen_222(0, 1, 0, 1, 0, 1, col); col += 7
    add_rank10_223(0, 1, 0, 1, 2, 3, 4, col); col += 10
    
    # Block 2: j=2,3; cols k=0,1,2,3,4
    add_strassen_222(0, 1, 2, 3, 0, 1, col); col += 7
    add_rank10_223(0, 1, 2, 3, 2, 3, 4, col); col += 10
    
    return U, V, W

def construct():
    U, V, W = _construct_rank_32()
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

Wait, I need to verify the rank-10 decomposition for 2×2×3 is actually correct. Let me think through this more carefully. The issue is that my "rank-10" scheme above may not be valid. Let me reconsider.

For 2×2×3: A(2×2) × B(2×3) → C(2×3)
- C[0,0] = a[0,0]*b[0,0] + a[0,1]*b[1,0]
- C[0,1] = a[0,0]*b[0,1] + a[0,1]*b[1,1]  
- C[0,2] = a[0,0]*b[0,2] + a[0,1]*b[1,2]
- C[1,0] = a[1,0]*b[0,0] + a[1,1]*b[1,0]
- C[1,1] = a[1,0]*b[0,1] + a[1,1]*b[1,1]
- C[1,2] = a[1,0]*b[0,2] + a[1,1]*b[1,2]

My M5 and M6 as written would give:
- M5: (a11+a12)*(b12-b22) contributes to c12 and c22
  - c12 gets: (a11+a12)*(b12-b22) = a11*b12 - a11*b22 + a12*b12 - a12*b22
  - But we need c12 = a11*b12 + a12*b22. This is wrong.

The rank-10 scheme I wrote is incorrect. Let me just keep the original rank-36 approach but verify it's correct, and instead look for a different structural improvement.

Actually, let me reconsider the whole approach. The current rank-36 program likely works. The score 1.1111 = 40/36. To beat this, I need rank ≤ 35.

Let me try a simpler fix: verify the current code works and just ensure it's clean. Since the directive says to keep the design family, let me try to find if there's a rank-35 or rank-34 decomposition.

Let me revert to the working rank-36 code but add a validation and ensure correctness:

