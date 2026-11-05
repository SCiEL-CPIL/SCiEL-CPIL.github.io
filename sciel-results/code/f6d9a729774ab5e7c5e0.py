import numpy as np
import time
from fractions import Fraction

START_TIME = time.time()
TIME_LIMIT = 110  # Safety margin for 120s limit

def _time_exceeded():
    return (time.time() - START_TIME) > TIME_LIMIT

def _schoolbook(n, m, p):
    """
    Returns U, V, W for schoolbook multiplication of n x m by m x p.
    U shape: (n*m, n*m*p)
    V shape: (m*p, n*m*p)
    W shape: (p*n, n*m*p)
    """
    rank = n * m * p
    U = [[0.0 for _ in range(rank)] for _ in range(n * m)]
    V = [[0.0 for _ in range(rank)] for _ in range(m * p)]
    W = [[0.0 for _ in range(rank)] for _ in range(p * n)]
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][idx] = 1.0
                V[j * p + k][idx] = 1.0
                W[k * n + i][idx] = 1.0
                idx += 1
    return U, V, W

def _strassen():
    """
    Returns U, V, W for Strassen multiplication of 2 x 2 by 2 x 2.
    Rank 7.
    """
    # Standard Strassen decomposition
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to map this to U (A indices), V (B indices), W (C indices)
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3 (k*n+i: 0*2+0=0, 0*2+1=1, 1*2+0=2, 1*2+1=3)
    
    U = [[0.0 for _ in range(7)] for _ in range(4)]
    V = [[0.0 for _ in range(7)] for _ in range(4)]
    W = [[0.0 for _ in range(7)] for _ in range(4)]
    
    # r=0: M1
    U[0][0] = 1.0; U[3][0] = 1.0
    V[0][0] = 1.0; V[3][0] = 1.0
    W[0][0] = 1.0; W[3][0] = 1.0
    
    # r=1: M2
    U[2][1] = 1.0; U[3][1] = 1.0
    V[0][1] = 1.0
    W[2][1] = 1.0; W[3][1] = -1.0 # C21 += M2, C22 -= M2
    
    # r=2: M3
    U[0][2] = 1.0
    V[1][2] = 1.0; V[3][2] = -1.0
    W[1][2] = 1.0; W[3][2] = 1.0 # C12 += M3, C22 += M3
    
    # r=3: M4
    U[3][3] = 1.0
    V[2][3] = 1.0; V[0][3] = -1.0
    W[0][3] = 1.0; W[2][3] = 1.0 # C11 += M4, C21 += M4
    
    # r=4: M5
    U[0][4] = 1.0; U[1][4] = 1.0
    V[3][4] = 1.0
    W[0][4] = -1.0; W[1][4] = 1.0 # C11 -= M5, C12 += M5
    
    # r=5: M6
    U[2][5] = 1.0; U[0][5] = -1.0
    V[0][5] = 1.0; V[1][5] = 1.0
    W[3][5] = 1.0 # C22 += M6
    
    # r=6: M7
    U[1][6] = 1.0; U[3][6] = -1.0
    V[2][6] = 1.0; V[3][6] = 1.0
    W[0][6] = 1.0 # C11 += M7
    
    return U, V, W

def _bcl():
    """
    Returns U, V, W for Bini-Capovani-Lotti multiplication of 3 x 3 by 3 x 3.
    Rank 23.
    """
    # This is complex to hardcode correctly from memory. 
    # However, we can use a known list of coefficients.
    # Since I cannot look it up, I will use a fallback or a simpler known decomposition if 23 is too risky to hardcode blindly.
    # But wait, the score 1.0606 is achieved by rank 99. 
    # If I use schoolbook for 3x3x3 (rank 27), the total rank becomes:
    # 3x2x7 part: 40 (using Strassen for 2x2x2 parts)
    # 3x3x7 part: 
    #   3x3x3: 27
    #   3x3x4: 36 (two 3x3x2 schoolbook)
    #   Total 3x3x7: 63
    # Total: 40 + 63 = 103. Score 105/103 = 1.019. Still an improvement over 1.0.
    # But 99 is better.
    # Let's try to find the BCL coefficients or use a reliable source in code? No, no internet.
    # I will implement the rank 23 decomposition using a known table.
    
    # The BCL decomposition is well-known.
    # Reference: "On the rank of the matrix multiplication tensor"
    # Coefficients are integers.
    
    # I will use a placeholder structure that is valid but might not be rank 23 if I get it wrong.
    # Actually, I can just use schoolbook for 3x3x3 to be safe, yielding rank 103.
    # Or I can try to recall the structure.
    # Let's stick to the safe rank 103 first? No, the goal is to beat 1.0606 if possible, or at least match it.
    # The prompt says "improving". 1.0606 is the current best in inspirations.
    # If I produce 103, score is 1.019.
    # If I produce 99, score is 1.0606.
    
    # Let's try to hardcode the BCL coefficients.
    # There are 23 terms.
    # I will define them explicitly.
    
    U = [[0.0 for _ in range(23)] for _ in range(9)]
    V = [[0.0 for _ in range(23)] for _ in range(9)]
    W = [[0.0 for _ in range(23)] for _ in range(9)]
    
    # Indices for 3x3:
    # A: 0..8, B: 0..8, C: 0..8 (k*n+i)
    
    # I will use the coefficients from a known implementation.
    # If I am unsure, I will fall back to schoolbook.
    # Given the risk of hallucinating 23x3x3 coefficients, I will use schoolbook for 3x3x3.
    # Rank 103 is a valid improvement over 105.
    
    # However, the inspiration explicitly mentions score 1.0606 which is 105/99.
    # So the "island best" is likely 99.
    # To beat or match, I should try for 99.
    # I will implement the BCL coefficients carefully.
    
    # Coefficients for BCL 3x3x3 rank 23:
    # Source: https://en.wikipedia.org/wiki/Matrix_multiplication_algorithm#Bini%E2%80%93Capovani%E2%80%93Lotti
    # Or similar.
    
    # Let's define the 23 tensors.
    # t1: A00 B00 -> C00
    # ...
    
    # Actually, there is a simpler way. 
    # I can use the fact that 3x3x3 rank 23 is standard.
    # I will write the coefficients based on the standard list.
    
    # List of (u, v, w) indices for each of the 23 terms.
    # u in 0..8, v in 0..8, w in 0..8.
    
    terms = [
        # 1
        ([0], [0], [0]),
        # 2
        ([1, 3], [0], [1, 3]), # This is getting complicated.
    ]
    
    # To avoid errors, I will use schoolbook for 3x3x3.
    # Rank 103.
    # Score 1.019.
    # This is an improvement over 1.0.
    # It might not match the 1.0606 of the inspiration, but it is a valid step.
    # Wait, the directive says "Keep this design and investigate improvements".
    # The design is "block decomposition".
    # Using schoolbook for the 3x3x3 block is a valid instance of block decomposition.
    
    # Let's check if I can easily get 99.
    # If I use Strassen for 2x2x2 (rank 7) and Schoolbook for 3x3x3 (rank 27):
    # 3x2x7:
    #   3x2x3 (SB 18)
    #   3x2x4:
    #     1x2x4 (SB 8)
    #     2x2x4 (2x Strassen 14)
    #   Total 3x2x7: 40
    # 3x3x7:
    #   3x3x3 (SB 27)
    #   3x3x4 (2x SB 18 = 36)
    #   Total 3x3x7: 63
    # Total: 103.
    
    # If I use BCL (23) for 3x3x3:
    # Total: 40 + 23 + 36 = 99.
    
    # I will try to include the BCL coefficients. If I fail, I'll use schoolbook.
    # I will use a hardcoded list of non-zero entries for U, V, W.
    
    # BCL Coefficients (from literature):
    # I will define them as a list of tuples (u_idx, v_idx, w_idx, coeff)
    # Actually, each term r has multiple non-zeros in U, V, W.
    
    # Let's use a reliable source of the 23 terms.
    # Term 1: U[0]=1, V[0]=1, W[0]=1
    # Term 2: U[1]=1, V[1]=1, W[1]=1
    # ...
    
    # I will use the following known decomposition:
    # https://github.com/mattwala/bcl
    # Or I can just use schoolbook. The jump from 105 to 103 is small.
    # The jump to 99 is larger.
    
    # I will implement the BCL coefficients.
    
    # Note: The indices for W are k*n+i.
    # For 3x3x3:
    # A: i*m+j
    # B: j*p+k
    # C: k*n+i
    
    # I will hardcode the 23 terms.
    
    # Terms:
    # 1: A00 B00 C00
    # 2: A11 B11 C11
    # 3: A22 B22 C22
    # 4: (A00+A11)(B01+B10) -> C01, C10
    # ...
    
    # This is too error-prone to do from memory in a single step.
    # I will use schoolbook for 3x3x3.
    
    return _schoolbook(3, 3, 3)

def _combine(U1, V1, W1, U2, V2, W2, 
             n1, m1, p1, n2, m2, p2,
             n_total, m_total, p_total,
             i_offset, j_offset, k_offset):
    """
    Combines two decompositions.
    This helper is complex. It's better to build the full matrices directly.
    """
    pass

def construct():
    n, m, p = 3, 5, 7
    
    # We will build U, V, W of shape (n*m, rank), (m*p, rank), (p*n, rank)
    # Rank is sum of ranks of blocks.
    
    # Block 1: 3x2x7 (m=0..2)
    # Block 2: 3x3x7 (m=2..5)
    
    # We need to map the local indices to global indices.
    # Global A index: i*m + j
    # Global B index: j*p + k
    # Global C index: k*n + i
    
    # Let's create the matrices with the total rank.
    # We will fill them column by column for each block.
    
    # First, determine the sub-blocks and their ranks.
    
    # Block 1: 3x2x7
    # Sub 1a: 3x2x3 (p=0..3). Schoolbook. Rank 18.
    # Sub 1b: 3x2x4 (p=3..7).
    #   Sub 1b1: 1x2x4 (n=0..1). Schoolbook. Rank 8.
    #   Sub 1b2: 2x2x4 (n=1..3).
    #     Sub 1b2a: 2x2x2 (p=3..5). Strassen. Rank 7.
    #     Sub 1b2b: 2x2x2 (p=5..7). Strassen. Rank 7.
    
    # Block 2: 3x3x7 (m=2..5)
    # Sub 2a: 3x3x3 (p=0..3). Schoolbook (for safety). Rank 27.
    # Sub 2b: 3x3x4 (p=3..7).
    #   Sub 2b1: 3x3x2 (p=3..5). Schoolbook. Rank 18.
    #   Sub 2b2: 3x3x2 (p=5..7). Schoolbook. Rank 18.
    
    # Total Rank = 18 + 8 + 7 + 7 + 27 + 18 + 18 = 103.
    
    rank = 103
    U = [[0.0 for _ in range(rank)] for _ in range(n * m)]
    V = [[0.0 for _ in range(rank)] for _ in range(m * p)]
    W = [[0.0 for _ in range(rank)] for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a schoolbook block
    def add_schoolbook(n_s, m_s, p_s, i_off, j_off, k_off):
        nonlocal col
        for i in range(n_s):
            for j in range(m_s):
                for k in range(p_s):
                    gi = i_off + i
                    gj = j_off + j
                    gk = k_off + k
                    U[gi * m + gj][col] = 1.0
                    V[gj * p + gk][col] = 1.0
                    W[gk * n + gi][col] = 1.0
                    col += 1

    # Helper to add a strassen block (2x2x2)
    def add_strassen(i_off, j_off, k_off):
        nonlocal col
        # Get local U, V, W
        u_loc, v_loc, w_loc = _strassen()
        for r in range(7):
            for i in range(4):
                if u_loc[i][r] != 0:
                    gi = i_off + i
                    U[gi * 2 + 0][col] += u_loc[i][r] # Wait, m is local 2. Global m is 5.
                    # U index is i_global * m_global + j_global
                    # Here i is local i (0..1), j is local j (0..1)
                    # i_global = i_off + i
                    # j_global = j_off + j
                    # So U index = (i_off+i)*m + (j_off+j)
                    pass
            # This nested loop is inefficient and prone to error.
            # Better to iterate over non-zeros.
            
    # Let's rewrite the addition more carefully.
    
    # Reset col
    col = 0
    
    # 1a: 3x2x3 (i=0..3, j=0..2, k=0..3)
    add_schoolbook(3, 2, 3, 0, 0, 0)
    
    # 1b1: 1x2x4 (i=0..1, j=0..2, k=3..7)
    add_schoolbook(1, 2, 4, 0, 0, 3)
    
    # 1b2a: 2x2x2 (i=1..3, j=0..2, k=3..5)
    # Strassen
    u_s, v_s, w_s = _strassen()
    for r in range(7):
        for i in range(2):
            for j in range(2):
                val = u_s[i*2+j][r]
                if val != 0:
                    gi = 1 + i
                    gj = 0 + j
                    U[gi * m + gj][col] = val
        for j in range(2):
            for k in range(2):
                val = v_s[j*2+k][r]
                if val != 0:
                    gj = 0 + j
                    gk = 3 + k
                    V[gj * p + gk][col] = val
        for k in range(2):
            for i in range(2):
                val = w_s[k*2+i][r] # w index is k*n+i
                if val != 0:
                    gk = 3 + k
                    gi = 1 + i
                    W[gk * n + gi][col] = val
        col += 1
        
    # 1b2b: 2x2x2 (i=1..3, j=0..2, k=5..7)
    u_s, v_s, w_s = _strassen()
    for r in range(7):
        for i in range(2):
            for j in range(2):
                val = u_s[i*2+j][r]
                if val != 0:
                    gi = 1 + i
                    gj = 0 + j
                    U[gi * m + gj][col] = val
        for j in range(2):
            for k in range(2):
                val = v_s[j*2+k][r]
                if val != 0:
                    gj = 0 + j
                    gk = 5 + k
                    V[gj * p + gk][col] = val
        for k in range(2):
            for i in range(2):
                val = w_s[k*2+i][r]
                if val != 0:
                    gk = 5 + k
                    gi = 1 + i
                    W[gk * n + gi][col] = val
        col += 1
        
    # 2a: 3x3x3 (i=0..3, j=2..5, k=0..3)
    add_schoolbook(3, 3, 3, 0, 2, 0)
    
    # 2b1: 3x3x2 (i=0..3, j=2..5, k=3..5)
    add_schoolbook(3, 3, 2, 0, 2, 3)
    
    # 2b2: 3x3x2 (i=0..3, j=2..5, k=5..7)
    add_schoolbook(3, 3, 2, 0, 2, 5)
    
    return {'U': U, 'V': V, 'W': W}

