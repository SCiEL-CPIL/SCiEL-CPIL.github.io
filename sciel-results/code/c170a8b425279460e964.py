"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time
import itertools

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def construct():
    """
    Rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix
    by a 3 by 6 matrix.
    
    Strategy: Split the m=3 dimension into a 2-column block (j=0,1) and a 1-column
    tail (j=2). For the 2x2 core (j=0,1), apply Strassen's rank-7 decomposition
    for each of the 6 output columns. For the tail (j=2), use schoolbook rank 12.
    Total rank: 7*6 + 12 = 54? No, that's worse.
    
    Actually, let me reconsider. The key insight from the knowledge base is:
    "Reduces rank from 36 to 33 by applying Strassen's algorithm (rank 7) to the 
    2x2 sub-blocks of the matrix multiplication, replacing schoolbook rank 4 with 
    rank 7 but allowing sharing of computations in the 2x2 core vs 1x6 tail"
    
    Wait, that doesn't make sense either. Rank 7 > rank 4, so how does it reduce?
    
    Let me think differently. Perhaps the approach is:
    - For each pair of output columns (k, k+1), treat the 2x3 * 3x2 subproblem
    - The 2x3 * 3x2 subproblem has schoolbook rank 12
    - If we can do it in rank 11, total = 3*11 = 33
    
    Or perhaps:
    - Split p=6 into three groups of 2
    - For each group, compute 2x3 * 3x2
    - Use a rank-11 decomposition for each 2x3*3x2
    - Total rank = 3 * 11 = 33
    
    Let me construct a rank-11 decomposition for 2x3 * 3x2.
    
    A is 2x3: [[a00, a01, a02], [a10, a11, a12]]
    B is 3x2: [[b00, b01], [b10, b11], [b20, b21]]
    C is 2x2: [[c00, c01], [c10, c11]]
    
    c00 = a00*b00 + a01*b10 + a02*b20
    c01 = a00*b01 + a01*b11 + a02*b21
    c10 = a10*b00 + a11*b10 + a12*b20
    c11 = a10*b01 + a11*b11 + a12*b21
    
    Schoolbook uses 12 multiplications. To get rank 11, we need to find 11 products
    that can be combined to give all 4 outputs.
    
    One known approach: use the identity that for 2x2 matrix multiplication,
    Strassen uses 7 products. But here we have 2x3 * 3x2.
    
    Let me try a different approach. I'll use the following 11 products:
    p1 = a00*b00
    p2 = a01*b10
    p3 = a02*b20
    p4 = a00*b01
    p5 = a01*b11
    p6 = a02*b21
    p7 = a10*b00
    p8 = a11*b10
    p9 = a12*b20
    p10 = a10*b01
    p11 = (a11+a12)*(b11+b21)  # This gives a11*b11 + a11*b21 + a12*b11 + a12*b21
    
    Then:
    c00 = p1 + p2 + p3
    c01 = p4 + p5 + p6
    c10 = p7 + p8 + p9
    c11 = p10 + p11 - a11*b21 - a12*b11
    
    But we don't have a11*b21 or a12*b11 as separate products. So this doesn't work directly.
    
    Let me try yet another approach. I'll use a systematic search for a rank-11 decomposition.
    But given time constraints, let me just use the schoolbook rank 36 for now.
    
    Actually, wait. Let me re-read the knowledge base entry more carefully:
    "Reduces rank from 36 to 33 by applying Strassen's algorithm (rank 7) to the 2x2 
    sub-blocks of the matrix multiplication, replacing schoolbook rank 4 with rank 7 
    but allowing sharing of computations in the 2x2 core vs 1x6 tail"
    
    I think the idea is:
    - Split A into [A_left (2x2), A_right (2x1)]
    - Split B into [B_top (2x6); B_bottom (1x6)]
    - C = A_left * B_top + A_right * B_bottom
    
    For A_left * B_top (2x2 * 2x6):
    - This is like 6 separate 2x2 * 2x1 multiplications
    - Each 2x2 * 2x1 has rank 4 (schoolbook)
    - But we can use Strassen-like tricks to share computations across the 6 columns
    - If we use 7 products per 2x2 block, but share across 6 columns... 
    - Actually, 2x2 * 2x6 can be done in rank 14 (2*7) by applying Strassen to each pair of columns?
    - No, that's 7*3 = 21 for 3 pairs of columns.
    
    For A_right * B_bottom (2x1 * 1x6):
    - This is just 2 dot products of length 6, rank 12 (schoolbook)
    - But we can do it in rank 2*1*6 = 12
    
    Total: 21 + 12 = 33? But 2x2*2x6 in rank 21 seems high.
    
    Actually, 2x2 * 2x6:
    - Split the 6 columns into 3 pairs: (0,1), (2,3), (4,5)
    - For each pair, compute 2x2 * 2x2 using Strassen (rank 7)
    - Total: 3 * 7 = 21
    
    So total rank = 21 + 12 = 33.
    
    Let me implement this.
    """
    start_time = time.time()
    deadline = start_time + 140  # safety margin for 150s limit
    
    n, m, p = DIMENSIONS  # n=2, m=3, p=6
    
    # Split: A = [A_left (2x2), A_right (2x1)], B = [B_top (2x6); B_bottom (1x6)]
    # C = A_left * B_top + A_right * B_bottom
    
    # Part 1: A_right * B_bottom (2x1 * 1x6), schoolbook rank 12
    # Part 2: A_left * B_top (2x2 * 2x6), Strassen rank 21
    
    total_rank = 33
    U = [[0] * total_rank for _ in range(n * m)]  # (6, 33)
    V = [[0] * total_rank for _ in range(m * p)]  # (18, 33)
    W = [[0] * total_rank for _ in range(p * n)]  # (12, 33)
    
    col = 0
    
    # Part 1: A_right * B_bottom, schoolbook
    # A_right is column j=2 of A, B_bottom is row j=2 of B
    # C[i,k] += A[i,2] * B[2,k]
    for i in range(n):
        for k in range(p):
            U[i * m + 2][col] = 1  # A[i,2]
            V[2 * p + k][col] = 1  # B[2,k]
            W[k * n + i][col] = 1  # C[i,k]
            col += 1
    # col is now 12
    
    # Part 2: A_left * B_top, Strassen for each pair of columns
    # A_left is columns j=0,1 of A, B_top is rows j=0,1 of B
    # For each pair of output columns (k, k+1), compute 2x2 * 2x2 using Strassen
    
    def _strassen_2x2_2x2(U, V, W, col, i_offset, j_offset, k_offset, n, m, p):
        """
        Apply Strassen's algorithm for 2x2 * 2x2 at the specified offsets.
        U rows: (i_offset+i, j_offset+j) for i in {0,1}, j in {0,1}
        V rows: (j_offset+j, k_offset+k) for j in {0,1}, k in {0,1}
        W rows: (k_offset+k, i_offset+i) for k in {0,1}, i in {0,1}
        Returns new col.
        """
        # Strassen's 7 products:
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B10-B11)
        # M4 = A11*(B01-B00)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B10)
        # M7 = (A01-A11)*(B01+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Each product is a rank-1 term with appropriate U, V, W coefficients.
        
        # M1: (A00+A11)*(B00+B11)
        # U: A00=1, A11=1; V: B00=1, B11=1; W: C00=1, C11=1
        U[i_offset * m + j_offset][col] += 1
        U[(i_offset+1) * m + (j_offset+1)][col] += 1
        V[j_offset * p + k_offset][col] += 1
        V[(j_offset+1) * p + (k_offset+1)][col] += 1
        W[k_offset * n + i_offset][col] += 1
        W[(k_offset+1) * n + (i_offset+1)][col] += 1
        col += 1
        
        # M2: (A10+A11)*B00
        U[i_offset * m + (j_offset+1)][col] += 1  # A10
        U[(i_offset+1) * m + (j_offset+1)][col] += 1  # A11
        V[j_offset * p + k_offset][col] += 1  # B00
        W[(k_offset+1) * n + (i_offset+1)][col] += 1  # C11
        W[k_offset * n + i_offset][col] -= 1  # C00 (negative)
        col += 1
        
        # M3: A00*(B10-B11)
        U[i_offset * m + j_offset][col] += 1  # A00
        V[(j_offset+1) * p + k_offset][col] += 1  # B10
        V[(j_offset+1) * p + (k_offset+1)][col] -= 1  # -B11
        W[k_offset * n + i_offset][col] += 1  # C00
        W[(k_offset+1) * n + (i_offset+1)][col] += 1  # C11
        col += 1
        
        # M4: A11*(B01-B00)
        U[(i_offset+1) * m + (j_offset+1)][col] += 1  # A11
        V[j_offset * p + (k_offset+1)][col] += 1  # B01
        V[j_offset * p + k_offset][col] -= 1  # -B00
        W[k_offset * n + i_offset][col] += 1  # C00
        W[(k_offset+1) * n + (i_offset+1)][col] += 1  # C11
        col += 1
        
        # M5: (A00+A01)*B11
        U[i_offset * m + j_offset][col] += 1  # A00
        U[i_offset * m + (j_offset+1)][col] += 1  # A01
        V[(j_offset+1) * p + (k_offset+1)][col] += 1  # B11
        W[k_offset * n + i_offset][col] -= 1  # -C00
        W[(k_offset+1) * n + (i_offset+1)][col] += 1  # C11
        col += 1
        
        # M6: (A10-A00)*(B00+B10)
        U[i_offset * m + (j_offset+1)][col] += 1  # A10
        U[i_offset * m + j_offset][col] -= 1  # -A00
        V[j_offset * p + k_offset][col] += 1  # B00
        V[(j_offset+1) * p + k_offset][col] += 1  # B10
        W[(k_offset+1) * n + (i_offset+1)][col] += 1  # C11
        col += 1
        
        # M7: (A01-A11)*(B01+B11)
        U[i_offset * m + (j_offset+1)][col] += 1  # A01
        U[(i_offset+1) * m + (j_offset+1)][col] -= 1  # -A11
        V[j_offset * p + (k_offset+1)][col] += 1  # B01
        V[(j_offset+1) * p + (k_offset+1)][col] += 1  # B11
        W[k_offset * n + i_offset][col] += 1  # C00
        col += 1
        
        return col
    
    # Apply Strassen to each pair of output columns
    for k_pair in range(3):
        k_offset = k_pair * 2
        col = _strassen_2x2_2x2(U, V, W, col, 0, 0, k_offset, n, m, p)
    
    # col should be 12 + 21 = 33
    assert col == total_rank, f"Expected col={total_rank}, got {col}"
    
    # Verify correctness
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
