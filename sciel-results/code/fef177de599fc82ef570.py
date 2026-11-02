"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-52 decomposition for multiplying a 3x3 matrix by a 3x6 matrix.
    
    Strategy: Decompose into blocks.
    A = [[A11, A12, A13],
         [A21, A22, A23],
         [A31, A32, A33]]
    
    B is 3x6. We can view B as having 6 columns.
    Let's use the fact that R(3,3,6) <= R(2,2,2) + R(2,2,6) + R(1,1,6) ... no, that's not how block decomposition works directly with ranks unless we split indices carefully.
    
    Actually, a standard block decomposition for matrix multiplication [n,m,p] -> [n1, m1, p1] + [n2, m2, p2] + ... requires the sum of the dimensions to match.
    However, tensor rank does not simply add for block diagonal structures in the same way because the indices overlap or are distinct.
    
    Let's try a different construction.
    We can use the "winograd" style or just a known small rank algorithm.
    
    Known:
    R(2,2,2) = 7 (Strassen)
    R(3,3,3) = 23 (Landsberg) - this is complex to hardcode.
    
    Alternative:
    Use the block decomposition:
    Multiply 3x3 by 3x6.
    Split the 3 rows of A into 2+1.
    Split the 3 cols of A into 2+1.
    Split the 6 cols of B into 4+2? Or 3+3?
    
    Let's use a simpler recursive block method that is known to be optimal or near-optimal for small sizes, even if not the absolute theoretical minimum, it must beat 54.
    
    Construction:
    A = [[A11 (2x2), A12 (2x1)],
         [A21 (1x2), A22 (1x1)]]
    
    B = [[B11 (2x6)],
         [B21 (1x6)]]
     
    C = A * B
    C11 = A11 B11 + A12 B21  (2x6)
    C12 = A11 B21 + A12 B31 ... wait, B is 3x6.
    Let's stick to row/col splits of A and B.
    
    A (3x3) = [[A00 (2x2), A01 (2x1)],
               [A10 (1x2), A11 (1x1)]]
               
    B (3x6) = [[B00 (2x6)],
               [B10 (1x6)]]
    
    C = A B
    C is 3x6.
    C = [[C00 (2x6)],
         [C10 (1x6)]]
         
    C00 = A00 B00 + A01 B10
    C10 = A10 B00 + A11 B10
    
    We need to compute:
    1. A00 B00 : [2,2,6]
    2. A01 B10 : [2,1,6]
    3. A10 B00 : [1,2,6]
    4. A11 B10 : [1,1,6]
    
    Ranks:
    R(2,2,6): Split 6 into 2+2+2?
    R(2,2,2) = 7.
    R(2,2,6) = R(2,2,2) + R(2,2,2) + R(2,2,2) = 7+7+7 = 21.
    (This works because we can split the output columns of B and C independently and sum the tensors).
    
    R(2,1,6): Split 6 into 2+2+2.
    R(2,1,2) = 4 (Schoolbook: 2*1*2).
    R(2,1,6) = 4+4+4 = 12.
    
    R(1,2,6): Split 6 into 2+2+2.
    R(1,2,2) = 4.
    R(1,2,6) = 4+4+4 = 12.
    
    R(1,1,6): Split 6 into 2+2+2.
    R(1,1,2) = 2.
    R(1,1,6) = 2+2+2 = 6.
    
    Total Rank = 21 + 12 + 12 + 6 = 51.
    
    This beats 52 and 54.
    Is it 51?
    Let's verify the block decomposition logic.
    
    The tensor for [n,m,p] can be decomposed into a direct sum of tensors for [n1,m1,p1], [n2,m2,p2] etc if the indices are disjoint.
    
    Indices for A (3x3):
    A00: rows 0-1, cols 0-1
    A01: rows 0-1, col 2
    A10: row 2, cols 0-1
    A11: row 2, col 2
    
    Indices for B (3x6):
    B00: rows 0-1, cols 0-5
    B10: row 2, cols 0-5
    
    Wait, in the formula C00 = A00 B00 + A01 B10:
    A00 is 2x2. B00 is 2x6. Result 2x6.
    A01 is 2x1. B10 is 1x6. Result 2x6.
    
    In the formula C10 = A10 B00 + A11 B10:
    A10 is 1x2. B00 is 2x6. Result 1x6.
    A11 is 1x1. B10 is 1x6. Result 1x6.
    
    So the terms are:
    T1 = A00 B00  -> [2,2,6]
    T2 = A01 B10  -> [2,1,6]
    T3 = A10 B00  -> [1,2,6]
    T4 = A11 B10  -> [1,1,6]
    
    Are the tensor indices disjoint?
    T1 uses A indices (0,1)x(0,1) and B indices (0,1)x(0..5) and C indices (0,1)x(0..5).
    T2 uses A indices (0,1)x(2) and B indices (2)x(0..5) and C indices (0,1)x(0..5).
    T3 uses A indices (2)x(0,1) and B indices (0,1)x(0..5) and C indices (2)x(0..5).
    T4 uses A indices (2)x(2) and B indices (2)x(0..5) and C indices (2)x(0..5).
    
    The A indices are disjoint across T1, T2, T3, T4?
    T1: A(0,0), A(0,1), A(1,0), A(1,1)
    T2: A(0,2), A(1,2)
    T3: A(2,0), A(2,1)
    T4: A(2,2)
    Yes, all 9 entries of A are used exactly once.
    
    The B indices:
    T1: B(0,0..5), B(1,0..5)
    T2: B(2,0..5)
    T3: B(0,0..5), B(1,0..5)
    T4: B(2,0..5)
    
    Wait. T1 and T3 both use B(0,0..5) and B(1,0..5).
    T2 and T4 both use B(2,0..5).
    
    This means the B indices are NOT disjoint.
    Therefore, we cannot simply sum the ranks. The tensor terms share the B inputs.
    If inputs are shared, the rank is NOT the sum of the ranks.
    The rank of the sum of two tensors with shared input indices is at most the sum of their ranks, but it is not guaranteed that we can achieve the sum. Actually, if the input spaces are the same, we can still add them, but the resulting tensor lives in a larger space.
    
    However, the definition of tensor rank for matrix multiplication is specific.
    The tensor $T$ is an element of $V_A \otimes V_B \otimes V_C$.
    If we decompose $T = T_1 + T_2 + T_3 + T_4$, and $T_i$ are rank $r_i$, then $T$ has rank $\le \sum r_i$.
    This upper bound holds regardless of whether indices are disjoint.
    So Rank $\le 21 + 12 + 12 + 6 = 51$.
    
    So 51 is a valid upper bound.
    
    Can we do better?
    Maybe split 6 into 3+3?
    R(2,2,3):
    R(2,2,3) = R(2,2,2) + R(2,2,1) = 7 + 4 = 11.
    
    If we split B's columns into 3+3:
    T1 = A00 B00 (2x3) -> Rank 11
    T2 = A01 B10 (2x3) -> Rank ?
    R(2,1,3) = R(2,1,2) + R(2,1,1) = 4 + 2 = 6.
    T3 = A10 B00 (1x3) -> Rank ?
    R(1,2,3) = R(1,2,2) + R(1,2,1) = 4 + 2 = 6.
    T4 = A11 B10 (1x3) -> Rank ?
    R(1,1,3) = R(1,1,2) + R(1,1,1) = 2 + 1 = 3.
    
    Total = 11 + 6 + 6 + 3 = 26?
    Wait, this is way too low.
    Why?
    Because B00 is 2x3, B10 is 1x3.
    T1 uses B00. T3 uses B00.
    T2 uses B10. T4 uses B10.
    
    Still, Rank(T1+T2+T3+T4) <= Sum of Ranks.
    So Rank <= 26?
    This implies R(3,3,6) <= 26?
    That seems unlikely. R(3,3,3) is 23. R(3,3,6) should be larger.
    Ah, the indices of C are also shared.
    But the upper bound logic holds: If T = T1+T2+T3+T4, and each Ti has rank ri, then T has rank <= sum ri.
    
    So why is this so low?
    Let's check R(2,2,3).
    Is R(2,2,3) = 11?
    R(2,2,2)=7. R(2,2,1)=4.
    Yes, 7+4=11.
    
    Is R(3,3,3) = 23?
    Using the same block logic for [3,3,3]:
    A = [[A00 (2x2), A01 (2x1)],
         [A10 (1x2), A11 (1x1)]]
    B = [[B00 (2x3)],
         [B10 (1x3)]]
     
    T1 = A00 B00 [2,2,3] -> 11
    T2 = A01 B10 [2,1,3] -> 6
    T3 = A10 B00 [1,2,3] -> 6
    T4 = A11 B10 [1,1,3] -> 3
    
    Sum = 11 + 6 + 6 + 3 = 26.
    So R(3,3,3) <= 26.
    Known R(3,3,3) = 23.
    So 26 is a valid upper bound, just not tight.
    
    For R(3,3,6):
    Sum = 21 + 12 + 12 + 6 = 51.
    Or using 3+3 split:
    T1: [2,2,3] -> 11 (for first 3 cols) + 11 (for next 3 cols) = 22?
    No, if we split 6 into 3+3, we calculate T1 for col 0-2 and T1' for col 3-5.
    T1 (cols 0-2): Rank 11.
    T1' (cols 3-5): Rank 11.
    Total for A00 B00 part: 22.
    
    T2 (A01 B10, cols 0-2): Rank 6.
    T2' (A01 B10, cols 3-5): Rank 6.
    Total for A01 B10 part: 12.
    
    T3 (A10 B00, cols 0-2): Rank 6.
    T3' (A10 B00, cols 3-5): Rank 6.
    Total for A10 B00 part: 12.
    
    T4 (A11 B10, cols 0-2): Rank 3.
    T4' (A11 B10, cols 3-5): Rank 3.
    Total for A11 B10 part: 6.
    
    Sum = 22 + 12 + 12 + 6 = 52.
    
    So the 2+2+2 split gave 51.
    The 3+3 split gave 52.
    The 2+2+1+1 split?
    R(2,2,1)=4.
    R(2,1,1)=2.
    R(1,2,1)=2.
    R(1,1,1)=1.
    Sum = 4+2+2+1 = 9 per "unit" of column block? No, the rank depends on the size.
    
    Let's stick with the 2+2+2 split for the 6 columns.
    Rank = 51.
    
    We need to construct the 51-rank decomposition.
    We need the explicit matrices for:
    1. [2,2,2] Strassen (Rank 7)
    2. [2,2,1] Schoolbook (Rank 4)
    3. [2,1,2] Schoolbook (Rank 4)
    4. [2,1,1] Schoolbook (Rank 2)
    5. [1,2,2] Schoolbook (Rank 4)
    6. [1,2,1] Schoolbook (Rank 2)
    7. [1,1,2] Schoolbook (Rank 2)
    8. [1,1,1] Schoolbook (Rank 1)
    
    And combine them.
    
    Let's define a helper function to generate a rank-R decomposition for [n,m,p] given a list of (U, V, W) triples.
    But U, V, W are matrices.
    It is easier to work with the "term" representation: a list of (u_vec, v_vec, w_vec).
    u_vec length n*m, v_vec length m*p, w_vec length p*n.
    
    We will generate the terms for each sub-problem, map the indices to the global indices, and append them to a global list.
    
    Sub-problem 1: A00 (2x2) * B00 (2x6) -> C00_part1 (2x6)
    A00 indices in A (3x3):
    A row 0 -> Global A row 0
    A row 1 -> Global A row 1
    A col 0 -> Global A col 0
    A col 1 -> Global A col 1
    
    B00 indices in B (3x6):
    B row 0 -> Global B row 0
    B row 1 -> Global B row 1
    B cols 0-5 -> Global B cols 0-5
    
    C00_part1 indices in C (3x6):
    C row 0 -> Global C row 0
    C row 1 -> Global C row 1
    C cols 0-5 -> Global C cols 0-5
    
    We need to decompose [2,2,6] into three [2,2,2]s.
    [2,2,6] = [2,2,2] (cols 0-1) + [2,2,2] (cols 2-3) + [2,2,2] (cols 4-5).
    
    For each [2,2,2] block, we use Strassen.
    
    Let's code this step by step.
    """
    import numpy as np
    
    n, m, p = DIMENSIONS # 3, 3, 6
    
    # Global terms list: (u, v, w)
    terms = []
    
    def add_term(u, v, w):
        terms.append((u, v, w))
        
    # Helper: Strassen decomposition for 2x2x2
    # Returns list of (u, v, w) of length 7
    # u, v, w are length 4 vectors (row-major)
    def strassen_2x2_2():
        # Strassen's 7 products
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Indices for 2x2 matrix (row-major):
        # 0: 00, 1: 01, 2: 10, 3: 11
        
        # M1
        u = [1, 0, 0, 1]
        v = [1, 0, 0, 1]
        w = [1, 0, 0, 1] # M1 contributes to C00 and C11
        # Wait, w is the output coefficient.
        # M1 adds to C00 (idx 0) and C11 (idx 3).
        w = [1, 0, 0, 1]
        add_term(u, v, w)
        
        # M2
        u = [0, 0, 1, 1]
        v = [1, 0, 0, 0]
        w = [0, 0, 1, -1] # C10 (idx 2) +1, C11 (idx 3) -1
        add_term(u, v, w)
        
        # M3
        u = [1, 0, 0, 0]
        v = [0, 1, 0, -1]
        w = [0, 1, 0, 0] # C01 (idx 1) +1
        add_term(u, v, w)
        
        # M4
        u = [0, 0, 0, 1]
        v = [0, 0, 1, -1]
        w = [1, 0, 1, 0] # C00 (idx 0) +1, C10 (idx 2) +1
        add_term(u, v, w)
        
        # M5
        u = [1, 1, 0, 0]
        v = [0, 0, 0, 1]
        w = [-1, 1, 0, 0] # C00 (idx 0) -1, C01 (idx 1) +1
        add_term(u, v, w)
        
        # M6
        u = [0, 0, 1, -1]
        v = [1, 1, 0, 0]
        w = [0, 0, 0, 1] # C11 (idx 3) +1
        add_term(u, v, w)
        
        # M7
        u = [0, 1, 0, -1]
        v = [0, 0, 1, 1]
        w = [1, 0, 0, 0] # C00 (idx 0) +1
        add_term(u, v, w)
        
        # Return count? No, just adds to global terms.
        return 7
        
    def schoolbook_2x2_1():
        # [2,2,1]
        # A is 2x2, B is 2x1, C is 2x1.
        # B indices: 0 (row 0), 1 (row 1).
        # C indices: 0 (row 0), 1 (row 1).
        # A indices: 00, 01, 10, 11.
        
        # C0 = A00 B0 + A01 B1
        # C1 = A10 B0 + A11 B1
        
        # Term 1: A00 B0
        u = [1, 0, 0, 0]
        v = [1, 0]
        w = [1, 0]
        add_term(u, v, w)
        
        # Term 2: A01 B1
        u = [0, 1, 0, 0]
        v = [0, 1]
        w = [1, 0]
        add_term(u, v, w)
        
        # Term 3: A10 B0
        u = [0, 0, 1, 0]
        v = [1, 0]
        w = [0, 1]
        add_term(u, v, w)
        
        # Term 4: A11 B1
        u = [0, 0, 0, 1]
        v = [0, 1]
        w = [0, 1]
        add_term(u, v, w)
        
        return 4

    def schoolbook_2x1_2():
        # [2,1,2]
        # A is 2x1, B is 1x2, C is 2x2.
        # A indices: 0 (row 0), 1 (row 1).
        # B indices: 0 (col 0), 1 (col 1).
        # C indices: 00, 01, 10, 11.
        
        # C = A * B
        # C00 = A0 B0
        # C01 = A0 B1
        # C10 = A1 B0
        # C11 = A1 B1
        
        # Term 1: A0 B0
        u = [1, 0]
        v = [1, 0]
        w = [1, 0, 0, 0]
        add_term(u, v, w)
        
        # Term 2: A0 B1
        u = [1, 0]
        v = [0, 1]
        w = [0, 1, 0, 0]
        add_term(u, v, w)
        
        # Term 3: A1 B0
        u = [0, 1]
        v = [1, 0]
        w = [0, 0, 1, 0]
        add_term(u, v, w)
        
        # Term 4: A1 B1
        u = [0, 1]
        v = [0, 1]
        w = [0, 0, 0, 1]
        add_term(u, v, w)
        
        return 4

    def schoolbook_2x1_1():
        # [2,1,1]
        # A 2x1, B 1x1, C 2x1.
        # C0 = A0 B0
        # C1 = A1 B0
        
        # Term 1
        u = [1, 0]
        v = [1]
        w = [1, 0]
        add_term(u, v, w)
        
        # Term 2
        u = [0, 1]
        v = [1]
        w = [0, 1]
        add_term(u, v, w)
        
        return 2

    def schoolbook_1x2_2():
        # [1,2,2]
        # A 1x2, B 2x2, C 1x2.
        # A indices: 0, 1.
        # B indices: 00, 01, 10, 11.
        # C indices: 0, 1.
        
        # C0 = A0 B00 + A1 B10
        # C1 = A0 B01 + A1 B11
        
        # Term 1: A0 B00
        u = [1, 0]
        v = [1, 0, 0, 0]
        w = [1, 0]
        add_term(u, v, w)
        
        # Term 2: A1 B10
        u = [0, 1]
        v = [0, 0, 1, 0]
        w = [1, 0]
        add_term(u, v, w)
        
        # Term 3: A0 B01
        u = [1, 0]
        v = [0, 1, 0, 0]
        w = [0, 1]
        add_term(u, v, w)
        
        # Term 4: A1 B11
        u = [0, 1]
        v = [0, 0, 0, 1]
        w = [0, 1]
        add_term(u, v, w)
        
        return 4

    def schoolbook_1x2_1():
        # [1,2,1]
        # A 1x2, B 2x1, C 1x1.
        # C0 = A0 B0 + A1 B1
        
        # Term 1
        u = [1, 0]
        v = [1, 0]
        w = [1]
        add_term(u, v, w)
        
        # Term 2
        u = [0, 1]
        v = [0, 1]
        w = [1]
        add_term(u, v, w)
        
        return 2

    def schoolbook_1x1_2():
        # [1,1,2]
        # A 1x1, B 1x2, C 1x2.
        # C0 = A0 B0
        # C1 = A0 B1
        
        # Term 1
        u = [1]
        v = [1, 0]
        w = [1, 0]
        add_term(u, v, w)
        
        # Term 2
        u = [1]
        v = [0, 1]
        w = [0, 1]
        add_term(u, v, w)
        
        return 2

    def schoolbook_1x1_1():
        # [1,1,1]
        # Term 1
        u = [1]
        v = [1]
        w = [1]
        add_term(u, v, w)
        return 1

    # Global Index Mapping Functions
    # A (3x3) row-major index: i*3 + j
    # B (3x6) row-major index: i*6 + j
    # C (3x6) row-major index: i*6 + j
    
    def map_A(i, j, a_i, a_j, u):
        # u is local 2x2 or similar.
        # a_i, a_j are the global starting row/col of this block in A.
        # This is getting complicated to do with generic mapping.
        # Let's just explicitly construct the terms for the 51-rank case.
        
        pass

    # Given the complexity of generic mapping, I will explicitly construct the 51 terms.
    # But wait, I can write a small helper to expand a local term to global.
    
    def expand_term(local_u, local_v, local_w, a_offset, b_offset, c_offset, a_shape, b_shape, c_shape, n, m, p):
        # a_offset: (row_start, col_start) in A
        # b_offset: (row_start, col_start) in B
        # c_offset: (row_start, col_start) in C
        # a_shape: (rows, cols) of local A
        # b_shape: (rows, cols) of local B
        # c_shape: (rows, cols) of local C
        
        # n, m, p are global dims.
        # A global dim: n x m
        # B global dim: m x p
        # C global dim: p x n  <-- WAIT.
        
        # The task says:
        # U: (n*m, rank)
        # V: (m*p, rank)
        # W: (p*n, rank)
        # W[k*n+i, r]
        
        # So C is p x n?
        # A is n x m. B is m x p.
        # Product C = A B is n x p.
        # But W is p x n?
        # "Output factor follows the official trace convention W[k*n+i,r]"
        # Usually C[i,k] is stored.
        # If W index is k*n+i, then row k, col i?
        # If C is n x p, then i is row (0..n-1), k is col (0..p-1).
        # Index k*n + i means row k, col i.
        # So the matrix stored in W is transposed? Or the indices are swapped?
        # If C[i,k] is the standard (row i, col k).
        # W[k*n+i] suggests that the first index in W corresponds to k (col of C) and second to i (row of C).
        # So W is of size p x n.
        
        # Let's assume:
        # Global A index: i*m + j (i=0..n-1, j=0..m-1)
        # Global B index: j*p + k (j=0..m-1, k=0..p-1)
        # Global W index: k*n + i (k=0..p-1, i=0..n-1)
        
        # Local indices:
        # local_u corresponds to A block.
        # local_v corresponds to B block.
        # local_w corresponds to C block (transposed storage).
        
        # I will create the global U, V, W matrices directly.
        
        pass

    # This approach is too prone to error. I will hardcode the 51 terms by generating them programmatically but carefully mapping indices.
    
    # Let's restart the construction with explicit index mapping.
    
    U = [0] * 9
    V = [0] * 18
    W = [0] * 18
    # Wait, rank is 51.
    # U should be 9x51.
    # V should be 18x51.
    # W should be 18x51.
    
    rank = 51
    U = [[0] * rank for _ in range(9)]
    V = [[0] * rank for _ in range(18)]
    W = [[0] * rank for _ in range(18)]
    
    col = 0
    
    def add_global_term(gu, gv, gw):
        nonlocal col
        for i in range(len(gu)):
            if gu[i] != 0:
                U[i][col] = gu[i]
        for i in range(len(gv)):
            if gv[i] != 0:
                V[i][col] = gv[i]
        for i in range(len(gw)):
            if gw[i] != 0:
                W[i][col] = gw[i]
        col += 1
        
    # We need to map local terms to global terms.
    # Local A (2x2) at global (a_r, a_c)
    # Local B (2x2) at global (b_r, b_c)
    # Local C (2x2) at global (c_r, c_c)
    
    # But the blocks are not all 2x2.
    # T1: A00 (2x2), B00 (2x6), C00 (2x6).
    # We split B00 and C00 into three 2x2 blocks.
    
    # Block 1: T1_part0: A00 (2x2), B00_c01 (2x2), C00_c01 (2x2)
    # A00 is at A(0,0).
    # B00_c01 is at B(0,0).
    # C00_c01 is at C(0,0).
    
    # Local indices for 2x2:
    # A: 00, 01, 10, 11
    # B: 00, 01, 10, 11
    # C: 00, 01, 10, 11 (stored as C^T? No, W index k*n+i.
    # If C block is 2x2, k is 0,1 and i is 0,1.
    # W indices: 0*3+0=0, 0*3+1=1, 1*3+0=3, 1*3+1=4.
    # So W for this block uses global indices 0,1,3,4.
    
    # Let's write a helper to map a local Strassen term to global.
    
    def map_strassen_term(local_u, local_v, local_w, a_r, a_c, b_r, b_c, c_r, c_c):
        # local_u: length 4
        # local_v: length 4
        # local_w: length 4
        
        gu = [0] * 9
        gv = [0] * 18
        gw = [0] * 18
        
        # A mapping
        # local 0 -> global a_r, a_c
        # local 1 -> global a_r, a_c+1
        # local 2 -> global a_r+1, a_c
        # local 3 -> global a_r+1, a_c+1
        
        for l_i in range(2):
            for l_j in range(2):
                idx = l_i * 2 + l_j
                g_i = a_r + l_i
                g_j = a_c + l_j
                gu[g_i * 3 + g_j] = local_u[idx]
                
        # B mapping
        # local 0 -> global b_r, b_c
        # local 1 -> global b_r, b_c+1
        # local 2 -> global b_r+1, b_c
        # local 3 -> global b_r+1, b_c+1
        
        for l_i in range(2):
            for l_j in range(2):
                idx = l_i * 2 + l_j
                g_i = b_r + l_i
                g_j = b_c + l_j
                gv[g_i * 6 + g_j] = local_v[idx]
                
        # C mapping
        # local C is 2x2.
        # C[i,j] where i is row (0,1), j is col (0,1).
        # Global C row c_r + i, col c_c + j.
        # W index: col * 3 + row.
        # Global Col = c_c + j
        # Global Row = c_r + i
        # W index = (c_c + j) * 3 + (c_r + i)
        
        for l_i in range(2):
            for l_j in range(2):
                idx = l_i * 2 + l_j
                g_i = c_r + l_i
                g_j = c_c + l_j
                gw[g_j * 3 + g_i] = local_w[idx]
                
        add_global_term(gu, gv, gw)

    # Generate Strassen terms
    s_terms = []
    # M1
    s_terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
    # M2
    s_terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]))
    # M3
    s_terms.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]))
    # M4
    s_terms.append(([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]))
    # M5
    s_terms.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]))
    # M6
    s_terms.append(([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]))
    # M7
    s_terms.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]))
    
    # T1: A00 (0,0), B00 (0,0), C00 (0,0)
    # Split B/C cols into 0-1, 2-3, 4-5.
    for c_offset in [0, 2, 4]:
        for lu, lv, lw in s_terms:
            map_strassen_term(lu, lv, lw, 0, 0, 0, c_offset, 0, c_offset)
            
    # T2: A01 (0,2), B10 (2,0), C00 (0,0)
    # A01 is 2x1. B10 is 1x6. C00 is 2x6.
    # Split B/C cols into 0-1, 2-3, 4-5.
    # Sub-problem: [2,1,2]
    
    # [2,1,2] Schoolbook terms:
    # 1: A0 B0 -> C00
    # 2: A0 B1 -> C01
    # 3: A1 B0 -> C10
    # 4: A1 B1 -> C11
    
    # Local A: 2x1 (indices 0,1)
    # Local B: 1x2 (indices 0,1)
    # Local C: 2x2 (indices 00,01,10,11)
    
    def map_212_term(local_u, local_v, local_w, a_r, a_c, b_r, b_c, c_r, c_c):
        gu = [0] * 9
        gv = [0] * 18
        gw = [0] * 18
        
        # A
        for l_i in range(2):
            g_i = a_r + l_i
            g_j = a_c
            gu[g_i * 3 + g_j] = local_u[l_i]
            
        # B
        for l_j in range(2):
            g_i = b_r
            g_j = b_c + l_j
            gv[g_i * 6 + g_j] = local_v[l_j]
            
        # C
        for l_i in range(2):
            for l_j in range(2):
                idx = l_i * 2 + l_j
                g_i = c_r + l_i
                g_j = c_c + l_j
                gw[g_j * 3 + g_i] = local_w[idx]
                
        add_global_term(gu, gv, gw)

    # T2 terms
    t2_terms = [
        ([1, 0], [1, 0], [1, 0, 0, 0]),
        ([1, 0], [0, 1], [0, 1, 0, 0]),
        ([0, 1], [1, 0], [0, 0, 1, 0]),
        ([0, 1], [0, 1], [0, 0, 0, 1])
    ]
    
    for c_offset in [0, 2, 4]:
        for lu, lv, lw in t2_terms:
            map_212_term(lu, lv, lw, 0, 2, 2, c_offset, 0, c_offset)
            
    # T3: A10 (2,0), B00 (0,0), C10 (2,0)
    # A10 is 1x2. B00 is 2x6. C10 is 1x6.
    # Split B/C cols into 0-1, 2-3, 4-5.
    # Sub-problem: [1,2,2]
    
    def map_122_term(local_u, local_v, local_w, a_r, a_c, b_r, b_c, c_r, c_c):
        gu = [0] * 9
        gv = [0] * 18
        gw = [0] * 18
        
        # A
        for l_j in range(2):
            g_i = a_r
            g_j = a_c + l_j
            gu[g_i * 3 + g_j] = local_u[l_j]
            
        # B
        for l_i in range(2):
            for l_j in range(2):
                idx = l_i * 2 + l_j
                g_i = b_r + l_i
                g_j = b_c + l_j
                gv[g_i * 6 + g_j] = local_v[idx]
                
        # C
        for l_j in range(2):
            g_i = c_r
            g_j = c_c + l_j
            gw[g_j * 3 + g_i] = local_w[l_j]
            
        add_global_term(gu, gv, gw)

    # T3 terms
    t3_terms = [
        ([1, 0], [1, 0, 0, 0], [1, 0]),
        ([0, 1], [0, 0, 1, 0], [1, 0]),
        ([1, 0], [0, 1, 0, 0], [0, 1]),
        ([0, 1], [0, 0, 0, 1], [0, 1])
    ]
    
    for c_offset in [0, 2, 4]:
        for lu, lv, lw in t3_terms:
            map_122_term(lu, lv, lw, 2, 0, 0, c_offset, 2, c_offset)
            
    # T4: A11 (2,2), B10 (2,0), C10 (2,0)
    # A11 is 1x1. B10 is 1x6. C10 is 1x6.
    # Split B/C cols into 0-1, 2-3, 4-5.
    # Sub-problem: [1,1,2]
    
    def map_112_term(local_u, local_v, local_w, a_r, a_c, b_r, b_c, c_r, c_c):
        gu = [0] * 9
        gv = [0] * 18
        gw = [0] * 18
        
        # A
        gu[a_r * 3 + a_c] = local_u[0]
        
        # B
        for l_j in range(2):
            g_i = b_r
            g_j = b_c + l_j
            gv[g_i * 6 + g_j] = local_v[l_j]
            
        # C
        for l_j in range(2):
            g_i = c_r
            g_j = c_c + l_j
            gw[g_j * 3 + g_i] = local_w[l_j]
            
        add_global_term(gu, gv, gw)

    # T4 terms
    t4_terms = [
        ([1], [1, 0], [1, 0]),
        ([1], [0, 1], [0, 1])
    ]
    
    for c_offset in [0, 2, 4]:
        for lu, lv, lw in t4_terms:
            map_112_term(lu, lv, lw, 2, 2, 2, c_offset, 2, c_offset)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
