import time
from fractions import Fraction

DEADLINE = 135.0

def _verify(U, V, W, n, m, p):
    """Verify exact decomposition: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k)"""
    rank = U.shape[1] if hasattr(U, 'shape') else len(U[0])
    
    if hasattr(U, 'shape'):
        U_arr = U
        V_arr = V
        W_arr = W
        get_U = lambda i, j: U_arr[i*m+j, :]
        get_V = lambda j, k: V_arr[j*p+k, :]
        get_W = lambda k, i: W_arr[k*n+i, :]
        dot = lambda a, b, c: sum(a[r]*b[r]*c[r] for r in range(rank))
    else:
        get_U = lambda i, j: U[i*m+j]
        get_V = lambda j, k: V[j*p+k]
        get_W = lambda k, i: W[k*n+i]
        
        def dot(a, b, c):
            s = Fraction(0)
            for r in range(rank):
                s += a[r] * b[r] * c[r]
            return s

    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = dot(get_U(i, j), get_V(j, k), get_W(k, i))
                if isinstance(val, Fraction):
                    if val != 1:
                        return False
                else:
                    if abs(val - 1.0) > 1e-9:
                        return False
    return True

def _strassen_block(U, V, W, start_col, i_off, j_off, k_off, n, m, p):
    """
    Add a 2x2x2 Strassen block.
    Indices: i in {0,1} + i_off, j in {0,1} + j_off, k in {0,1} + k_off.
    Returns number of columns used (7).
    """
    # Strassen formulas:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each Mx corresponds to a rank-1 term (or sum of two for some? No, standard Strassen is 7 rank-1 terms for 2x2x2).
    # Wait, standard Strassen has 7 multiplications.
    # U factor: row (i,j), col r.
    # V factor: row (j,k), col r.
    # W factor: row (k,i), col r.
    
    # We need to define 7 basis functions for U, V, W.
    # Let's define them explicitly.
    
    # M1: A00+A11, B00+B11 -> C00 + C11 (distributed)
    # Actually, it's easier to construct the 7 terms directly.
    
    # Term 1: M1 = (A00+A11)(B00+B11)
    # U1: i=0,j=0: 1; i=1,j=1: 1; others 0.
    # V1: j=0,k=0: 1; j=1,k=1: 1; others 0.
    # W1: k=0,i=0: 1; k=1,i=1: 1; others 0.
    # Contribution to C00: U[0,0]*V[0,0]*W[0,0] = 1*1*1 = 1.
    # Contribution to C11: U[1,1]*V[1,1]*W[1,1] = 1*1*1 = 1.
    # We need C00 = M1 + ... and C11 = M1 + ...
    # So M1 contributes to both C00 and C11.
    
    # Let's use the standard set of 7 rank-1 tensors.
    # t1: A00, B00 -> C00
    # t2: A00, B01 -> C01
    # t3: A10, B00 -> C10
    # t4: A11, B11 -> C11
    # t5: (A00+A11), (B00+B11) -> C00 + C11 (Wait, this is not rank 1 in the tensor sense if it splits to two outputs? No, the output is a matrix. The tensor is T[i,j,k].
    # A rank-1 decomposition is sum_r u_r[i,j] v_r[j,k] w_r[k,i].
    # For Strassen:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We can just define the 7 terms M1..M7 and add them with appropriate signs in W.
    # But W is shared? No, each term has its own W.
    # So we create 7 new columns.
    
    # M1: (A00+A11)(B00+B11)
    # U1: (0,0)->1, (1,1)->1. V1: (0,0)->1, (1,1)->1.
    # We want this to contribute to C00 and C11.
    # If we set W1: (0,0)->1, (1,1)->1.
    # Then C00 gets 1*1*1 = 1. C11 gets 1*1*1 = 1.
    # But we need coefficients.
    # C00 = ... + 1*M1 ...
    # C11 = ... + 1*M1 ...
    # So W1[0,0]=1, W1[1,1]=1.
    
    # M2: (A10+A11)B00
    # U2: (1,0)->1, (1,1)->1. V2: (0,0)->1.
    # C10 needs +1*M2. C11 needs -1*M2.
    # W2: (0,1)->1, (0,0)->-1? No, W index is (k,i).
    # C10 is k=0, i=1. C11 is k=1, i=1.
    # So W2[0,1] = 1, W2[1,1] = -1.
    
    # M3: A00(B01-B11)
    # U3: (0,0)->1. V3: (0,1)->1, (1,1)->-1.
    # C01 needs +1*M3. C11 needs +1*M3.
    # C01 is k=1, i=0. C11 is k=1, i=1.
    # W3: (1,0)->1, (1,1)->1.
    
    # M4: A11(B10-B00)
    # U4: (1,1)->1. V4: (1,0)->1, (0,0)->-1.
    # C10 needs +1*M4. C00 needs +1*M4.
    # C10 is k=0, i=1. C00 is k=0, i=0.
    # W4: (0,1)->1, (0,0)->1.
    
    # M5: (A00+A01)B11
    # U5: (0,0)->1, (0,1)->1. V5: (1,1)->1.
    # C01 needs +1*M5. C00 needs -1*M5.
    # C01 is k=1, i=0. C00 is k=0, i=0.
    # W5: (1,0)->1, (0,0)->-1.
    
    # M6: (A10-A00)(B00+B01)
    # U6: (1,0)->1, (0,0)->-1. V6: (0,0)->1, (0,1)->1.
    # C11 needs +1*M6.
    # C11 is k=1, i=1.
    # W6: (1,1)->1.
    
    # M7: (A01-A11)(B10+B11)
    # U7: (0,1)->1, (1,1)->-1. V7: (1,0)->1, (1,1)->1.
    # C00 needs +1*M7.
    # C00 is k=0, i=0.
    # W7: (0,0)->1.
    
    # Now we just populate the matrices.
    # We will create 7 new columns.
    
    # Helper to add a value to a cell in U, V, W
    # U row: i*m + j
    # V row: j*p + k
    # W row: k*n + i
    
    def add_U(i, j, col, val):
        U[i*m + j + i_off*m + j_off, col] += val  # Wait, indices are global.
        # Actually, U is a full matrix of size n*m x rank.
        # The "local" indices i, j are relative to the block.
        # Global i = i + i_off. Global j = j + j_off.
        U[(i + i_off)*m + (j + j_off), col] += val
        
    def add_V(j, k, col, val):
        V[(j + j_off)*p + (k + k_off), col] += val
        
    def add_W(k, i, col, val):
        W[(k + k_off)*n + (i + i_off), col] += val

    # M1
    col = start_col
    add_U(0, 0, col, 1)
    add_U(1, 1, col, 1)
    add_V(0, 0, col, 1)
    add_V(1, 1, col, 1)
    add_W(0, 0, col, 1)
    add_W(1, 1, col, 1)
    
    # M2
    col = start_col + 1
    add_U(1, 0, col, 1)
    add_U(1, 1, col, 1)
    add_V(0, 0, col, 1)
    add_W(0, 1, col, 1)
    add_W(1, 1, col, -1)
    
    # M3
    col = start_col + 2
    add_U(0, 0, col, 1)
    add_V(0, 1, col, 1)
    add_V(1, 1, col, -1)
    add_W(1, 0, col, 1)
    add_W(1, 1, col, 1)
    
    # M4
    col = start_col + 3
    add_U(1, 1, col, 1)
    add_V(1, 0, col, 1)
    add_V(0, 0, col, -1)
    add_W(0, 1, col, 1)
    add_W(0, 0, col, 1)
    
    # M5
    col = start_col + 4
    add_U(0, 0, col, 1)
    add_U(0, 1, col, 1)
    add_V(1, 1, col, 1)
    add_W(1, 0, col, 1)
    add_W(0, 0, col, -1)
    
    # M6
    col = start_col + 5
    add_U(1, 0, col, 1)
    add_U(0, 0, col, -1)
    add_V(0, 0, col, 1)
    add_V(0, 1, col, 1)
    add_W(1, 1, col, 1)
    
    # M7
    col = start_col + 6
    add_U(0, 1, col, 1)
    add_U(1, 1, col, -1)
    add_V(1, 0, col, 1)
    add_V(1, 1, col, 1)
    add_W(0, 0, col, 1)
    
    return 7

def _schoolbook_block(U, V, W, start_col, i_off, j_off, k_off, n, m, p, n_size, m_size, p_size):
    """
    Add a schoolbook block of size n_size x m_size x p_size.
    Rank = n_size * m_size * p_size.
    """
    rank = n_size * m_size * p_size
    for i in range(n_size):
        for j in range(m_size):
            for k in range(p_size):
                col = start_col + i * m_size * p_size + j * p_size + k
                U[(i + i_off)*m + (j + j_off), col] = 1
                V[(j + j_off)*p + (k + k_off), col] = 1
                W[(k + k_off)*n + (i + i_off), col] = 1
    return rank

def construct():
    n, m, p = 3, 5, 7
    
    # Strategy:
    # 1. Place 2x2x2 Strassen blocks where possible.
    # 2. Fill the rest with schoolbook.
    
    # Dimensions: 3x5x7.
    # We can fit:
    # - One 2x2x2 block at (0,0,0). Rank 7.
    # - Another 2x2x2 block?
    #   i in {0,1}, j in {0,1}, k in {0,1} used.
    #   Can we put another at i=1..2, j=2..3, k=2..3?
    #   i range: 0..2. j range: 0..4. k range: 0..6.
    
    # Let's try to pack 2x2x2 blocks.
    # Block 1: i=0,1; j=0,1; k=0,1.
    # Remaining i: 2. Remaining j: 2,3,4. Remaining k: 2,3,4,5,6.
    # Can we fit another 2x2x2?
    # Need 2 i's, 2 j's, 2 k's.
    # Only 1 i left (2). So no more 2x2x2 blocks can be placed using disjoint i-indices.
    # Unless we overlap? No, decomposition must be exact. If we overlap, we must cancel terms or add them.
    # Strassen is for disjoint sub-matrices.
    
    # So only one 2x2x2 block?
    # Rank 7.
    # Remaining:
    # i=0,1: j=2,3,4 (3 cols), k=0,1 (2 rows) -> 1*2*2 = 4 terms? No.
    # Let's just use schoolbook for the rest.
    
    # Total Rank = 7 (Strassen) + Schoolbook(Rest)
    
    # Schoolbook for rest:
    # We need to cover all (i,j,k) not in Block 1.
    # Block 1 covers: i in {0,1}, j in {0,1}, k in {0,1}.
    
    # Rest:
    # i=0, j in {2,3,4}, k in {0,1} -> 1 * 3 * 2 = 6
    # i=0, j in {0,1}, k in {2..6} -> 1 * 2 * 5 = 10
    # i=0, j in {2,3,4}, k in {2..6} -> 1 * 3 * 5 = 15
    # i=1, j in {2,3,4}, k in {0,1} -> 1 * 3 * 2 = 6
    # i=1, j in {0,1}, k in {2..6} -> 1 * 2 * 5 = 10
    # i=1, j in {2,3,4}, k in {2..6} -> 1 * 3 * 5 = 15
    # i=2, j in {0..4}, k in {0..6} -> 1 * 5 * 7 = 35
    
    # Total Schoolbook Rank = 6 + 10 + 15 + 6 + 10 + 15 + 35 = 97.
    # Total Rank = 7 + 97 = 104.
    # This is only 1 better than 105.
    
    # Can we do better?
    # Maybe use more Strassen blocks by overlapping and adjusting?
    # Or use a different partition?
    
    # Let's try to pack 2x2x2 blocks in a grid.
    # i: 3 (1 block of 2, 1 left)
    # j: 5 (2 blocks of 2, 1 left)
    # k: 7 (3 blocks of 2, 1 left)
    
    # If we place blocks at:
    # (0,0,0), (0,2,0), (0,0,2), (0,2,2) -> 4 blocks in i=0,1 plane?
    # No, i=0,1 is one block height.
    # We can place:
    # 1. i=0,1; j=0,1; k=0,1
    # 2. i=0,1; j=2,3; k=0,1
    # 3. i=0,1; j=0,1; k=2,3
    # 4. i=0,1; j=2,3; k=2,3
    
    # These 4 blocks cover i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}.
    # Rank = 4 * 7 = 28.
    
    # Remaining:
    # i in {0,1}, j=4, k in {0..6} -> 2 * 1 * 7 = 14
    # i in {0,1}, j in {0..4}, k in {4..6} -> 2 * 5 * 3 = 30
    # i=2, j in {0..4}, k in {0..6} -> 1 * 5 * 7 = 35
    
    # Total Schoolbook Rank = 14 + 30 + 35 = 79.
    # Total Rank = 28 + 79 = 107.
    # Worse than 104.
    
    # Why? Because Strassen saves rank only if the sub-blocks are "dense" enough?
    # Schoolbook for 2x2x2 is 8. Strassen is 7. Save 1.
    # But if we combine with others, the boundaries might not save much.
    
    # Let's check the 104 solution again.
    # Block 1: 2x2x2. Save 1.
    # Rest is schoolbook.
    # Total 104.
    
    # Can we save more?
    # What if we use a 2x1x1 block? Rank 2 (Schoolbook 2, Save 0).
    # 1x2x1 block? Rank 2.
    
    # It seems 104 is a good baseline.
    
    # Let's try to construct the 104 solution.
    
    total_rank = 104
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # 1. Strassen Block at (0,0,0)
    cols_used = _strassen_block(U, V, W, 0, 0, 0, 0, n, m, p)
    
    # 2. Schoolbook for rest
    # We need to iterate over all (i,j,k) and if not in Strassen block, add schoolbook term.
    # Strassen block covers i in {0,1}, j in {0,1}, k in {0,1}.
    
    next_col = cols_used
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue # Covered by Strassen
                
                U[i*m + j, next_col] = 1
                V[j*p + k, next_col] = 1
                W[k*n + i, next_col] = 1
                next_col += 1
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to pure schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}

