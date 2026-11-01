"""FTW-inspired decomposition for 2x4x6 matrix multiplication."""
DIMENSIONS = [2, 4, 6]

def _frac(x):
    """Represent half-integer as float."""
    return float(x)

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 6
    # We will use a rank-42 decomposition based on Kronecker-like tensor sum
    # of Strassen-based blocks.
    
    # Rank 42 = 3 * 14
    # 14 = rank of 2x4x2 multiplication (sum of two 2x2x2 Strassen blocks)
    # Actually, let's use a cleaner construction.
    
    # Block partition: split p=6 into three blocks of size 2
    # So C = A B where A is 2x4, B is 4x6 = [B1, B2, B3] each 4x2
    # C = [C1, C2, C3] where Ci = A * Bi
    # Each Ci is 2x2, so rank of each 2x2 multiplication can be 7 (Strassen)
    # Total rank = 3 * 7 = 21? No, that's not right because A is shared.
    
    # Actually, if we compute C = A * B directly:
    # Let's use the standard block tensor product approach.
    
    # For n=2, m=4, p=6:
    # We can view this as a sum of 4 blocks (from m=4 split into 2x2)
    # Each block is (2, 2, 6) -> but that's not efficient.
    
    # Better: use Strassen on 2x2x2 cores.
    # m=4 = 2*2, so A is 2x4 = [A1, A2] where A1, A2 are 2x2
    # p=6, so B is 4x6. Split B into 3 blocks of 4x2: B = [B1, B2, B3]
    # C = A*B = [A1*B1 + A2*B1, A1*B2 + A2*B2, A1*B3 + A2*B3]
    # Each column block of C is 2x2.
    
    # For each column block r (r=0,1,2), we need to compute Cr = A1*Br + A2*Br where Br is 4x2
    # Wait, A is 2x4 = row-wise [A1, A2] where A1 is 2x2 (cols 0,1), A2 is 2x2 (cols 2,3)
    # Br is 4x2 = [Br_top; Br_bot] where Br_top is 2x2, Br_bot is 2x2
    # Cr = A1 * Br_top + A2 * Br_bot (both 2x2 matrices)
    
    # So for each of the 3 column blocks, we compute a sum of two 2x2x2 tensor products.
    # The tensor for computing X*Y + X'*Y' where X,X' are 2x2 and Y,Y' are 2x2
    # is the sum of two 2x2x2 tensors, each of rank 7, total rank 14.
    # For 3 such blocks: total rank = 3 * 14 = 42.
    
    # However, we can be smarter: the same A blocks are reused.
    # But since B blocks are different, we still need separate computations.
    
    # Let's construct the rank-42 decomposition explicitly.
    # 
    # Strassen for 2x2x2: 
    # Given 2x2 matrices X, Y, compute Z = X*Y
    # Strassen uses 7 multiplications with integer coefficients.
    
    # Standard Strassen:
    # M1 = (X00 + X11)(Y00 + Y11)
    # M2 = (X10 + X11)Y00
    # M3 = X00(Y01 - Y11)
    # M4 = X11(Y10 - Y00)
    # M5 = (X00 + X01)Y11
    # M6 = (X10 - X00)(Y00 + Y01)
    # M7 = (X01 - X11)(Y10 + Y11)
    #
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # For our problem, for column block r:
    # We need Zr = A1*Br_top + A2*Br_bot
    # This is a sum of two tensor products.
    # We can decompose this as:
    # Zr = A1*Br_top + A2*Br_bot
    # 
    # Let's define the combined tensor.
    # The "input" is (A1, A2, Br_top, Br_bot) and output is Zr.
    # We can use 14 scalar multiplications:
    # 7 for A1*Br_top and 7 for A2*Br_bot, then add the results.
    
    # U, V, W construction:
    # U has shape (n*m, rank) = (8, 42)
    # V has shape (m*p, rank) = (24, 42)
    # W has shape (p*n, rank) = (12, 42)
    
    rank = 42
    U = [[0.0] * rank for _ in range(8)]
    V = [[0.0] * rank for _ in range(24)]
    
    # Strassen coefficients for one 2x2x2 block
    # For X * Y = Z where X,Y,Z are 2x2
    # X indices: (i, j) for i in {0,1}, j in {0,1}
    # Y indices: (j, k) for j in {0,1}, k in {0,1}
    # Z indices: (i, k) for i in {0,1}, k in {0,1}
    
    # In our global indexing:
    # A is 2x4, index (i, a) where i in {0,1}, a in {0,1,2,3}
    # U row index = i * m + a = i*4 + a
    # 
    # B is 4x6, index (a, b) where a in {0,1,2,3}, b in {0,1,2,3,4,5}
    # V row index = a * p + b = a*6 + b
    # 
    # C is 2x6, index (i, c) where i in {0,1}, c in {0,1,2,3,4,5}
    # W row index: following trace convention W[k*n + i, r]
    # But the output is C with shape 2x6, so the tensor output index should be (i, b) for C[i,b]
    # The problem says W[k*n+i, r] following trace convention.
    # For matrix multiplication C = A*B where A is n x m, B is m x p, C is n x p
    # The tensor T[i,a,b] = C[i,b] = sum_a A[i,a] B[a,b]
    # W should have shape (p*n, rank) and W[b*n + i, r] gives coefficient for C[i,b]
    # So W row index = b * n + i
    
    # For column block r (r=0,1,2), the B columns are 2*r, 2*r+1
    # So b = 2*r + kb where kb in {0,1}
    
    # Let's enumerate all 42 columns of the decomposition.
    col = 0
    for r in range(3):
        # Block r: columns 2*r, 2*r+1 of B and C
        # Sub-blocks: A1 (cols 0,1 of A), A2 (cols 2,3 of A)
        # Br_top = rows 0,1 of B, cols 2*r, 2*r+1
        # Br_bot = rows 2,3 of B, cols 2*r, 2*r+1
        
        # Strassen for A1 * Br_top:
        # A1[i, j] where i in {0,1}, j in {0,1} -> global A index (i, j)
        # Br_top[j, kb] where j in {0,1}, kb in {0,1} -> global B index (j, 2*r + kb)
        # Output: (i, 2*r + kb)
        
        # 7 products for A1 * Br_top:
        # Product 1: (A1[0,0] + A1[1,1]) * (Br_top[0,0] + Br_top[1,1])
        # Product 2: (A1[1,0] + A1[1,1]) * Br_top[0,0]
        # Product 3: A1[0,0] * (Br_top[0,1] - Br_top[1,1])
        # Product 4: A1[1,1] * (Br_top[1,0] - Br_top[0,0])
        # Product 5: (A1[0,0] + A1[0,1]) * Br_top[1,1]
        # Product 6: (A1[1,0] - A1[0,0]) * (Br_top[0,0] + Br_top[0,1])
        # Product 7: (A1[0,1] - A1[1,1]) * (Br_top[1,0] + Br_top[1,1])
        
        # A1 index in A: (i, j) -> row = i*4 + j
        # Br_top index in B: (j, 2*r + kb) -> row = j*6 + (2*r + kb)
        # Output (i, 2*r + kb) -> W row = (2*r + kb)*2 + i
        
        strassen_a1 = [
            {  # Product 1
                'a': [(0,0,1), (1,1,1)],  # (i, j, coeff)
                'b': [(0,0,1), (1,1,1)],  # (j, kb, coeff)
            },
            {  # Product 2
                'a': [(1,0,1), (1,1,1)],
                'b': [(0,0,1)],
            },
            {  # Product 3
                'a': [(0,0,1)],
                'b': [(0,1,1), (1,1,-1)],
            },
            {  # Product 4
                'a': [(1,1,1)],
                'b': [(1,0,1), (0,0,-1)],
            },
            {  # Product 5
                'a': [(0,0,1), (0,1,1)],
                'b': [(1,1,1)],
            },
            {  # Product 6
                'a': [(1,0,1), (0,0,-1)],
                'b': [(0,0,1), (0,1,1)],
            },
            {  # Product 7
                'a': [(0,1,1), (1,1,-1)],
                'b': [(1,0,1), (1,1,1)],
            },
        ]
        
        for prod_idx in range(7):
            prod = strassen_a1[prod_idx]
            # Set U contributions
            for (i, j, coeff) in prod['a']:
                U[i*4 + j][col] += float(coeff)
            # Set V contributions
            for (j, kb, coeff) in prod['b']:
                V[j*6 + (2*r + kb)][col] += float(coeff)
            col += 1
        
        # Strassen for A2 * Br_bot:
        # A2[i, j] where i in {0,1}, j in {0,1} -> global A index (i, j+2)
        # Br_bot[j, kb] where j in {0,1}, kb in {0,1} -> global B index (j+2, 2*r + kb)
        # Output: same (i, 2*r + kb)
        
        strassen_a2 = [
            {  # Product 1
                'a': [(0,2,1), (1,3,1)],
                'b': [(2,0,1), (3,1,1)],
            },
            {  # Product 2
                'a': [(1,2,1), (1,3,1)],
                'b': [(2,0,1)],
            },
            {  # Product 3
                'a': [(0,2,1)],
                'b': [(2,1,1), (3,1,-1)],
            },
            {  # Product 4
                'a': [(1,3,1)],
                'b': [(3,0,1), (2,0,-1)],
            },
            {  # Product 5
                'a': [(0,2,1), (0,3,1)],
                'b': [(3,1,1)],
            },
            {  # Product 6
                'a': [(1,2,1), (0,2,-1)],
                'b': [(2,0,1), (2,1,1)],
            },
            {  # Product 7
                'a': [(0,3,1), (1,3,-1)],
                'b': [(3,0,1), (3,1,1)],
            },
        ]
        
        for prod_idx in range(7):
            prod = strassen_a2[prod_idx]
            for (i, a_idx, coeff) in prod['a']:
                U[i*4 + a_idx][col] += float(coeff)
            for (b_idx, kb, coeff) in prod['b']:
                V[b_idx*6 + (2*r + kb)][col] += float(coeff)
            col += 1
    
    # Now set W: the output combination
    # For each column block r and each Strassen product, we need to set W.
    # W row = (b)*n + i where b is the column of C, i is the row.
    # For block r, b = 2*r + kb
    
    W = [[0.0] * rank for _ in range(12)]
    
    # Strassen output combination for one 2x2x2:
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # For block r, M1..M7 are the first 7 columns (for A1*Br_top)
    # and M8..M14 are the next 7 columns (for A2*Br_bot)
    # Total output = Zr_A1 + Zr_A2
    
    # Zr_A1:
    # (i, kb):
    # (0,0): M1 + M4 - M5 + M7
    # (0,1): M3 + M5
    # (1,0): M2 + M4
    # (1,1): M1 - M2 + M3 + M6
    
    # Zr_A2:
    # (0,0): M8 + M11 - M12 + M14
    # (0,1): M10 + M12
    # (1,0): M9 + M11
    # (1,1): M8 - M9 + M10 + M13
    
    # Combined:
    # (0,0): M1 + M4 - M5 + M7 + M8 + M11 - M12 + M14
    # (0,1): M3 + M5 + M10 + M12
    # (1,0): M2 + M4 + M9 + M11
    # (1,1): M1 - M2 + M3 + M6 + M8 - M9 + M10 + M13
    
    for r in range(3):
        base_col = r * 14
        m_cols = list(range(base_col, base_col + 14))
        
        # Output (i, kb) -> W row = (2*r + kb)*2 + i
        # i=0, kb=0: row = 2*r*2 + 0 = 4r
        # i=0, kb=1: row = 2*r*2 + 1 + 1 = 4r+1? No: (2*r+1)*2 + 0 = 4r+2
        # i=1, kb=0: row = (2*r)*2 + 1 = 4r+1
        # i=1, kb=1: row = (2*r+1)*2 + 1 = 4r+3
        
        # (i=0, kb=0): row = (2*r+0)*2 + 0 = 4r
        W[4*r][m_cols[0]] += 1    # M1
        W[4*r][m_cols[3]] += 1    # M4
        W[4*r][m_cols[4]] += -1   # M5
        W[4*r][m_cols[6]] += 1    # M7
        W[4*r][m_cols[7]] += 1    # M8
        W[4*r][m_cols[10]] += 1   # M11
        W[4*r][m_cols[11]] += -1  # M12
        W[4*r][m_cols[13]] += 1   # M14
        
        # (i=0, kb=1): row = (2*r+1)*2 + 0 = 4r+2
        W[4*r+2][m_cols[2]] += 1   # M3
        W[4*r+2][m_cols[4]] += 1   # M5
        W[4*r+2][m_cols[9]] += 1   # M10
        W[4*r+2][m_cols[11]] += 1  # M12
        
        # (i=1, kb=0): row = (2*r+0)*2 + 1 = 4r+1
        W[4*r+1][m_cols[1]] += 1   # M2
        W[4*r+1][m_cols[3]] += 1   # M4
        W[4*r+1][m_cols[8]] += 1   # M9
        W[4*r+1][m_cols[10]] += 1  # M11
        
        # (i=1, kb=1): row = (2*r+1)*2 + 1 = 4r+3
        W[4*r+3][m_cols[0]] += 1   # M1
        W[4*r+3][m_cols[1]] += -1  # M2
        W[4*r+3][m_cols[2]] += 1   # M3
        W[4*r+3][m_cols[5]] += 1   # M6
        W[4*r+3][m_cols[7]] += 1   # M8
        W[4*r+3][m_cols[8]] += -1  # M9
        W[4*r+3][m_cols[9]] += 1   # M10
        W[4*r+3][m_cols[12]] += 1  # M13
    
    # Verify dimensions
    assert len(U) == 8 and len(U[0]) == 42
    assert len(V) == 24 and len(V[0]) == 42
    assert len(W) == 12 and len(W[0]) == 42
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
