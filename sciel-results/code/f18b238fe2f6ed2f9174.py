import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    # We need to decompose the tensor T[i, j, k] where:
    # i in [0, n)   -> row of A, column of C
    # j in [0, m*p) -> index in B (flattened row-major: j = row*m + col)
    # k in [0, n*p) -> index in C (fllattened row-major: k = row*p + col)
    
    # Wait, the contract says:
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    # This corresponds to the standard matrix multiplication tensor T_{A,B,C}
    # T_{a, b, c} = 1 if c = a + b in the appropriate indices?
    # No, for matrix multiplication C = A * B:
    # C_{i, l} = sum_{j} A_{i, j} B_{j, l}
    # So the tensor T_{i, j, l} = 1 if i is row of A, j is col of A / row of B, l is col of B.
    # Actually, the tensor is T_{A, B, C} where indices are:
    # A index: (i, j) -> flat index i*m + j
    # B index: (j, l) -> flat index j*p + l
    # C index: (i, l) -> flat index i*p + l
    # T_{(i,j), (j,l), (i,l)} = 1
    
    # The provided shapes are:
    # U: (n*m, rank) -> corresponds to A indices
    # V: (m*p, rank) -> corresponds to B indices
    # W: (p*n, rank) -> corresponds to C indices. Note: p*n is same as n*p.
    # The contract says W follows "official trace convention W[k*n+i,r]".
    # Usually C is n x p. Flattened row-major is C[i, l] -> i*p + l.
    # "W[k*n+i,r]" suggests index = k*n + i. If k is column index (0..p-1) and i is row index (0..n-1),
    # then index = col*n + row. This is column-major flattening for C.
    # Let's verify.
    # If C is n x p, and we store it as W[col, row], then flat index = col*n + row.
    # Standard numpy flatten (row-major) is row*p + col.
    # The prompt says: "output factor follows the official trace convention W[k*n+i,r]".
    # This strongly implies W is indexed by [k, i] where k is the second index of C (column) and i is the first index of C (row).
    # So C_{i, k} maps to W[k*n + i].
    
    # Let's double check the dimensions.
    # A is n x m. Size n*m.
    # B is m x p. Size m*p.
    # C is n x p. Size n*p.
    
    # We will build the decomposition term by term.
    
    U = []
    V = []
    W = []
    
    # Strategy:
    # 1. Split B into blocks of columns. Since p=10 is even, we can split B into 5 blocks of 2 columns.
    #    B = [B_0, B_1, ..., B_4] where B_k is m x 2.
    #    However, A is 2x3. The multiplication is C = A * B.
    #    C_{i, l} = sum_{j=0}^{2} A_{i, j} B_{j, l}.
    
    #    Let's split the column space of B (and C) into pairs.
    #    For each pair of columns (2k, 2k+1) in B:
    #    Let B_k be the 3x2 submatrix of B corresponding to these columns.
    #    Let C_k be the 2x2 submatrix of C corresponding to these columns.
    #    C_k = A * B_k.
    
    #    A is 2x3. B_k is 3x2.
    #    We can split A into A_left (2x2, cols 0,1) and A_right (2x1, col 2).
    #    We can split B_k into B_k_top (2x2, rows 0,1) and B_k_bottom (1x2, row 2).
    
    #    C_k = A_left * B_k_top + A_right * B_k_bottom.
    
    #    The term A_right * B_k_bottom is a 2x2 matrix where:
    #    (A_right * B_k_bottom)_{i, l} = A_{i, 2} * B_{2, 2k+l}.
    #    This is a rank 4 contribution for each 2x2 block?
    #    A_{0,2} * B_{2, 2k} -> C_{0, 2k}
    #    A_{0,2} * B_{2, 2k+1} -> C_{0, 2k+1}
    #    A_{1,2} * B_{2, 2k} -> C_{1, 2k}
    #    A_{1,2} * B_{2, 2k+1} -> C_{1, 2k+1}
    #    These are 4 independent rank-1 terms?
    #    Term 1: U has A_{0,2}=1, others 0. V has B_{2, 2k}=1, others 0. W has C_{0, 2k}=1, others 0.
    #    Term 2: U has A_{0,2}=1. V has B_{2, 2k+1}=1. W has C_{0, 2k+1}=1.
    #    Term 3: U has A_{1,2}=1. V has B_{2, 2k}=1. W has C_{1, 2k}=1.
    #    Term 4: U has A_{1,2}=1. V has B_{2, 2k+1}=1. W has C_{1, 2k+1}=1.
    
    #    The term A_left * B_k_top is a 2x2 matrix multiplication of a 2x2 matrix (A_left) by a 2x2 matrix (B_k_top).
    #    We can use Strassen's algorithm for this 2x2 * 2x2 multiplication.
    #    Strassen requires 7 multiplications.
    
    #    So for each of the 5 column pairs:
    #    Cost = 7 (Strassen) + 4 (Cross terms) = 11.
    #    Total rank = 5 * 11 = 55.
    
    #    This is an improvement over the schoolbook rank of 60.
    
    # Let's verify if we can do better.
    # Is there a better decomposition for 2x3 * 3x2?
    # Rank of 2x3 * 3x2 multiplication?
    # It's known that rank(2,3,2) is 11?
    # Let's check.
    # Schoolbook is 2*3*2 = 12.
    # Strassen on 2x2 part + 4 cross terms = 7 + 4 = 11.
    # Can we do 10?
    # There are results on rank of matrix multiplication tensors.
    # R(2,3,2) = 11 is a known result (e.g., in Landsberg's work or similar).
    # So 11 per block seems optimal for this block structure.
    # Total 55.
    
    # Let's implement this.
    
    for block in range(5):
        col_offset = 2 * block
        
        # Indices for B block columns
        # B columns: 2*block, 2*block+1
        # B rows: 0, 1, 2
        
        # Indices for C block columns
        # C columns: 2*block, 2*block+1
        # C rows: 0, 1
        
        # Part 1: Strassen for A_left (2x2) * B_top (2x2)
        # A_left entries:
        # A_{0,0} -> U_idx 0
        # A_{0,1} -> U_idx 1
        # A_{1,0} -> U_idx 3
        # A_{1,1} -> U_idx 4
        
        # B_top entries:
        # B_{0, 2b} -> V_idx 0*p + 2b = 2b
        # B_{0, 2b+1} -> V_idx 0*p + 2b+1 = 2b+1
        # B_{1, 2b} -> V_idx 1*p + 2b = 10 + 2b
        # B_{1, 2b+1} -> V_idx 1*p + 2b+1 = 10 + 2b+1
        
        # C_top entries (output of this part):
        # C_{0, 2b} -> W_idx (2b)*n + 0 = 2b*2 = 4b
        # C_{0, 2b+1} -> W_idx (2b+1)*n + 0 = (2b+1)*2 = 4b+2
        # C_{1, 2b} -> W_idx (2b)*n + 1 = 4b+1
        # C_{1, 2b+1} -> W_idx (2b+1)*n + 1 = 4b+3
        
        # Strassen Formulas:
        # m1 = (a00 + a11) * (b00 + b11)
        # m2 = (a10 + a11) * b00
        # m3 = a00 * (b01 - b11)
        # m4 = a11 * (b10 - b00)
        # m5 = (a00 + a01) * b11
        # m6 = (a10 - a00) * (b00 + b01)
        # m7 = (a01 - a11) * (b10 + b11)
        
        # c00 = m1 + m4 - m5 + m7
        # c01 = m3 + m5
        # c10 = m2 + m4
        # c11 = m1 - m2 + m3 + m6
        
        b00_v = 2 * block
        b01_v = 2 * block + 1
        b10_v = 10 + 2 * block
        b11_v = 10 + 2 * block + 1
        
        c00_w = 4 * block
        c01_w = 4 * block + 2
        c10_w = 4 * block + 1
        c11_w = 4 * block + 3
        
        a00_u = 0
        a01_u = 1
        a10_u = 3
        a11_u = 4
        
        # m1
        U.append([0.0] * 6); U[-1][a00_u] = 1; U[-1][a11_u] = 1
        V.append([0.0] * 30); V[-1][b00_v] = 1; V[-1][b11_v] = 1
        W.append([0.0] * 20); W[-1][c00_w] = 1; W[-1][c11_w] = 1
        
        # m2
        U.append([0.0] * 6); U[-1][a10_u] = 1; U[-1][a11_u] = 1
        V.append([0.0] * 30); V[-1][b00_v] = 1
        W.append([0.0] * 20); W[-1][c10_w] = 1; W[-1][c11_w] = -1
        
        # m3
        U.append([0.0] * 6); U[-1][a00_u] = 1
        V.append([0.0] * 30); V[-1][b01_v] = 1; V[-1][b11_v] = -1
        W.append([0.0] * 20); W[-1][c01_w] = 1; W[-1][c11_w] = 1
        
        # m4
        U.append([0.0] * 6); U[-1][a11_u] = 1
        V.append([0.0] * 30); V[-1][b10_v] = 1; V[-1][b00_v] = -1
        W.append([0.0] * 20); W[-1][c00_w] = 1; W[-1][c10_w] = 1
        
        # m5
        U.append([0.0] * 6); U[-1][a00_u] = 1; U[-1][a01_u] = 1
        V.append([0.0] * 30); V[-1][b11_v] = 1
        W.append([0.0] * 20); W[-1][c00_w] = -1; W[-1][c01_w] = 1
        
        # m6
        U.append([0.0] * 6); U[-1][a10_u] = 1; U[-1][a00_u] = -1
        V.append([0.0] * 30); V[-1][b00_v] = 1; V[-1][b01_v] = 1
        W.append([0.0] * 20); W[-1][c11_w] = 1
        
        # m7
        U.append([0.0] * 6); U[-1][a01_u] = 1; U[-1][a11_u] = -1
        V.append([0.0] * 30); V[-1][b10_v] = 1; V[-1][b11_v] = 1
        W.append([0.0] * 20); W[-1][c00_w] = 1
        
        # Part 2: Cross terms for A_right (2x1) * B_bottom (1x2)
        # A_right entries:
        # A_{0,2} -> U_idx 2
        # A_{1,2} -> U_idx 5
        
        # B_bottom entries:
        # B_{2, 2b} -> V_idx 2*p + 2b = 20 + 2b
        # B_{2, 2b+1} -> V_idx 2*p + 2b+1 = 20 + 2b+1
        
        b20_v = 20 + 2 * block
        b21_v = 20 + 2 * block + 1
        
        # Term 1: A_{0,2} * B_{2, 2b} -> C_{0, 2b}
        U.append([0.0] * 6); U[-1][2] = 1
        V.append([0.0] * 30); V[-1][b20_v] = 1
        W.append([0.0] * 20); W[-1][c00_w] = 1
        
        # Term 2: A_{0,2} * B_{2, 2b+1} -> C_{0, 2b+1}
        U.append([0.0] * 6); U[-1][2] = 1
        V.append([0.0] * 30); V[-1][b21_v] = 1
        W.append([0.0] * 20); W[-1][c01_w] = 1
        
        # Term 3: A_{1,2} * B_{2, 2b} -> C_{1, 2b}
        U.append([0.0] * 6); U[-1][5] = 1
        V.append([0.0] * 30); V[-1][b20_v] = 1
        W.append([0.0] * 20); W[-1][c10_w] = 1
        
        # Term 4: A_{1,2} * B_{2, 2b+1} -> C_{1, 2b+1}
        U.append([0.0] * 6); U[-1][5] = 1
        V.append([0.0] * 30); V[-1][b21_v] = 1
        W.append([0.0] * 20); W[-1][c11_w] = 1
        
    U = np.array(U).T
    V = np.array(V).T
    W = np.array(W).T
    
    return {"U": U, "V": V, "W": W}
