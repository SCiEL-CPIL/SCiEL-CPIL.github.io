import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # Rank is 28
    R = 28
    
    # Initialize matrices
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    # Local Strassen matrices (4x7)
    # U_loc[i, r], V_loc[j, r], W_loc[k, r]
    # Indices 0..3 for 2x2 blocks
    U_loc = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V_loc = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W_loc = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    # 4 products
    # P1: A1 B11 -> C11
    # P2: A2 B21 -> C21
    # P3: A1 B12 -> C12
    # P4: A2 B22 -> C22
    
    # Mappings:
    # A index: i (0,1), j (0,1,2,3)
    # B index: k (0,1,2,3), l (0,1,2,3)
    # C index: i (0,1), l (0,1,2,3)
    
    # P1: A1 (j 0,1), B11 (k 0,1, l 0,1), C11 (l 0,1)
    # P2: A2 (j 2,3), B21 (k 2,3, l 0,1), C21 (l 0,1)
    # P3: A1 (j 0,1), B12 (k 0,1, l 2,3), C12 (l 2,3)
    # P4: A2 (j 2,3), B22 (k 2,3, l 2,3), C22 (l 2,3)
    
    products = [
        {
            'A_offset_j': 0, # j = 0 + j_loc
            'B_offset_k': 0, # k = 0 + k_loc
            'B_offset_l': 0, # l = 0 + l_loc
            'C_offset_l': 0, # l = 0 + l_loc
            'col_start': 0
        },
        {
            'A_offset_j': 2, # j = 2 + j_loc
            'B_offset_k': 2, # k = 2 + k_loc
            'B_offset_l': 0, # l = 0 + l_loc
            'C_offset_l': 0, # l = 0 + l_loc
            'col_start': 7
        },
        {
            'A_offset_j': 0, # j = 0 + j_loc
            'B_offset_k': 0, # k = 0 + k_loc
            'B_offset_l': 2, # l = 2 + l_loc
            'C_offset_l': 2, # l = 2 + l_loc
            'col_start': 14
        },
        {
            'A_offset_j': 2, # j = 2 + j_loc
            'B_offset_k': 2, # k = 2 + k_loc
            'B_offset_l': 2, # l = 2 + l_loc
            'C_offset_l': 2, # l = 2 + l_loc
            'col_start': 21
        }
    ]
    
    for prod in products:
        a_off_j = prod['A_offset_j']
        b_off_k = prod['B_offset_k']
        b_off_l = prod['B_offset_l']
        c_off_l = prod['C_offset_l']
        c_start = prod['col_start']
        
        for r_loc in range(7):
            r_glob = c_start + r_loc
            
            for i in range(2):
                for j_loc in range(2):
                    j_glob = a_off_j + j_loc
                    U[i * 4 + j_glob][r_glob] = U_loc[j_loc + i * 2][r_loc] 
                    # Wait, U_loc index is (i, j_loc).
                    # My U_loc is defined as rows 0..3.
                    # Row 0: i=0, j=0
                    # Row 1: i=0, j=1
                    # Row 2: i=1, j=0
                    # Row 3: i=1, j=1
                    # So index is i * 2 + j_loc.
                    
            for k_loc in range(2):
                for l_loc in range(2):
                    k_glob = b_off_k + k_loc
                    l_glob = b_off_l + l_loc
                    V[k_glob * 4 + l_glob][r_glob] = V_loc[k_loc * 2 + l_loc][r_loc]
                    
            for i in range(2):
                for l_loc in range(2):
                    l_glob = c_off_l + l_loc
                    W[l_glob * 2 + i][r_glob] = W_loc[l_loc + i * 2][r_loc]
                    # Wait, W index in global is (k, i) or (l, i)?
                    # Task: W shape (p*n, rank). p=4, n=2.
                    # Index is k * n + i?
                    # "W[k*n+i,r]"
                    # In my mapping, the output is C[i, l].
                    # The tensor index for W is (k, i) in the sum?
                    # Let's check the seed code: W[k * n + i][column] = 1.
                    # So W index is k * n + i.
                    # In my C mapping, the "k" of the output C[i, l] is actually l?
                    # No, C is 2x4. Indices i (row), l (col).
                    # The tensor contraction is sum_j A[i,j] B[j,k] = C[i,k].
                    # So the third index of the tensor is k (which is l in my notation).
                    # So W should be indexed by (l, i).
                    # So W index is l * n + i.
                    # l_glob is the column of C.
                    # i is the row of C.
                    # So W[l_glob * 2 + i] is correct.
                    
                    # Check W_loc index.
                    # W_loc is 4x7. Rows 0..3.
                    # Row 0: l=0, i=0
                    # Row 1: l=0, i=1
                    # Row 2: l=1, i=0
                    # Row 3: l=1, i=1
                    # So index is l_loc * 2 + i.
                    
    return {"U": U, "V": V, "W": W}
