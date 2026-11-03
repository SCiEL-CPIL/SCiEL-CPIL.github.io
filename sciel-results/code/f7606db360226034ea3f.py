import numpy as np

def construct():
    n, m, p = 4, 4, 9
    
    U_rows = n * m # 16
    V_rows = m * p # 36
    W_rows = p * n # 36
    
    U_list = []
    V_list = []
    W_list = []
    
    # Helper to add a term
    def add_term(u_vec, v_vec, w_vec):
        U_list.append(u_vec)
        V_list.append(v_vec)
        W_list.append(w_vec)

    # --- Part 1: 4x4 * 4x4 -> 4x4 (Columns 0-3 of B, 0-3 of C) ---
    # We use Strassen recursively.
    
    # Define the 7 outer Strassen products
    # A_blocks: 0:11, 1:12, 2:21, 3:22
    # B_blocks: 0:11, 1:12, 2:21, 3:22
    
    # Outer A coefficients for each of the 7 products
    # Each is a 4x4 matrix of coefficients for A
    # A_index = i*4 + j
    outer_A_coeffs = [
        # 1: A11 + A22
        [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]], # Wait, A11 is 2x2 top-left.
        # Let's represent A as flat 16 vector.
        # A11: 0,1,4,5. A12: 2,3,6,7. A21: 8,9,12,13. A22: 10,11,14,15.
        # 1: A11 + A22 -> indices 0,1,4,5,10,11,14,15 are 1.
        # 2: A21 + A22 -> indices 8,9,12,13,10,11,14,15 are 1.
        # 3: A11 -> indices 0,1,4,5 are 1.
        # 4: A22 -> indices 10,11,14,15 are 1.
        # 5: A11 + A12 -> indices 0,1,4,5,2,3,6,7 are 1.
        # 6: A21 - A11 -> indices 8,9,12,13 are 1, 0,1,4,5 are -1.
        # 7: A12 - A22 -> indices 2,3,6,7 are 1, 10,11,14,15 are -1.
    ]
    
    # Let's create these vectors explicitly.
    def get_block_indices(block_id):
        # 0: 11, 1: 12, 2: 21, 3: 22
        # A is 4x4.
        # 11: rows 0-1, cols 0-1 -> 0,1,4,5
        # 12: rows 0-1, cols 2-3 -> 2,3,6,7
        # 21: rows 2-3, cols 0-1 -> 8,9,12,13
        # 22: rows 2-3, cols 2-3 -> 10,11,14,15
        if block_id == 0: return [0,1,4,5]
        if block_id == 1: return [2,3,6,7]
        if block_id == 2: return [8,9,12,13]
        if block_id == 3: return [10,11,14,15]
        
    outer_A_vecs = []
    # 1: 11+22
    v = [0]*16
    for idx in get_block_indices(0): v[idx]=1
    for idx in get_block_indices(3): v[idx]=1
    outer_A_vecs.append(v)
    # 2: 21+22
    v = [0]*16
    for idx in get_block_indices(2): v[idx]=1
    for idx in get_block_indices(3): v[idx]=1
    outer_A_vecs.append(v)
    # 3: 11
    v = [0]*16
    for idx in get_block_indices(0): v[idx]=1
    outer_A_vecs.append(v)
    # 4: 22
    v = [0]*16
    for idx in get_block_indices(3): v[idx]=1
    outer_A_vecs.append(v)
    # 5: 11+12
    v = [0]*16
    for idx in get_block_indices(0): v[idx]=1
    for idx in get_block_indices(1): v[idx]=1
    outer_A_vecs.append(v)
    # 6: 21-11
    v = [0]*16
    for idx in get_block_indices(2): v[idx]=1
    for idx in get_block_indices(0): v[idx]=-1
    outer_A_vecs.append(v)
    # 7: 12-22
    v = [0]*16
    for idx in get_block_indices(1): v[idx]=1
    for idx in get_block_indices(3): v[idx]=-1
    outer_A_vecs.append(v)
    
    # Outer B coefficients
    # B is 4x4. Columns 0-3.
    # B11: 0,1,4,5. B12: 2,3,6,7. B21: 8,9,12,13. B22: 10,11,14,15.
    # Note: B index is j*p + k. Here p=4 for this subproblem.
    # But in the global V, p=9.
    # For this block, B columns are 0,1,2,3.
    # So B index is j*9 + k.
    # B11: j=0,1; k=0,1 -> 0,1, 9,10.
    # B12: j=0,1; k=2,3 -> 2,3, 11,12.
    # B21: j=2,3; k=0,1 -> 18,19, 27,28.
    # B22: j=2,3; k=2,3 -> 20,21, 29,30.
    
    def get_b_block_indices_global(block_id, col_offset):
        # block_id 0:11, 1:12, 2:21, 3:22
        # rows j: 0,1 for 11,12; 2,3 for 21,22
        # cols k: 0,1 for 11,21; 2,3 for 12,22
        # global index: j*9 + (col_offset + k_local)
        idxs = []
        if block_id == 0: # 11
            for j in [0,1]:
                for k in [0,1]:
                    idxs.append(j*9 + col_offset + k)
        elif block_id == 1: # 12
            for j in [0,1]:
                for k in [2,3]:
                    idxs.append(j*9 + col_offset + k)
        elif block_id == 2: # 21
            for j in [2,3]:
                for k in [0,1]:
                    idxs.append(j*9 + col_offset + k)
        elif block_id == 3: # 22
            for j in [2,3]:
                for k in [2,3]:
                    idxs.append(j*9 + col_offset + k)
        return idxs

    outer_B_vecs = []
    col_offset = 0
    # 1: 11+22
    v = [0]*36
    for idx in get_b_block_indices_global(0, 0): v[idx]=1
    for idx in get_b_block_indices_global(3, 0): v[idx]=1
    outer_B_vecs.append(v)
    # 2: 11
    v = [0]*36
    for idx in get_b_block_indices_global(0, 0): v[idx]=1
    outer_B_vecs.append(v)
    # 3: 12-22
    v = [0]*36
    for idx in get_b_block_indices_global(1, 0): v[idx]=1
    for idx in get_b_block_indices_global(3, 0): v[idx]=-1
    outer_B_vecs.append(v)
    # 4: 21-11
    v = [0]*36
    for idx in get_b_block_indices_global(2, 0): v[idx]=1
    for idx in get_b_block_indices_global(0, 0): v[idx]=-1
    outer_B_vecs.append(v)
    # 5: 22
    v = [0]*36
    for idx in get_b_block_indices_global(3, 0): v[idx]=1
    outer_B_vecs.append(v)
    # 6: 11+12
    v = [0]*36
    for idx in get_b_block_indices_global(0, 0): v[idx]=1
    for idx in get_b_block_indices_global(1, 0): v[idx]=1
    outer_B_vecs.append(v)
    # 7: 21+22
    v = [0]*36
    for idx in get_b_block_indices_global(2, 0): v[idx]=1
    for idx in get_b_block_indices_global(3, 0): v[idx]=1
    outer_B_vecs.append(v)
    
    # Inner Strassen
    # For each outer product j (0..6), we have A^(j) and B^(j).
    # We need to decompose the 2x2 multiplication of these.
    # The 2x2 matrices A^(j) and B^(j) are formed by the linear combinations above.
    # The entries of A^(j) are linear combinations of A's entries.
    # The entries of B^(j) are linear combinations of B's entries.
    
    # Inner Strassen products for 2x2:
    # M1: (a00+a11)(b00+b10)
    # M2: (a10+a11)b00
    # M3: a00(b01-b11)
    # M4: a11(b10-b00)
    # M5: (a00+a01)b11
    # M6: (a10-a00)(b00+b01)
    # M7: (a01-a11)(b10+b11)
    
    # We need to map these to the global A and B vectors.
    # For a given outer j, A^(j) is a 2x2 matrix.
    # Its entries are:
    # A^(j)_{00} = sum_{i in block} coeff * A_{i,j}
    # This is getting complicated to trace manually.
    
    # Alternative: Just generate the 49 terms by iterating through the structure.
    
    pass
