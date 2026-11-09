"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS
    # Target rank 66
    # Structure:
    # Split p (6) into p1 (4) and p2 (2)
    # For each block, split n (3) into n1 (2) and n2 (1)
    # For each sub-block, split m (4) into m1 (2) and m2 (2)
    
    rank = 0
    U = []
    V = []
    W = []
    
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal rank
        U.append([0.0] * rank if rank > 0 else [u_val]) # This logic is flawed, need to build matrices properly
        # Better to build lists of tuples and convert to matrices at the end
        
    # Let's use a list of tuples (u_index, v_index, w_index, u_val, v_val, w_val)
    terms = []
    
    def add_term_simple(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        terms.append((u_idx, v_idx, w_idx, u_val, v_val, w_val))

    # Helper: Schoolbook decomposition for n x m x p
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
    def add_schoolbook(n_sub, m_sub, p_sub, u_offset, v_offset, w_offset, u_val=1, v_val=1, w_val=1):
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    u_idx = u_offset + i * m_sub + j
                    v_idx = v_offset + j * p_sub + k
                    w_idx = w_offset + k * n_sub + i
                    add_term_simple(u_idx, v_idx, w_idx, u_val, v_val, w_val)

    # Helper: Strassen decomposition for 2 x 2 x 2
    # Indices:
    # A: [[a00, a01], [a10, a11]] -> u_idx 0..3
    # B: [[b00, b01], [b10, b11]] -> v_idx 0..3
    # C: [[c00, c01], [c10, c11]] -> w_idx 0..3
    # W convention: W[k*n+i, r]
    # C[0,0] is w_idx 0, C[0,1] is w_idx 1, C[1,0] is w_idx 2, C[1,1] is w_idx 3
    
    def add_strassen_2x2x2(u_offset, v_offset, w_offset):
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b01-b11)
        # M4 = a11(b10-b00)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b10)
        # M7 = (a01-a11)(b11+b21) -- wait, standard Strassen:
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B01 - B11)
        # M4 = A11(B10 - B00)
        # M5 = (A00 + A01)B11
        # M6 = (A10 - A00)(B00 + B10)
        # M7 = (A01 - A11)(B11 + B21) -> (A01 - A11)(B11 + B01)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We need to map these to u, v, w coefficients.
        # Each term is a sum of products.
        # We add 7 terms.
        
        # Term 1: M1
        # u: A00 (1), A11 (1)
        # v: B00 (1), B11 (1)
        # w: C00 (1), C11 (-1)
        # This is a rank 1 tensor? No, it's a product of sums.
        # (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
        # This requires 4 rank 1 terms if expanded?
        # No, in tensor decomposition, a term is u vector x v vector x w vector.
        # u = [1, 0, 0, 1] (A00, A11)
        # v = [1, 0, 0, 1] (B00, B11)
        # w = [1, 0, 0, -1] (C00, C11)
        # This single term contributes to C00 and C11.
        
        add_term_simple(u_offset + 0, v_offset + 0, w_offset + 0, 1, 1, 1) # A00 B00 -> C00
        add_term_simple(u_offset + 3, v_offset + 3, w_offset + 0, 1, 1, 1) # A11 B11 -> C00
        add_term_simple(u_offset + 0, v_offset + 3, w_offset + 0, 1, 1, 1) # A00 B11 -> C00
        add_term_simple(u_offset + 3, v_offset + 0, w_offset + 0, 1, 1, 1) # A11 B00 -> C00
        
        # This is just schoolbook!
        # Strassen works by combining these into 7 terms where u, v, w are linear combinations.
        # We need to output the U, V, W matrices directly.
        
        # Let's just add the 7 Strassen terms.
        # Term 1: M1
        # u: [1, 0, 0, 1]
        # v: [1, 0, 0, 1]
        # w: [1, 0, 0, -1]
        # This is not a single entry in U,V,W.
        # The output format is U, V, W matrices.
        # Each column c corresponds to a term.
        # U[:, c] is the u vector.
        
        # So for M1:
        # U col: [1, 0, 0, 1]
        # V col: [1, 0, 0, 1]
        # W col: [1, 0, 0, -1]
        
        # We will accumulate these into the final U, V, W.
        # Since we are building a list of terms, we can just append the vectors.
        
        pass # Placeholder

    # I will rewrite the logic to build U, V, W directly as lists of lists.
    
    U_rows = n * m
    V_rows = m * p
    W_rows = p * n
    
    U_mat = []
    V_mat = []
    W_mat = []
    
    def add_term_vecs(u_vec, v_vec, w_vec):
        U_mat.append(u_vec)
        V_mat.append(v_vec)
        W_mat.append(w_vec)
        
    # Zero vectors
    def zero_vec(size): return [0.0] * size
    
    # Schoolbook for n_sub x m_sub x p_sub
    # Offsets: u_off, v_off, w_off
    def add_schoolbook_vecs(n_sub, m_sub, p_sub, u_off, v_off, w_off):
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    u_vec = zero_vec(U_rows)
                    v_vec = zero_vec(V_rows)
                    w_vec = zero_vec(W_rows)
                    u_vec[u_off + i * m_sub + j] = 1.0
                    v_vec[v_off + j * p_sub + k] = 1.0
                    w_vec[w_off + k * n_sub + i] = 1.0
                    add_term_vecs(u_vec, v_vec, w_vec)

    # Strassen for 2x2x2
    def add_strassen_vecs(u_off, v_off, w_off):
        # A indices: 0,1,2,3 (00,01,10,11)
        # B indices: 0,1,2,3
        # C indices: 0,1,2,3 (00,01,10,11)
        
        # M1
        u = zero_vec(U_rows); v = zero_vec(V_rows); w = zero_vec(W_rows)
        u[u_off+0] = 1; u[u_off+3] = 1
        v[v_off+0] = 1; v[v_off+3] = 1
        w[w_off+0] = 1; w[w_off+3] = -1
        add_term_vecs(u, v, w)
        
        # M2
        u = zero_vec(U_rows); v = zero_vec(V_rows); w = zero_vec(W_rows)
        u[u_off+2] = 1; u[u_off+3] = 1
        v[v_off+0] = 1
        w[w_off+2] = 1; w[w_off+3] = -1
        add_term_vecs(u, v, w)
        
        # M3
        u = zero_vec(U_rows); v = zero_vec(V_rows); w = zero_vec(W_rows)
        u[u_off+0] = 1
        v[v_off+1] = 1; v[v_off+3] = -1
        w[w_off+0] = 1; w[w_off+1] = 1
        add_term_vecs(u, v, w)
        
        # M4
        u = zero_vec(U_rows); v = zero_vec(V_rows); w = zero_vec(W_rows)
        u[u_off+3] = 1
        v[v_off+2] = 1; v[v_off+0] = -1
        w[w_off+0] = 1; w[w_off+2] = 1
        add_term_vecs(u, v, w)
        
        # M5
        u = zero_vec(U_rows); v = zero_vec(V_rows); w = zero_vec(W_rows)
        u[u_off+0] = 1; u[u_off+1] = 1
        v[v_off+3] = 1
        w[w_off+0] = -1; w[w_off+1] = 1
        add_term_vecs(u, v, w)
        
        # M6
        u = zero_vec(U_rows); v = zero_vec(V_rows); w = zero_vec(W_rows)
        u[u_off+2] = 1; u[u_off+0] = -1
        v[v_off+0] = 1; v[v_off+2] = 1
        w[w_off+3] = 1
        add_term_vecs(u, v, w)
        
        # M7
        u = zero_vec(U_rows); v = zero_vec(V_rows); w = zero_vec(W_rows)
        u[u_off+1] = 1; u[u_off+3] = -1
        v[v_off+3] = 1; v[v_off+1] = 1
        w[w_off+0] = 1; w[w_off+3] = 1
        add_term_vecs(u, v, w)

    # Decomposition of 2x2x4 using two Strassen 2x2x2
    # B is 2x4. Split into B1 (2x2) and B2 (2x2).
    # C1 = A B1, C2 = A B2.
    def add_2x2x4_strassen(u_off, v_off, w_off):
        # B1 is cols 0,1 of B. B2 is cols 2,3.
        # v_off points to start of B (2x4).
        # B1 v indices: v_off+0, v_off+1
        # B2 v indices: v_off+2, v_off+3
        # C1 w indices: w_off+0, w_off+1
        # C2 w indices: w_off+2, w_off+3
        
        # Strassen for A B1
        add_strassen_vecs(u_off, v_off, w_off) # This assumes B is 2x2. 
        # But my add_strassen_vecs uses v_off+0..3.
        # I need a variant or adjust offsets.
        # Let's just call schoolbook for 2x2x2 if Strassen is complex to offset.
        # Actually, Strassen for 2x2x2 is Rank 7. Schoolbook is 8.
        # I'll use Schoolbook for simplicity and correctness first, then optimize if time permits.
        # Schoolbook 2x2x2 Rank 8.
        # 2x2x4 Rank 16.
        # 2x4x4 Rank 32.
        # 1x4x4 Rank 16.
        # Total 32+16 = 48 for 3x4x4.
        # 2x4x2 Rank 16.
        # 1x4x2 Rank 8.
        # Total 24 for 3x4x2.
        # Total 72. No gain.
        
        # I MUST implement Strassen correctly.
        
    # Revised Plan:
    # 1. Implement add_strassen_2x2x2 with correct offsets.
    # 2. Use it for 2x2x2 blocks.
    # 3. Use Schoolbook for 1x2xK blocks.
    
    # Re-implementing add_strassen_vecs to take explicit indices
    def add_strassen_2x2x2(u_idx_map, v_idx_map, w_idx_map):
        # u_idx_map: dict mapping logical A index (0..3) to actual U row
        # v_idx_map: dict mapping logical B index (0..3) to actual V row
        # w_idx_map: dict mapping logical C index (0..3) to actual W row
        
        def get_idx(map, key): return map[key]
        
        # M1
        u = zero_vec(U_rows); v = zero_vec(V_rows); w = zero_vec(W_rows)
        u[get_idx(u_idx_map, 0)] = 1; u[get_idx(u_idx_map, 3)] = 1
        v[get_idx(v_idx_map, 0)] = 1; v[get_idx(v_idx_map, 3)] = 1
        w[get_idx(w_idx_map, 0)] = 1; w[get_idx(w_idx_map, 3)] = -1
        add_term_vecs(u, v, w)
        
        # M2
        u = zero_vec(U_rows); v = zero_vec(V_rows); w = zero_vec(W_rows)
        u[get_idx(u_idx_map, 2)] = 1; u[get_idx(u_idx_map, 3)] = 1
        v[get_idx(v_idx_map, 0)] = 1
        w[get_idx(w_idx_map, 2)] = 1; w[get_idx(w_idx_map, 3)] = -1
        add_term_vecs(u, v, w)
        
        # M3
        u = zero_vec(U_rows); v = zero_vec(V_rows); w = zero_vec(W_rows)
        u[get_idx(u_idx_map, 0)] = 1
        v[get_idx(v_idx_map, 1)] = 1; v[get_idx(v_idx_map, 3)] = -1
        w[get_idx(w_idx_map, 0)] = 1; w[get_idx(w_idx_map, 1)] = 1
        add_term_vecs(u, v, w)
        
        # M4
        u = zero_vec(U_rows); v = zero_vec(V_rows); w = zero_vec(W_rows)
        u[get_idx(u_idx_map, 3)] = 1
        v[get_idx(v_idx_map, 2)] = 1; v[get_idx(v_idx_map, 0)] = -1
        w[get_idx(w_idx_map, 0)] = 1; w[get_idx(w_idx_map, 2)] = 1
        add_term_vecs(u, v, w)
        
        # M5
        u = zero_vec(U_rows); v = zero_vec(V_rows); w = zero_vec(W_rows)
        u[get_idx(u_idx_map, 0)] = 1; u[get_idx(u_idx_map, 1)] = 1
        v[get_idx(v_idx_map, 3)] = 1
        w[get_idx(w_idx_map, 0)] = -1; w[get_idx(w_idx_map, 1)] = 1
        add_term_vecs(u, v, w)
        
        # M6
        u = zero_vec(U_rows); v = zero_vec(V_rows); w = zero_vec(W_rows)
        u[get_idx(u_idx_map, 2)] = 1; u[get_idx(u_idx_map, 0)] = -1
        v[get_idx(v_idx_map, 0)] = 1; v[get_idx(v_idx_map, 2)] = 1
        w[get_idx(w_idx_map, 3)] = 1
        add_term_vecs(u, v, w)
        
        # M7
        u = zero_vec(U_rows); v = zero_vec(V_rows); w = zero_vec(W_rows)
        u[get_idx(u_idx_map, 1)] = 1; u[get_idx(u_idx_map, 3)] = -1
        v[get_idx(v_idx_map, 3)] = 1; v[get_idx(v_idx_map, 1)] = 1
        w[get_idx(w_idx_map, 0)] = 1; w[get_idx(w_idx_map, 3)] = 1
        add_term_vecs(u, v, w)

    # Indices for A (3x4)
    # A_top (2x4): rows 0,1. Cols 0..3.
    # A_bot (1x4): row 2. Cols 0..3.
    
    # Indices for B (4x6)
    # B_left (4x4): rows 0..3. Cols 0..3.
    # B_right (4x2): rows 0..3. Cols 4..5.
    
    # Block 1: A_top * B_left (2x4 * 4x4)
    # Split inner 4 into 2+2.
    # A_top = [A11 (2x2), A12 (2x2)]
    # B_left = [B11 (2x4), B21 (2x4)]
    
    # A11: A rows 0,1; Cols 0,1.
    # A12: A rows 0,1; Cols 2,3.
    # B11: B rows 0,1; Cols 0..3.
    # B21: B rows 2,3; Cols 0..3.
    
    # Term 1.1: A11 * B11 (2x2 * 2x4)
    # Split B11 cols into 2+2.
    # B11a: Cols 0,1. B11b: Cols 2,3.
    # A11 * B11a (2x2x2) -> Strassen
    # A11 * B11b (2x2x2) -> Strassen
    
    # A11 u_map: 0->0, 1->1, 2->4, 3->5 (Row 0 Col 0, Row 0 Col 1, Row 1 Col 0, Row 1 Col 1)
    # Wait, U index is i*m + j.
    # A11 (i=0,1; j=0,1):
    # 00: 0*4+0=0
    # 01: 0*4+1=1
    # 10: 1*4+0=4
    # 11: 1*4+1=5
    u_map_A11 = {0: 0, 1: 1, 2: 4, 3: 5}
    
    # B11a (i=0,1; j=0,1):
    # 00: 0*6+0=0
    # 01: 0*6+1=1
    # 10: 1*6+0=6
    # 11: 1*6+1=7
    v_map_B11a = {0: 0, 1: 1, 2: 6, 3: 7}
    
    # C11a (i=0,1; j=0,1):
    # 00: 0*3+0=0
    # 01: 0*3+1=1
    # 10: 1*3+0=3
    # 11: 1*3+1=4
    w_map_C11a = {0: 0, 1: 1, 2: 3, 3: 4}
    
    add_strassen_2x2x2(u_map_A11, v_map_B11a, w_map_C11a)
    
    # B11b (i=0,1; j=2,3):
    # 00: 0*6+2=2
    # 01: 0*6+3=3
    # 10: 1*6+2=8
    # 11: 1*6+3=9
    v_map_B11b = {0: 2, 1: 3, 2: 8, 3: 9}
    
    # C11b (i=0,1; j=2,3):
    # 00: 0*3+2=2
    # 01: 0*3+3=3
    # 10: 1*3+2=5
    # 11: 1*3+3=6
    w_map_C11b = {0: 2, 1: 3, 2: 5, 3: 6}
    
    add_strassen_2x2x2(u_map_A11, v_map_B11b, w_map_C11b)
    
    # Term 1.2: A12 * B21 (2x2 * 2x4)
    # A12 (i=0,1; j=2,3):
    # 00: 0*4+2=2
    # 01: 0*4+3=3
    # 10: 1*4+2=6
    # 11: 1*4+3=7
    u_map_A12 = {0: 2, 1: 3, 2: 6, 3: 7}
    
    # B21a (i=2,3; j=0,1):
    # 00: 2*6+0=12
    # 01: 2*6+1=13
    # 10: 3*6+0=18
    # 11: 3*6+1=19
    v_map_B21a = {0: 12, 1: 13, 2: 18, 3: 19}
    
    # C12a (i=0,1; j=0,1): Same as C11a
    add_strassen_2x2x2(u_map_A12, v_map_B21a, w_map_C11a)
    
    # B21b (i=2,3; j=2,3):
    # 00: 2*6+2=14
    # 01: 2*6+3=15
    # 10: 3*6+2=20
    # 11: 3*6+3=21
    v_map_B21b = {0: 14, 1: 15, 2: 20, 3: 21}
    
    # C12b (i=0,1; j=2,3): Same as C11b
    add_strassen_2x2x2(u_map_A12, v_map_B21b, w_map_C11b)
    
    # Block 2: A_bot * B_left (1x4 * 4x4)
    # Schoolbook
    # A_bot (i=2; j=0..3)
    # B_left (i=0..3; j=0..3)
    # C_bot (i=2; j=0..3)
    
    # i=2
    for j in range(4):
        for k in range(4):
            u_vec = zero_vec(U_rows); v_vec = zero_vec(V_rows); w_vec = zero_vec(W_rows)
            u_vec[2*4 + j] = 1.0
            v_vec[j*6 + k] = 1.0
            w_vec[k*3 + 2] = 1.0
            add_term_vecs(u_vec, v_vec, w_vec)
            
    # Block 3: A_top * B_right (2x4 * 4x2)
    # A_top = [A11, A12]
    # B_right = [B12 (2x2), B22 (2x2)]
    
    # Term 3.1: A11 * B12 (2x2 * 2x2)
    # B12 (i=0,1; j=4,5):
    # 00: 0*6+4=4
    # 01: 0*6+5=5
    # 10: 1*6+4=10
    # 11: 1*6+5=11
    v_map_B12 = {0: 4, 1: 5, 2: 10, 3: 11}
    
    # C12 (i=0,1; j=4,5):
    # 00: 0*3+4=4
    # 01: 0*3+5=5
    # 10: 1*3+4=7
    # 11: 1*3+5=8
    w_map_C12 = {0: 4, 1: 5, 2: 7, 3: 8}
    
    add_strassen_2x2x2(u_map_A11, v_map_B12, w_map_C12)
    
    # Term 3.2: A12 * B22 (2x2 * 2x2)
    # B22 (i=2,3; j=4,5):
    # 00: 2*6+4=16
    # 01: 2*6+5=17
    # 10: 3*6+4=22
    # 11: 3*6+5=23
    v_map_B22 = {0: 16, 1: 17, 2: 22, 3: 23}
    
    add_strassen_2x2x2(u_map_A12, v_map_B22, w_map_C12)
    
    # Block 4: A_bot * B_right (1x4 * 4x2)
    # Schoolbook
    for j in range(4):
        for k in range(2):
            u_vec = zero_vec(U_rows); v_vec = zero_vec(V_rows); w_vec = zero_vec(W_rows)
            u_vec[2*4 + j] = 1.0
            v_vec[j*6 + (4+k)] = 1.0
            w_vec[(4+k)*3 + 2] = 1.0
            add_term_vecs(u_vec, v_vec, w_vec)

    return {"U": U_mat, "V": V_mat, "W": W_mat}
# EVOLVE-BLOCK-END
