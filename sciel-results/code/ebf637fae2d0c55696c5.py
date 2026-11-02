import numpy as np

DIMENSIONS = [5, 5, 5]

def get_strassen_2x2():
    """
    Returns U, V, W matrices for 2x2 Strassen decomposition.
    U, V, W have shape (4, 7).
    U rows: (0,0), (0,1), (1,0), (1,1)
    V rows: (0,0), (0,1), (1,0), (1,1)
    W rows: (0,0), (0,1), (1,0), (1,1)
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # M1 = (A00 + A11)(B00 + B11)
    # A part: A00 + A11 -> U[0]=1, U[3]=1
    # B part: B00 + B11 -> V[0]=1, V[3]=1
    # C part: M1 contributes to C00(+1), C11(+1)
    # Wait, the decomposition is sum of rank 1 tensors.
    # We need to expand M1 into 4 rank 1 terms?
    # No, the rank 7 decomposition IS the set of 7 terms.
    # Each term is a rank 1 tensor in the 2x2x2 space?
    # No, the 2x2 multiplication tensor is in R^{4x4x4}.
    # The rank 7 decomposition approximates this tensor.
    
    # The standard Strassen decomposition defines 7 products M1..M7.
    # Each Mj is a product of two linear forms.
    # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
    # This M1 is used in C00 and C11.
    
    # To create the U, V, W matrices, we iterate through the 7 M's.
    # For each M, we expand it into elementary A_ij B_kl products.
    # Each elementary product contributes one column to U, V, W.
    
    # Let's define the expansions.
    # M1: A00B00, A00B11, A11B00, A11B11
    # M2: A10B00, A11B00
    # M3: A00B01, -A00B11
    # M4: A11B10, -A11B00
    # M5: A00B11, A01B11
    # M6: A10B00, A10B01, -A00B00, -A00B01
    # M7: A01B10, A01B11, -A11B10, -A11B11
    
    # And the C combinations:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We will create a list of tuples (a_idx, b_idx, c_idx, coeff)
    # a_idx is 0..3 for A, b_idx is 0..3 for B, c_idx is 0..3 for C.
    
    terms = []
    
    # Helper to add terms from an M
    def add_m(m_id, a_terms, b_terms):
        # a_terms: list of (idx, coeff)
        # b_terms: list of (idx, coeff)
        # We need to know which C indices this M contributes to and with what coeff.
        pass
        
    # It's easier to just list all elementary terms directly.
    # Term: A_ij * B_kl -> C_mn with weight w.
    
    # M1 contributes to C00 (w=1) and C11 (w=1)
    # M1 = A00B00 + A00B11 + A11B00 + A11B11
    for a_i, a_j in [(0,0), (0,1), (1,0), (1,1)]:
        for b_i, b_j in [(0,0), (0,1), (1,0), (1,1)]:
            # Only if a_i==0 and b_i==0 OR a_i==1 and b_i==1?
            # No, M1 = (A00+A11)(B00+B11).
            # The A part is A00+A11. The B part is B00+B11.
            # So the products are:
            # A00*B00, A00*B11, A11*B00, A11*B11.
            
            # Let's just hardcode the 4 terms for M1.
            pass
            
    # This manual expansion is error-prone. Let's use the structure.
    # M1: A=(0,0)+(1,1), B=(0,0)+(1,1)
    # M2: A=(1,0)+(1,1), B=(0,0)
    # M3: A=(0,0), B=(0,1)-(1,1)
    # M4: A=(1,1), B=(1,0)-(0,0)
    # M5: A=(0,0)+(0,1), B=(1,1)
    # M6: A=(1,0)-(0,0), B=(0,0)+(0,1)
    # M7: A=(0,1)-(1,1), B=(1,0)+(1,1)
    
    # C00: M1(+1), M4(+1), M5(-1), M7(+1)
    # C01: M3(+1), M5(+1)
    # C10: M2(+1), M4(+1)
    # C11: M1(+1), M2(-1), M3(+1), M6(+1)
    
    # We will generate the columns.
    U_list = []
    V_list = []
    W_list = []
    
    # Define M's as lists of (a_idx, coeff) and (b_idx, coeff)
    Ms = []
    # M1
    Ms.append( ([(0,1), (3,1)], [(0,1), (3,1)]) )
    # M2
    Ms.append( ([(2,1), (3,1)], [(0,1)]) )
    # M3
    Ms.append( ([(0,1)], [(1,1), (3,-1)]) )
    # M4
    Ms.append( ([(3,1)], [(2,1), (0,-1)]) )
    # M5
    Ms.append( ([(0,1), (1,1)], [(3,1)]) )
    # M6
    Ms.append( ([(2,1), (0,-1)], [(0,1), (1,1)]) )
    # M7
    Ms.append( ([(1,1), (3,-1)], [(2,1), (3,1)]) )
    
    # C coeffs: C[m] = sum_s w_s M_s
    # C00: M1(1), M4(1), M5(-1), M7(1)
    # C01: M3(1), M5(1)
    # C10: M2(1), M4(1)
    # C11: M1(1), M2(-1), M3(1), M6(1)
    
    c_coeffs = [
        {0:1, 3:1, 4:-1, 6:1}, # C00
        {2:1, 4:1},             # C01
        {1:1, 3:1},             # C10
        {0:1, 1:-1, 2:1, 5:1}   # C11
    ]
    
    # For each M_s, and each term in its expansion:
    for s in range(7):
        a_terms, b_terms = Ms[s]
        # Expand A and B into elementary products
        for a_idx, a_coeff in a_terms:
            for b_idx, b_coeff in b_terms:
                # This is one elementary product A_a_idx * B_b_idx
                # It contributes to C_m with coefficient w_s * a_coeff * b_coeff
                # where w_s is the coefficient of M_s in C_m.
                
                u_vec = np.zeros(4)
                v_vec = np.zeros(4)
                w_vec = np.zeros(4)
                
                u_vec[a_idx] = a_coeff
                v_vec[b_idx] = b_coeff
                
                for m in range(4):
                    if s in c_coeffs[m]:
                        w_vec[m] = c_coeffs[m][s] * a_coeff * b_coeff
                
                U_list.append(u_vec)
                V_list.append(v_vec)
                W_list.append(w_vec)
                
    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # We will build the 5x5 decomposition using 1+4 split.
    
    U_list = []
    V_list = []
    W_list = []
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, coeff=1):
        u_vec = np.zeros(25)
        v_vec = np.zeros(25)
        w_vec = np.zeros(25)
        u_vec[u_idx] = coeff
        v_vec[v_idx] = coeff
        w_vec[w_idx] = coeff
        U_list.append(u_vec)
        V_list.append(v_vec)
        W_list.append(w_vec)

    # 1. A11 B11 (1x1 * 1x1)
    # A(0,0) B(0,0) -> C(0,0)
    add_term(0*5+0, 0*5+0, 0*5+0)
    
    # 2. A14 B41 (1x4 * 4x1)
    # A(0, j) B(j, 0) -> C(0,0) for j=1..4
    for j in range(1, 5):
        add_term(0*5+j, j*5+0, 0*5+0)
        
    # 3. A11 B14 (1x1 * 1x4)
    # A(0,0) B(0,k) -> C(k,0) for k=1..4
    for k in range(1, 5):
        add_term(0*5+0, 0*5+k, k*5+0)
        
    # 4. A14 B44 (1x4 * 4x4)
    # A(0,j) B(j,k) -> C(k,0) for j=1..4, k=1..4
    for j in range(1, 5):
        for k in range(1, 5):
            add_term(0*5+j, j*5+k, k*5+0)
            
    # 5. A41 B11 (4x1 * 1x1)
    # A(i,0) B(0,0) -> C(0,i) for i=1..4
    for i in range(1, 5):
        add_term(i*5+0, 0*5+0, 0*5+i)
        
    # 6. A44 B41 (4x4 * 4x1)
    # A(i,j) B(j,0) -> C(0,i) for i=1..4, j=1..4
    for i in range(1, 5):
        for j in range(1, 5):
            add_term(i*5+j, j*5+0, 0*5+i)
            
    # 7. A41 B14 (4x1 * 1x4)
    # A(i,0) B(0,k) -> C(k,i) for i=1..4, k=1..4
    for i in range(1, 5):
        for k in range(1, 5):
            add_term(i*5+0, 0*5+k, k*5+i)
            
    # 8. A44 B44 (4x4 * 4x4)
    # Use Strassen 4x4 decomposition
    U2, V2, W2 = get_strassen_2x2()
    
    # We need to map the 4x4 Strassen to the global 5x5 indices.
    # The 4x4 block is at indices 1..4.
    # Local indices 0..3 map to global 1..4.
    
    # The 4x4 Strassen algorithm uses 7 terms M1..M7.
    # Each term is a 2x2 multiplication.
    # We need to define the mapping for each of the 7 terms.
    
    # M1: (A11+A22)(B11+B22)
    # A11 is local 0..1, 0..1 -> global 1..2, 1..2
    # A22 is local 2..3, 2..3 -> global 3..4, 3..4
    
    # We will iterate through the 7 terms of the 4x4 Strassen.
    # For each term, we have a specific A_map and B_map.
    
    # Define the 7 terms of 4x4 Strassen
    # M1: A=(11,22), B=(11,22)
    # M2: A=(21,22), B=(11)
    # M3: A=(11), B=(12,22)
    # M4: A=(22), B=(21,11)
    # M5: A=(11,12), B=(22)
    # M6: A=(21,11), B=(11,12)
    # M7: A=(12,22), B=(21,22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to generate the 4x4 decomposition factors.
    # This is complex. Let's just use Schoolbook for 4x4?
    # No, 125 is the target to beat.
    # If we use Schoolbook for 4x4, we get 125.
    # We MUST use Strassen for 4x4.
    
    # Let's implement the 4x4 Strassen expansion.
    
    # For each of the 7 M's in 4x4:
    # 1. Define the local 2x2 A and B in terms of global 4x4 blocks.
    # 2. Use the 2x2 Strassen decomposition (U2, V2, W2) to compute the product.
    # 3. Map the result to global C.
    
    # This requires a function to expand a 2x2 decomposition into a 4x4 context.
    
    def expand_2x2_to_4x4(U2, V2, W2, a_blocks, b_blocks, c_blocks, c_coeffs):
        """
        a_blocks: list of 2x2 blocks (indices in 4x4) for A
        b_blocks: list of 2x2 blocks for B
        c_blocks: list of 2x2 blocks for C
        c_coeffs: dict mapping M_id to C_block_id and coeff
        """
        U_exp = []
        V_exp = []
        W_exp = []
        
        # a_blocks is a list of (row_offset, col_offset)
        # For M1: A = A11 + A22.
        # A11 is at (0,0), A22 is at (1,1) in 2x2 block space?
        # No, in 4x4 space, A11 is top-left 2x2.
        
        # Let's define the terms explicitly.
        pass

    # Given the time, I will fallback to a simpler improvement if Strassen 4x4 is too hard to code correctly in one go.
    # But 125 is the baseline.
    # Is there a simpler decomposition?
    # What if we use 2+3 split?
    # 2x2 (7), 2x3 (12), 3x2 (12), 3x3 (27).
    # Total: 7 + 12 + 12 + 18 + 12 + 18 + 18 + 27 = 124.
    # 124 is better than 125.
    
    # Let's implement the 2+3 split with Schoolbook for 3x3.
    
    # Clear previous terms
    U_list = []
    V_list = []
    W_list = []
    
    # 1. A22 B22 (2x2 * 2x2) -> Strassen (7)
    # 2. A23 B32 (2x3 * 3x2) -> Schoolbook (12)
    # 3. A22 B23 (2x2 * 2x3) -> Schoolbook (12)
    # 4. A23 B33 (2x3 * 3x3) -> Schoolbook (18)
    # 5. A32 B22 (3x2 * 2x2) -> Schoolbook (12)
    # 6. A33 B32 (3x3 * 3x2) -> Schoolbook (18)
    # 7. A32 B23 (3x2 * 2x3) -> Schoolbook (18)
    # 8. A33 B33 (3x3 * 3x3) -> Schoolbook (27)
    
    # Total 124.
    
    # Let's implement this.
    
    # 1. A22 B22
    U2, V2, W2 = get_strassen_2x2()
    # Map 2x2 indices to global 5x5 indices (0..1)
    for c in range(7):
        u_vec = np.zeros(25)
        v_vec = np.zeros(25)
        w_vec = np.zeros(25)
        for r in range(2):
            for col in range(2):
                idx = r * 2 + col
                u_vec[idx] = U2[idx, c]
                v_vec[idx] = V2[idx, c]
                w_vec[idx] = W2[idx, c]
        U_list.append(u_vec)
        V_list.append(v_vec)
        W_list.append(w_vec)
        
    # 2. A23 B32
    # A(i, j) i in 0..1, j in 2..4
    # B(j', k) j' in 2..4, k in 0..1
    # C(i, k) i in 0..1, k in 0..1
    for i in range(2):
        for j in range(2, 5):
            for k in range(2):
                add_term(i*5+j, j*5+k, k*5+i)
                
    # 3. A22 B23
    # A(i, j) i in 0..1, j in 0..1
    # B(j', k) j' in 0..1, k in 2..4
    # C(i, k) i in 0..1, k in 2..4
    for i in range(2):
        for j in range(2):
            for k in range(2, 5):
                add_term(i*5+j, j*5+k, k*5+i)
                
    # 4. A23 B33
    # A(i, j) i in 0..1, j in 2..4
    # B(j', k) j' in 2..4, k in 2..4
    # C(i, k) i in 0..1, k in 2..4
    for i in range(2):
        for j in range(2, 5):
            for k in range(2, 5):
                add_term(i*5+j, j*5+k, k*5+i)
                
    # 5. A32 B22
    # A(i, j) i in 2..4, j in 0..1
    # B(j', k) j' in 0..1, k in 0..1
    # C(i, k) i in 2..4, k in 0..1
    for i in range(2, 5):
        for j in range(2):
            for k in range(2):
                add_term(i*5+j, j*5+k, k*5+i)
                
    # 6. A33 B32
    # A(i, j) i in 2..4, j in 2..4
    # B(j', k) j' in 2..4, k in 0..1
    # C(i, k) i in 2..4, k in 0..1
    for i in range(2, 5):
        for j in range(2, 5):
            for k in range(2):
                add_term(i*5+j, j*5+k, k*5+i)
                
    # 7. A32 B23
    # A(i, j) i in 2..4, j in 0..1
    # B(j', k) j' in 0..1, k in 2..4
    # C(i, k) i in 2..4, k in 2..4
    for i in range(2, 5):
        for j in range(2):
            for k in range(2, 5):
                add_term(i*5+j, j*5+k, k*5+i)
                
    # 8. A33 B33
    # A(i, j) i in 2..4, j in 2..4
    # B(j', k) j' in 2..4, k in 2..4
    # C(i, k) i in 2..4, k in 2..4
    for i in range(2, 5):
        for j in range(2, 5):
            for k in range(2, 5):
                add_term(i*5+j, j*5+k, k*5+i)
                
    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
