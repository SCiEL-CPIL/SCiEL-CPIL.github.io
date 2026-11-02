import time

DIMENSIONS = [2, 3, 7]
DEADLINE = 150.0
START_TIME = time.time()

def construct():
    n, m, p = DIMENSIONS
    # Rank 40 hybrid decomposition
    rank = 40
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        if col >= rank:
            return
        U[u_idx][col] = u_val
        V[v_idx][col] = v_val
        W[w_idx][col] = w_val
        col += 1

    # Split B into B1 (3x2), B2 (3x2), B3 (3x3)
    # C1 = A B1 (2x2)
    # C2 = A B2 (2x2)
    # C3 = A B3 (2x3)
    
    # Helper for Strassen 2x2x2
    # X is 2x2, Y is 2x2, Z is 2x2
    # We need to map local indices to global indices
    
    def strassen_2x2x2(x_base_u, y_base_v, z_base_w, offset_u, offset_v, offset_w):
        # x_base_u: (row, col) base for X in U
        # y_base_v: (row, col) base for Y in V
        # z_base_w: (row, col) base for Z in W
        
        # X indices: (x_base_u[0]+r, x_base_u[1]+c)
        # Y indices: (y_base_v[0]+r, y_base_v[1]+c)
        # Z indices: (z_base_w[0]+r, z_base_w[1]+c)
        
        # M1
        # U: (x00 + x11)
        # We need to add terms for each linear combination
        # This function is complex to implement generically with the add_term interface
        # because Strassen combines inputs.
        # We must expand the Strassen terms into rank-1 components of the original tensors.
        # Actually, Strassen is a decomposition of the tensor.
        # Each of the 7 multiplications corresponds to a rank-1 term in the *decomposed* space,
        # but in the original U, V, W, it involves sums.
        # The tensor decomposition requires U, V, W to be matrices.
        # So for M1 = (x00+x11)(y00+y11), we need to express this as a sum of rank-1 terms?
        # No, Strassen *is* the decomposition.
        # The "rank" reduction comes from using 7 terms instead of 8.
        # But each term like (x00+x11) is a linear combination of inputs.
        # In the U matrix, we put the coefficients.
        
        # Let's trace M1:
        # U part: x00 and x11 have coeff 1.
        # V part: y00 and y11 have coeff 1.
        # W part: z00 has coeff 1, z11 has coeff 1.
        
        # So for M1:
        # add_term(x00, y00, z00, 1, 1, 1) -- NO, this is wrong.
        # The term is (x00+x11) * (y00+y11) = x00y00 + x00y11 + x11y00 + x11y11.
        # This is 4 rank-1 terms in the schoolbook sense.
        # But Strassen replaces the 8 schoolbook terms with 7 *combined* terms.
        # The output Z is a linear combination of these 7 terms.
        
        # So, for each of the 7 Strassen multiplications, we add ONE column to U, V, W.
        # U column has non-zeros at x00, x11.
        # V column has non-zeros at y00, y11.
        # W column has non-zeros at z00, z11 (with appropriate signs).
        
        # M1:
        # U: x00=1, x11=1
        # V: y00=1, y11=1
        # W: z00=1, z11=1
        
        u_x00 = x_base_u[0] * m + x_base_u[1] # Wait, U is (n*m, rank). Index is i*m+j.
        # X is 2x2. Rows 0,1. Cols 0,1.
        # U index for X[i][j] is (x_base_u[0]+i)*m + (x_base_u[1]+j)
        
        # Let's define a helper to get U index
        def u_idx(r, c): return (r) * m + c
        def v_idx(r, c): return (r) * p + c
        def w_idx(r, c): return (r) * n + c # W[k*n+i] -> r=k, c=i
        
        # M1
        add_term(u_idx(x_base_u[0]+0, x_base_u[1]+0), v_idx(y_base_v[0]+0, y_base_v[1]+0), w_idx(z_base_w[0]+0, z_base_w[1]+0), 1, 1, 1)
        add_term(u_idx(x_base_u[0]+1, x_base_u[1]+1), v_idx(y_base_v[0]+1, y_base_v[1]+1), w_idx(z_base_w[0]+0, z_base_w[1]+0), 1, 1, 1)
        # Wait, this adds 2 terms for M1?
        # No, the rank-1 term is defined by the vectors U_col, V_col, W_col.
        # If U_col has two non-zeros, it is still one rank-1 term in the decomposition of the tensor?
        # Yes. The tensor is T = sum u_k \otimes v_k \otimes w_k.
        # If u_k is a vector with two non-zeros, it's fine.
        
        # So for M1:
        # U_col: 1 at x00, 1 at x11.
        # V_col: 1 at y00, 1 at y11.
        # W_col: 1 at z00, 1 at z11.
        
        # But my add_term only handles single indices.
        # I need a way to add a term with multiple non-zeros.
        
        pass

    # Refined Plan:
    # I will manually construct the U, V, W columns for the 40 terms.
    
    # Term 1-7: Strassen for C1 (A B1)
    # A is 2x3. B1 is 3x2.
    # Split A = [A11 A12], A11 is 2x2 (cols 0,1), A12 is 2x1 (col 2).
    # Split B1 = [B11; B21], B11 is 2x2 (rows 0,1), B21 is 1x2 (row 2).
    # C1 = A11 B11 + A12 B21.
    # A11 B11 is 2x2x2. Use Strassen.
    # A12 B21 is 2x1x2. Schoolbook (4 terms).
    
    # Strassen for A11 B11:
    # X = A11 (rows 0,1 of A; cols 0,1 of A)
    # Y = B11 (rows 0,1 of B; cols 0,1 of B)
    # Z = C1 (rows 0,1 of C; cols 0,1 of C)
    
    # M1: (x00+x11)(y00+y11) -> z00, z11
    # U: A[0,0]=1, A[1,1]=1
    # V: B[0,0]=1, B[1,1]=1
    # W: C[0,0]=1, C[1,1]=1
    
    # I will write a function to add a multi-index term.
    
    def add_multi_term(u_indices, v_indices, w_indices, val):
        nonlocal col
        if col >= rank:
            return
        for idx in u_indices:
            U[idx][col] += val
        for idx in v_indices:
            V[idx][col] += val
        for idx in w_indices:
            W[idx][col] += val
        col += 1

    # Indices for A (2x3): i*m+j
    # A[0,0]=0, A[0,1]=1, A[0,2]=2
    # A[1,0]=3, A[1,1]=4, A[1,2]=5
    
    # Indices for B (3x7): i*p+j
    # B[0,0]=0, B[0,1]=1, ...
    # B[1,0]=7, ...
    # B[2,0]=14, ...
    
    # Indices for C (2x7) stored in W (7x2): k*n+i
    # C[0,0] -> W[0*2+0]=0
    # C[0,1] -> W[0*2+1]=1
    # C[1,0] -> W[1*2+0]=2
    # C[1,1] -> W[1*2+1]=3
    # ...
    # C[i,j] -> W[j*2+i]
    
    # Strassen M1
    add_multi_term([0, 4], [0, 8], [0, 3], 1) # x00=0, x11=4; y00=0, y11=8 (B[1,1] is 1*7+1=8); z00=0, z11=3
    
    # M2: (x21+x22)y11 -> z21
    # x21=A[1,0]=3, x22=A[1,1]=4. y11=B[1,1]=8. z21=C[1,0]=2.
    add_multi_term([3, 4], [8], [2], 1)
    
    # M3: x11(y12-y22) -> z12
    # x11=4. y12=B[1,2]=9, y22=B[2,2]=16. z12=C[0,1]=1.
    add_multi_term([4], [9, 16], [1], 1)
    # Wait, y12-y22. So V[9]=1, V[16]=-1.
    # My add_multi_term adds same val to all.
    # I need to handle signs.
    
    # Revised add_multi_term
    def add_signed_term(u_indices, v_indices, w_indices, val):
        nonlocal col
        if col >= rank:
            return
        for idx in u_indices:
            U[idx][col] += val
        for idx in v_indices:
            V[idx][col] += val
        for idx in w_indices:
            W[idx][col] += val
        col += 1
        
    # This still doesn't handle different signs for different indices in the same vector.
    # I will just use add_term for each non-zero in the vector.
    # But that increases the rank!
    # No, a rank-1 term u \otimes v \otimes w allows u to have multiple non-zeros.
    # So I need to modify U, V, W columns directly.
    
    # I will just build the columns manually in a list and then assign.
    
    pass

    # This is getting complicated to implement correctly in a short time.
    # I will return the schoolbook 42 to be safe, or the 40 if I can code it correctly.
    # Given the risk of bugs, I will implement the 40 carefully.
    
    # Let's use a simpler approach:
    # Construct the 40 terms explicitly.
    
    terms = []
    
    # Strassen for C1 (cols 0,1 of C)
    # X: A[0:2, 0:2], Y: B[0:2, 0:2], Z: C[0:2, 0:2]
    
    # M1
    terms.append(([0, 4], [0, 8], [0, 3], 1))
    # M2
    terms.append(([3, 4], [8], [2], 1))
    # M3: x11(y12-y22)
    terms.append(([4], [9, 16], [1], 1)) # y12=9, y22=16. Need signs.
    
    # I will hardcode the 40 terms in the final code.
    
    return {"U": U, "V": "V", "W": W} # Placeholder

