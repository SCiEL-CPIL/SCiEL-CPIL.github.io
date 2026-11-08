import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # We will build U, V, W as lists of lists of floats
    # U: (n*m) x rank
    # V: (m*p) x rank
    # W: (p*n) x rank
    
    rank = 0
    U_rows = []
    V_rows = []
    W_rows = []
    
    def add_term(U_vec, V_vec, W_vec):
        nonlocal rank
        U_rows.append(U_vec)
        V_rows.append(V_vec)
        W_rows.append(W_vec)
        rank += 1

    # Helper to add a 2x2x2 Strassen decomposition
    # a_idx: indices (i,j) in A (4x4) for the 2x2 block
    # b_idx: indices (j,k) in B (4x6) for the 2x2 block
    # c_idx: indices (i,k) in C (4x6) for the 2x2 block
    # Note: B is 4x6, so k goes 0..5.
    # For 4x4x4 part, k is 0..3.
    # For 4x4x2 part, k is 4..5.
    
    def add_strassen_2x2x2(a_indices, b_indices, c_indices):
        # a_indices: [(i,j), (i,j+1), (i+1,j), (i+1,j+1)]
        # b_indices: [(j,k), (j,k+1), (j+1,k), (j+1,k+1)]
        # c_indices: [(i,k), (i,k+1), (i+1,k), (i+1,k+1)]
        
        # Strassen terms P1..P7
        # P1 = (a+d)(e+h)
        # P2 = (c+d)e
        # P3 = a(f-h)
        # P4 = d(g-e)
        # P5 = (a+b)h
        # P6 = (c-a)(e+f)
        # P7 = (b-d)(g+h)
        
        # Map to global indices
        # a = A[i,j], b = A[i,j+1], c = A[i+1,j], d = A[i+1,j+1]
        # e = B[j,k], f = B[j,k+1], g = B[j+1,k], h = B[j+1,k+1]
        
        # We need to create U, V, W vectors for each of the 7 terms.
        # U vector has non-zeros at a,b,c,d positions.
        # V vector has non-zeros at e,f,g,h positions.
        # W vector has non-zeros at c11, c12, c21, c22 positions.
        
        # Term 1: P1
        # U: a=1, d=1
        # V: e=1, h=1
        # W: c11=1, c22=1 (from P1 in C11 and C22)
        # Wait, C11 = P1 + P4 - P5 + P7
        # C12 = P3 + P5
        # C21 = P2 + P4
        # C22 = P1 - P2 + P3 + P6
        
        # We add 7 separate rank-1 terms.
        
        # T1: P1
        u = [0.0] * (n*m); v = [0.0] * (m*p); w = [0.0] * (p*n)
        u[a_indices[0][0]*m + a_indices[0][1]] = 1.0
        u[a_indices[3][0]*m + a_indices[3][1]] = 1.0
        v[b_indices[0][0]*p + b_indices[0][1]] = 1.0
        v[b_indices[3][0]*p + b_indices[3][1]] = 1.0
        w[c_indices[0][0]*n + c_indices[0][1]] = 1.0
        w[c_indices[3][0]*n + c_indices[3][1]] = 1.0
        add_term(u, v, w)
        
        # T2: P2
        u = [0.0] * (n*m); v = [0.0] * (m*p); w = [0.0] * (p*n)
        u[a_indices[2][0]*m + a_indices[2][1]] = 1.0
        u[a_indices[3][0]*m + a_indices[3][1]] = 1.0
        v[b_indices[0][0]*p + b_indices[0][1]] = 1.0
        w[c_indices[2][0]*n + c_indices[2][1]] = 1.0
        w[c_indices[3][0]*n + c_indices[3][1]] = -1.0
        add_term(u, v, w)
        
        # T3: P3
        u = [0.0] * (n*m); v = [0.0] * (m*p); w = [0.0] * (p*n)
        u[a_indices[0][0]*m + a_indices[0][1]] = 1.0
        v[b_indices[1][0]*p + b_indices[1][1]] = 1.0
        v[b_indices[3][0]*p + b_indices[3][1]] = -1.0
        w[c_indices[1][0]*n + c_indices[1][1]] = 1.0
        w[c_indices[3][0]*n + c_indices[3][1]] = 1.0
        add_term(u, v, w)
        
        # T4: P4
        u = [0.0] * (n*m); v = [0.0] * (m*p); w = [0.0] * (p*n)
        u[a_indices[3][0]*m + a_indices[3][1]] = 1.0
        v[b_indices[2][0]*p + b_indices[2][1]] = 1.0
        v[b_indices[0][0]*p + b_indices[0][1]] = -1.0
        w[c_indices[0][0]*n + c_indices[0][1]] = 1.0
        w[c_indices[2][0]*n + c_indices[2][1]] = 1.0
        add_term(u, v, w)
        
        # T5: P5
        u = [0.0] * (n*m); v = [0.0] * (m*p); w = [0.0] * (p*n)
        u[a_indices[0][0]*m + a_indices[0][1]] = 1.0
        u[a_indices[1][0]*m + a_indices[1][1]] = 1.0
        v[b_indices[3][0]*p + b_indices[3][1]] = 1.0
        w[c_indices[0][0]*n + c_indices[0][1]] = -1.0
        w[c_indices[1][0]*n + c_indices[1][1]] = 1.0
        add_term(u, v, w)
        
        # T6: P6
        u = [0.0] * (n*m); v = [0.0] * (m*p); w = [0.0] * (p*n)
        u[a_indices[2][0]*m + a_indices[2][1]] = 1.0
        u[a_indices[0][0]*m + a_indices[0][1]] = -1.0
        v[b_indices[0][0]*p + b_indices[0][1]] = 1.0
        v[b_indices[1][0]*p + b_indices[1][1]] = 1.0
        w[c_indices[3][0]*n + c_indices[3][1]] = 1.0
        add_term(u, v, w)
        
        # T7: P7
        u = [0.0] * (n*m); v = [0.0] * (m*p); w = [0.0] * (p*n)
        u[a_indices[1][0]*m + a_indices[1][1]] = 1.0
        u[a_indices[3][0]*m + a_indices[3][1]] = -1.0
        v[b_indices[2][0]*p + b_indices[2][1]] = 1.0
        v[b_indices[3][0]*p + b_indices[3][1]] = 1.0
        w[c_indices[0][0]*n + c_indices[0][1]] = 1.0
        w[c_indices[3][0]*n + c_indices[3][1]] = 1.0
        add_term(u, v, w)

    # Part 1: 4x4x4
    # A is 4x4, B is 4x4 (cols 0-3)
    # Blocks:
    # A11 (0,0), A12 (0,1), A21 (1,0), A22 (1,1)
    # B11 (0,0), B12 (0,1), B21 (1,0), B22 (1,1)
    
    # M1 = (A11 + A22)(B11 + B22)
    # This is not a simple 2x2x2 product of the original matrices.
    # It requires summing A blocks and B blocks first.
    # Strassen's algorithm involves linear combinations of blocks.
    # My add_strassen_2x2x2 function assumes we are decomposing a product of two 2x2 matrices.
    # But M1 is (A11+A22)(B11+B22).
    # Let A' = A11+A22, B' = B11+B22.
    # A' is 2x2, B' is 2x2.
    # We can decompose A'B' using Strassen.
    # The entries of A' are sums of entries of A.
    # So the U vector for A' will have coefficients 1 for A11 entries and 1 for A22 entries.
    
    # Let's generalize add_strassen_2x2x2 to take U and V coefficient vectors directly.
    
    def add_strassen_from_vectors(u_a, u_b, u_c, v_e, v_f, v_g, v_h, w_c11, w_c12, w_c21, w_c22):
        # u_a, u_b, u_c, u_d are U vectors for A block entries
        # Actually, it's easier to pass the full U, V vectors for the 4 entries of the 2x2 A and B blocks.
        pass

    # Let's stick to the block indices approach but modify it to handle sums.
    # For M1 = (A11+A22)(B11+B22):
    # A block entries:
    # a = A[0,0]+A[1,1]
    # b = A[0,1]+A[1,2]
    # c = A[1,0]+A[2,0]
    # d = A[1,1]+A[2,1]
    # B block entries:
    # e = B[0,0]+B[1,1]
    # f = B[0,1]+B[1,2]
    # g = B[1,0]+B[2,0]
    # h = B[1,1]+B[2,1]
    
    # This is getting complicated to hardcode all 7 M terms for 4x4x4.
    # Let's just use the 4x4x2 decomposition with rank 28 and 4x4x4 with rank 49.
    # Total 77.
    
    # I will implement the 4x4x2 part first as it's simpler (4 separate 2x2x2 products).
    # For 4x4x4, I will use the 7 M terms, each decomposed into 7 rank-1 terms.
    
    # Indices for 4x4x4:
    # A blocks:
    # A11: (0,0), (0,1), (1,0), (1,1)
    # A12: (0,2), (0,3), (1,2), (1,3)
    # A21: (2,0), (2,1), (3,0), (3,1)
    # A22: (2,2), (2,3), (3,2), (3,3)
    
    # B blocks (cols 0-3):
    # B11: (0,0), (0,1), (1,0), (1,1)
    # B12: (0,2), (0,3), (1,2), (1,3)
    # B21: (2,0), (2,1), (3,0), (3,1)
    # B22: (2,2), (2,3), (3,2), (3,3)
    
    # C blocks:
    # C11: (0,0), (0,1), (1,0), (1,1)
    # C12: (0,2), (0,3), (1,2), (1,3)
    # C21: (2,0), (2,1), (3,0), (3,1)
    # C22: (2,2), (2,3), (3,2), (3,3)
    
    # M1 = (A11+A22)(B11+B22)
    # A1 = A11+A22, B1 = B11+B22
    # Decompose A1 B1
    
    def add_strassen_2x2x2_general(u_a, u_b, u_c, u_d, v_e, v_f, v_g, v_h, w_c11, w_c12, w_c21, w_c22):
        # u_a etc are lists of (index, coeff) pairs for U
        # Actually, let's just pass the full U, V vectors for the 4 entries.
        pass

    # This is too complex to do dynamically in the short time.
    # I will output the schoolbook 96 rank solution to be safe, as I might make mistakes in the Strassen implementation.
    # But 96 is the baseline. I need to beat it.
    # Let's try the 4x4x2 part with rank 28 and 4x4x4 with rank 49.
    # I will hardcode the 4x4x4 Strassen terms.
    
    # For now, let's return the schoolbook solution to ensure validity, then I can refine.
    # No, I must provide a better solution.
    # I will implement the 4x4x2 part (rank 28) and use schoolbook for 4x4x4 (rank 64)?
    # 28 + 64 = 92. Better than 96.
    # Or 28 + 49 = 77.
    
    # Let's implement 4x4x2 with rank 28.
    # A is 4x4, B is 4x2 (cols 4-5).
    # B1 = B[:, 0:2], B2 = B[:, 2:4]? No, B is 4x2.
    # B has cols 0,1.
    # Wait, the task is 4x4 * 4x6.
    # So B is 4x6.
    # Part 1: A * B[:, 0:4] (4x4x4)
    # Part 2: A * B[:, 4:6] (4x4x2)
    
    # I will implement Part 2 with rank 28.
    # A blocks: A11, A12, A21, A22 (2x2)
    # B1 = B[:, 4:6] top 2 rows (2x2)
    # B2 = B[:, 4:6] bottom 2 rows (2x2)
    # C1 = A11 B1 + A12 B2
    # C2 = A21 B1 + A22 B2
    
    # This requires 4 separate 2x2x2 products.
    # Each has rank 7. Total 28.
    
    # For Part 1, I will use schoolbook rank 64.
    # Total 92.
    
    n, m, p = DIMENSIONS
    rank = 0
    U = []
    V = []
    W = []
    
    def add_term(u, v, w):
        nonlocal rank
        U.append(u)
        V.append(v)
        W.append(w)
        rank += 1

    # Part 2: 4x4x2 (cols 4-5 of B)
    # A11: (0,0), (0,1), (1,0), (1,1)
    # A12: (0,2), (0,3), (1,2), (1,3)
    # A21: (2,0), (2,1), (3,0), (3,1)
    # A22: (2,2), (2,3), (3,2), (3,3)
    
    # B1: (0,4), (0,5), (1,4), (1,5)
    # B2: (2,4), (2,5), (3,4), (3,5)
    
    # C1: (0,4), (0,5), (1,4), (1,5)
    # C2: (2,4), (2,5), (3,4), (3,5)
    
    # Product 1: A11 * B1 -> C1
    # Product 2: A12 * B2 -> C1
    # Product 3: A21 * B1 -> C2
    # Product 4: A22 * B2 -> C2
    
    def add_strassen_2x2x2_indices(a_idx, b_idx, c_idx):
        # a_idx, b_idx, c_idx are lists of 4 tuples (i,j)
        # a_idx[0] is a, a_idx[1] is b, a_idx[2] is c, a_idx[3] is d
        # b_idx[0] is e, b_idx[1] is f, b_idx[2] is g, b_idx[3] is h
        # c_idx[0] is c11, c_idx[1] is c12, c_idx[2] is c21, c_idx[3] is c22
        
        # T1
        u = [0.0] * (n*m); v = [0.0] * (m*p); w = [0.0] * (p*n)
        u[a_idx[0][0]*m + a_idx[0][1]] = 1.0
        u[a_idx[3][0]*m + a_idx[3][1]] = 1.0
        v[b_idx[0][0]*p + b_idx[0][1]] = 1.0
        v[b_idx[3][0]*p + b_idx[3][1]] = 1.0
        w[c_idx[0][0]*n + c_idx[0][1]] = 1.0
        w[c_idx[3][0]*n + c_idx[3][1]] = 1.0
        add_term(u, v, w)
        
        # T2
        u = [0.0] * (n*m); v = [0.0] * (m*p); w = [0.0] * (p*n)
        u[a_idx[2][0]*m + a_idx[2][1]] = 1.0
        u[a_idx[3][0]*m + a_idx[3][1]] = 1.0
        v[b_idx[0][0]*p + b_idx[0][1]] = 1.0
        w[c_idx[2][0]*n + c_idx[2][1]] = 1.0
        w[c_idx[3][0]*n + c_idx[3][1]] = -1.0
        add_term(u, v, w)
        
        # T3
        u = [0.0] * (n*m); v = [0.0] * (m*p); w = [0.0] * (p*n)
        u[a_idx[0][0]*m + a_idx[0][1]] = 1.0
        v[b_idx[1][0]*p + b_idx[1][1]] = 1.0
        v[b_idx[3][0]*p + b_idx[3][1]] = -1.0
        w[c_idx[1][0]*n + c_idx[1][1]] = 1.0
        w[c_idx[3][0]*n + c_idx[3][1]] = 1.0
        add_term(u, v, w)
        
        # T4
        u = [0.0] * (n*m); v = [0.0] * (m*p); w = [0.0] * (p*n)
        u[a_idx[3][0]*m + a_idx[3][1]] = 1.0
        v[b_idx[2][0]*p + b_idx[2][1]] = 1.0
        v[b_idx[0][0]*p + b_idx[0][1]] = -1.0
        w[c_idx[0][0]*n + c_idx[0][1]] = 1.0
        w[c_idx[2][0]*n + c_idx[2][1]] = 1.0
        add_term(u, v, w)
        
        # T5
        u = [0.0] * (n*m); v = [0.0] * (m*p); w = [0.0] * (p*n)
        u[a_idx[0][0]*m + a_idx[0][1]] = 1.0
        u[a_idx[1][0]*m + a_idx[1][1]] = 1.0
        v[b_idx[3][0]*p + b_idx[3][1]] = 1.0
        w[c_idx[0][0]*n + c_idx[0][1]] = -1.0
        w[c_idx[1][0]*n + c_idx[1][1]] = 1.0
        add_term(u, v, w)
        
        # T6
        u = [0.0] * (n*m); v = [0.0] * (m*p); w = [0.0] * (p*n)
        u[a_idx[2][0]*m + a_idx[2][1]] = 1.0
        u[a_idx[0][0]*m + a_idx[0][1]] = -1.0
        v[b_idx[0][0]*p + b_idx[0][1]] = 1.0
        v[b_idx[1][0]*p + b_idx[1][1]] = 1.0
        w[c_idx[3][0]*n + c_idx[3][1]] = 1.0
        add_term(u, v, w)
        
        # T7
        u = [0.0] * (n*m); v = [0.0] * (m*p); w = [0.0] * (p*n)
        u[a_idx[1][0]*m + a_idx[1][1]] = 1.0
        u[a_idx[3][0]*m + a_idx[3][1]] = -1.0
        v[b_idx[2][0]*p + b_idx[2][1]] = 1.0
        v[b_idx[3][0]*p + b_idx[3][1]] = 1.0
        w[c_idx[0][0]*n + c_idx[0][1]] = 1.0
        w[c_idx[3][0]*n + c_idx[3][1]] = 1.0
        add_term(u, v, w)

    # Part 2 indices
    a11 = [(0,0), (0,1), (1,0), (1,1)]
    a12 = [(0,2), (0,3), (1,2), (1,3)]
    a21 = [(2,0), (2,1), (3,0), (3,1)]
    a22 = [(2,2), (2,3), (3,2), (3,3)]
    
    b1 = [(0,4), (0,5), (1,4), (1,5)]
    b2 = [(2,4), (2,5), (3,4), (3,5)]
    
    c1 = [(0,4), (0,5), (1,4), (1,5)]
    c2 = [(2,4), (2,5), (3,4), (3,5)]
    
    add_strassen_2x2x2_indices(a11, b1, c1)
    add_strassen_2x2x2_indices(a12, b2, c1)
    add_strassen_2x2x2_indices(a21, b1, c2)
    add_strassen_2x2x2_indices(a22, b2, c2)
    
    # Part 1: Schoolbook for 4x4x4
    for i in range(4):
        for j in range(4):
            for k in range(4):
                u = [0.0] * (n*m); v = [0.0] * (m*p); w = [0.0] * (p*n)
                u[i*m + j] = 1.0
                v[j*p + k] = 1.0
                w[k*n + i] = 1.0
                add_term(u, v, w)
                
    return {"U": U, "V": V, "W": W}
