import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We aim for rank 71
    # 4 * (7 + 4) + 2 * (4 + 2) + 2 * (4 + 2) + 3
    # 44 + 12 + 12 + 3 = 71
    
    rank = 71
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        U[u_idx][col] = u_val
        V[v_idx][col] = v_val
        W[w_idx][col] = w_val
        col += 1
        
    def add_strassen(i_off, j_off, k_off):
        # Adds Strassen decomposition for 2x2x2 block at offsets
        # i_off: offset in i (0 or 1)
        # j_off: offset in j (0 or 1)
        # k_off: offset in k (0 or 1)
        # Local indices:
        # i: 0, 1 -> global i: i_off, i_off+1
        # j: 0, 1 -> global j: j_off, j_off+1
        # k: 0, 1 -> global k: k_off, k_off+1
        
        # Strassen terms:
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        
        # u vectors (for A)
        # v vectors (for B)
        # w vectors (for C output placement)
        
        # We need to map local (i,j,k) to global indices in U, V, W.
        # U index: i_global * m + j_global
        # V index: j_global * p + k_global
        # W index: k_global * n + i_global
        
        # Term 1: M1
        # u: [1, 1] for i=0,1? No, u is for specific i.
        # In rank-1 term, u is a vector of length n (local 2).
        # But our U is flattened.
        # We add 2 rank-1 terms for each M? No.
        # The Strassen algorithm computes 7 products.
        # Each product is a scalar.
        # We need to distribute this scalar to the output C.
        
        # Actually, the tensor decomposition of 2x2x2 using Strassen has rank 7.
        # Each of the 7 terms is a rank-1 tensor.
        
        # Term 1: u=[1,1], v=[1,1], w=[1,1] ?
        # Let's check the expansion.
        # (A v1) (W1^T B^T)^T
        # v1 = [1, 1]^T. A v1 = [a+b, c+d]^T.
        # w1 = [1, 1]^T. B^T w1 = [e+g, f+h]^T.
        # Product = [[(a+b)(e+g), (a+b)(f+h)], [(c+d)(e+g), (c+d)(f+h)]]
        # This is not M1.
        
        # The standard Strassen rank-1 decomposition is:
        # 1. u=[1,1], v=[1,1], w=[1,1] -> No, this gives (a+b)(e+g)...
        
        # Correct Strassen rank-1 tensors:
        # 1. u=[1,0], v=[1,1], w=[1,1] ?
        
        # Let's use the known rank-1 decomposition of the 2x2x2 multiplication tensor.
        # It is known that rank is 7.
        # The 7 terms are:
        # 1. u=[1,1], v=[1,1], w=[1,1] (This is not correct for Strassen)
        
        # Actually, Strassen's algorithm corresponds to:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Each M is a product of a linear form in A and a linear form in B.
        # So each M corresponds to a rank-1 term in the A-B part.
        # But we need to place it in C.
        # The placement is linear.
        # So for each M, we have a term:
        # u (linear form in A)
        # v (linear form in B)
        # w (linear form in C output)
        
        # For M1:
        # u_A = [1, 1] (coeffs for A00, A11) -> But A is 2x2.
        # u is a vector in R^2? No, u is a matrix?
        # In the flattened U, we specify the coefficients for each A_ij.
        
        # For M1:
        # A_part = A00 + A11.
        # U coeffs: U[0,0]=1, U[1,1]=1, others 0.
        # B_part = B00 + B11.
        # V coeffs: V[0,0]=1, V[1,1]=1, others 0.
        # C_part: M1 contributes to C00 (+1) and C11 (+1).
        # W coeffs: W[0,0]=1, W[1,1]=1, others 0.
        
        # So we add 1 term for M1.
        
        # Let's implement this logic.
        
        # Helper to add a Strassen term
        def add_strassen_term(u_coeffs, v_coeffs, w_coeffs):
            nonlocal col
            # u_coeffs: dict or list of (i,j, val)
            # v_coeffs: dict or list of (j,k, val)
            # w_coeffs: dict or list of (k,i, val)
            for i, j, val in u_coeffs:
                U[i_off + i, j_off + j] # Wait, U is flattened
                # U index: (i_off+i)*m + (j_off+j)
                U[(i_off + i) * m + (j_off + j)][col] = val
            for j, k, val in v_coeffs:
                V[(j_off + j) * p + (k_off + k)][col] = val
            for k, i, val in w_coeffs:
                W[(k_off + k) * n + (i_off + i)][col] = val
            col += 1
            
        # M1: (A00+A11)(B00+B11)
        add_strassen_term(
            [(0,0,1), (1,1,1)],
            [(0,0,1), (1,1,1)],
            [(0,0,1), (1,1,1)]
        )
        
        # M2: (A10+A11)B00
        add_strassen_term(
            [(1,0,1), (1,1,1)],
            [(0,0,1)],
            [(0,1,1)] # C10
        )
        
        # M3: A00(B01-B11)
        add_strassen_term(
            [(0,0,1)],
            [(0,1,1), (1,1,-1)],
            [(1,0,1)] # C01
        )
        
        # M4: A11(B10-B00)
        add_strassen_term(
            [(1,1,1)],
            [(1,0,1), (0,0,-1)],
            [(0,1,1), (1,0,1)] # C10 (+), C00 (-) -> Wait
            # C00 = M1 + M4 ...
            # C10 = M2 + M4 ...
            # So M4 contributes +1 to C10 and +1 to C00?
            # Formula: C00 = M1 + M4 - M5 + M7
            # So M4 coeff in C00 is +1.
            # C10 = M2 + M4
            # So M4 coeff in C10 is +1.
        )
        # Correction: W coeffs for M4
        # W[0,0] = 1 (C00)
        # W[1,0] = 1 (C10)
        # My previous call had [(0,1,1), (1,0,1)] which is C01 and C10?
        # Index (k,i).
        # C00 is k=0, i=0.
        # C10 is k=1, i=0.
        # So W[0,0]=1, W[1,0]=1.
        
        # Let's re-do M4
        # I already added it? No, I need to be careful.
        # I will restart the Strassen addition with a clean helper.
        
        col -= 4 # Rollback M1-M4 attempts to be safe? No, just implement correctly.
        
        # Let's define the 7 terms properly.
        
        # M1
        add_strassen_term(
            [(0,0,1), (1,1,1)],
            [(0,0,1), (1,1,1)],
            [(0,0,1), (1,1,1)]
        )
        
        # M2
        add_strassen_term(
            [(1,0,1), (1,1,1)],
            [(0,0,1)],
            [(1,0,1)] # C10
        )
        
        # M3
        add_strassen_term(
            [(0,0,1)],
            [(0,1,1), (1,1,-1)],
            [(1,0,1)] # C01
        )
        
        # M4
        add_strassen_term(
            [(1,1,1)],
            [(1,0,1), (0,0,-1)],
            [(0,0,1), (1,0,1)] # C00, C10
        )
        
        # M5
        add_strassen_term(
            [(0,0,1), (0,1,1)],
            [(1,1,1)],
            [(0,0,-1), (1,1,1)] # C00 (-), C11 (+)
        )
        
        # M6
        add_strassen_term(
            [(1,0,1), (0,0,-1)],
            [(0,0,1), (0,1,1)],
            [(1,1,1)] # C11
        )
        
        # M7
        add_strassen_term(
            [(0,1,1), (1,1,-1)],
            [(1,0,1), (1,1,1)],
            [(0,0,1), (1,1,-1)] # C00 (+), C11 (-)
        )
        
        # Total 7 terms.
        
    def add_schoolbook_1x2x2(i_off, j_off, k_off):
        # 1x2x2 multiplication
        # A is 1x2, B is 2x2, C is 1x2
        # C0 = A0 B00 + A1 B10
        # C1 = A0 B01 + A1 B11
        # Rank 4.
        
        # Term 1: A0 B00 -> C0
        add_term(
            (i_off + 0) * m + (j_off + 0),
            (j_off + 0) * p + (k_off + 0),
            (k_off + 0) * n + (i_off + 0),
            1, 1, 1
        )
        
        # Term 2: A1 B10 -> C0
        add_term(
            (i_off + 0) * m + (j_off + 1),
            (j_off + 1) * p + (k_off + 0),
            (k_off + 0) * n + (i_off + 0),
            1, 1, 1
        )
        
        # Term 3: A0 B01 -> C1
        add_term(
            (i_off + 0) * m + (j_off + 0),
            (j_off + 0) * p + (k_off + 1),
            (k_off + 1) * n + (i_off + 0),
            1, 1, 1
        )
        
        # Term 4: A1 B11 -> C1
        add_term(
            (i_off + 0) * m + (j_off + 1),
            (j_off + 1) * p + (k_off + 1),
            (k_off + 1) * n + (i_off + 0),
            1, 1, 1
        )

    # Part 1: 3x4x4
    # 4 blocks of 3x2x2
    # Block (0,0): i=0..2, j=0..1, k=0..1
    # Decompose into Strassen (i=0,1) + Schoolbook (i=2)
    
    # Block 0,0
    add_strassen(0, 0, 0)
    add_schoolbook_1x2x2(2, 0, 0)
    
    # Block 0,1: i=0..2, j=0..1, k=2..3
    add_strassen(0, 0, 2)
    add_schoolbook_1x2x2(2, 0, 2)
    
    # Block 1,0: i=0..2, j=2..3, k=0..1
    add_strassen(0, 2, 0)
    add_schoolbook_1x2x2(2, 2, 0)
    
    # Block 1,1: i=0..2, j=2..3, k=2..3
    add_strassen(0, 2, 2)
    add_schoolbook_1x2x2(2, 2, 2)
    
    # Part 2: 3x4x1
    # 2 blocks of 3x2x1
    # Block 0: i=0..2, j=0..1, k=4
    # Decompose into 2x2x1 (i=0,1) + 1x2x1 (i=2)
    
    def add_schoolbook_2x2x1(i_off, j_off, k_off):
        # 2x2x1
        # C0 = A00 B00 + A01 B10
        # C1 = A10 B00 + A11 B10
        add_term((i_off+0)*m+(j_off+0), (j_off+0)*p+(k_off), (k_off)*n+(i_off+0), 1,1,1)
        add_term((i_off+0)*m+(j_off+1), (j_off+1)*p+(k_off), (k_off)*n+(i_off+0), 1,1,1)
        add_term((i_off+1)*m+(j_off+0), (j_off+0)*p+(k_off), (k_off)*n+(i_off+1), 1,1,1)
        add_term((i_off+1)*m+(j_off+1), (j_off+1)*p+(k_off), (k_off)*n+(i_off+1), 1,1,1)
        
    def add_schoolbook_1x2x1(i_off, j_off, k_off):
        # 1x2x1
        # C0 = A0 B00 + A1 B10
        add_term((i_off+0)*m+(j_off+0), (j_off+0)*p+(k_off), (k_off)*n+(i_off+0), 1,1,1)
        add_term((i_off+0)*m+(j_off+1), (j_off+1)*p+(k_off), (k_off)*n+(i_off+0), 1,1,1)

    # Block 2,0
    add_schoolbook_2x2x1(0, 0, 4)
    add_schoolbook_1x2x1(2, 0, 4)
    
    # Block 2,1
    add_schoolbook_2x2x1(0, 2, 4)
    add_schoolbook_1x2x1(2, 2, 4)
    
    # Part 3: 3x1x4
    # 2 blocks of 3x1x2
    # Block 3,0: i=0..2, j=4, k=0..1
    # Decompose into 2x1x2 (i=0,1) + 1x1x2 (i=2)
    
    def add_schoolbook_2x1x2(i_off, j_off, k_off):
        # 2x1x2
        # C0 = A00 B00
        # C1 = A00 B01
        # C2 = A10 B00
        # C3 = A10 B01
        add_term((i_off+0)*m+(j_off), (j_off)*p+(k_off+0), (k_off+0)*n+(i_off+0), 1,1,1)
        add_term((i_off+0)*m+(j_off), (j_off)*p+(k_off+1), (k_off+1)*n+(i_off+0), 1,1,1)
        add_term((i_off+1)*m+(j_off), (j_off)*p+(k_off+0), (k_off+0)*n+(i_off+1), 1,1,1)
        add_term((i_off+1)*m+(j_off), (j_off)*p+(k_off+1), (k_off+1)*n+(i_off+1), 1,1,1)
        
    def add_schoolbook_1x1x2(i_off, j_off, k_off):
        # 1x1x2
        add_term((i_off+0)*m+(j_off), (j_off)*p+(k_off+0), (k_off+0)*n+(i_off+0), 1,1,1)
        add_term((i_off+0)*m+(j_off), (j_off)*p+(k_off+1), (k_off+1)*n+(i_off+0), 1,1,1)

    # Block 3,0
    add_schoolbook_2x1x2(0, 4, 0)
    add_schoolbook_1x1x2(2, 4, 0)
    
    # Block 3,1
    add_schoolbook_2x1x2(0, 4, 2)
    add_schoolbook_1x1x2(2, 4, 2)
    
    # Part 4: 3x1x1
    # Block 4,0: i=0..2, j=4, k=4
    def add_schoolbook_3x1x1(i_off, j_off, k_off):
        add_term((i_off+0)*m+(j_off), (j_off)*p+(k_off), (k_off)*n+(i_off+0), 1,1,1)
        add_term((i_off+1)*m+(j_off), (j_off)*p+(k_off), (k_off)*n+(i_off+1), 1,1,1)
        add_term((i_off+2)*m+(j_off), (j_off)*p+(k_off), (k_off)*n+(i_off+2), 1,1,1)
        
    add_schoolbook_3x1x1(0, 4, 4)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
