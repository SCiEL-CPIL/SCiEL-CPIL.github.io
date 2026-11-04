import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Global dimensions
    dim_u = n * m  # 16
    dim_v = m * p  # 28
    dim_w = p * n  # 28
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_term(u, v, w):
        U_list.append(u)
        V_list.append(v)
        W_list.append(w)
        
    # --- Base Decompositions ---
    
    # Strassen 2x2x2 (Rank 7)
    # Returns U (4x7), V (4x7), W (4x7)
    def get_strassen_2x2():
        # Standard Strassen
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We need to express this as sum of rank-1 terms.
        # Each Mk is a 2x2x2 product.
        # We can use the schoolbook 2x2x2 (Rank 4) for each Mk?
        # No, that would be 7*4=28.
        # We want Rank 7.
        # The "Rank 7" refers to the algorithmic complexity.
        # Actually, the tensor rank of 2x2x2 is 7.
        # We need the explicit vectors.
        
        # Let's hardcode the 7 rank-1 terms for 2x2x2.
        # This is a known result.
        # U, V, W are 4x7.
        
        # Indices: a = 2*i + j, b = 2*j + k, c = 2*k + i
        
        # Term 1:
        u1 = np.array([1, 0, 0, 1]) # A11, A22
        v1 = np.array([1, 0, 0, 1]) # B11, B22
        w1 = np.array([1, 0, 0, 1]) # C11, C22? No.
        
        # It's easier to generate them by solving the linear system or using a known list.
        # Let's use the fact that we can just use Schoolbook for 2x2x2 (Rank 4) 
        # and then the total rank would be 7 * 4 = 28 for 4x4x4.
        # Total rank would be 28 + 4*4 + 4*4 = 28 + 16 + 16 = 60?
        # No, 4x4x4 with Schoolbook 2x2x2 blocks:
        # 7 Strassen steps, each 2x2x2 (Rank 4). Total 28.
        # 4x4x2: 4 terms, each 2x2x2 (Rank 4). Total 16.
        # 4x4x1: 4 terms, each 2x2x1 (Rank 4). Total 16.
        # Total Rank = 28 + 16 + 16 = 60.
        
        # 60 is much better than 93.
        # Why did I think Strassen 2x2x2 has rank 7?
        # Because the *algorithm* uses 7 multiplications.
        # But the *tensor rank* of 2x2x2 is 7.
        # If I use Schoolbook (Rank 4) for the base 2x2x2, the total rank is lower?
        # No, Schoolbook is Rank 4. Strassen is Rank 7.
        # Wait, Rank 4 is LESS than Rank 7.
        # So using Schoolbook for the base case is BETTER for rank minimization?
        # Yes!
        
        # Let's re-calculate with Base Rank 4 (Schoolbook 2x2x2).
        
        # Part 1 (4x4x4):
        # 7 Strassen steps.
        # Each step is a 2x2x2 multiplication.
        # If we decompose 2x2x2 with Rank 4 (Schoolbook), each step costs 4.
        # Total 7 * 4 = 28.
        
        # Part 2 (4x4x2):
        # 4 terms (A11 Btop, A12 Bbot, A21 Btop, A22 Bbot).
        # Each is 2x2x2. Rank 4.
        # Total 4 * 4 = 16.
        
        # Part 3 (4x4x1):
        # 4 terms (A11 Btop, A12 Bbot, A21 Btop, A22 Bbot).
        # Each is 2x2x1. Rank 4.
        # Total 4 * 4 = 16.
        
        # Total Rank = 28 + 16 + 16 = 60.
        
        # This is a huge improvement over 112.
        
        # Let's implement this.
        
        return None # Placeholder

    # Schoolbook 2x2x2 (Rank 4)
    # U (4x4), V (4x4), W (4x4)
    def get_schoolbook_2x2():
        U = np.zeros((4, 4))
        V = np.zeros((4, 4))
        W = np.zeros((4, 4))
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    t = i * 2 * 2 + j * 2 + k # 0..3? No, 2*2*2=8 terms?
                    # Wait, 2x2x2 Schoolbook has 8 terms?
                    # C_ik = sum_j A_ij B_jk.
                    # Terms: (i,j,k). 2*2*2 = 8 terms.
                    # Rank is 8?
                    # No, rank of 2x2x2 is 7 (Strassen) or 8 (Schoolbook)?
                    # Schoolbook uses 8 multiplications.
                    # So Rank 8.
                    
                    # If Base Rank is 8:
                    # Part 1: 7 * 8 = 56.
                    # Part 2: 4 * 8 = 32.
                    # Part 3: 4 * 4 = 16.
                    # Total 104.
                    # Still better than 112.
                    
        return None

    # Okay, let's just use the Rank 93 approach with Strassen (Rank 7) for 2x2x2.
    # It is safer and well-defined.
    
    # I will hardcode the 7 vectors for Strassen 2x2x2.
    
    # U, V, W for Strassen 2x2x2
    # See "The rank of 2x2x2 is 7"
    # I will generate them by solving the linear equations if necessary, 
    # or use a known set.
    
    # For now, I will use the Schoolbook 2x2x2 (Rank 8) because it is trivial to code.
    # Total Rank 104.
    
    def get_schoolbook_2x2():
        U = []
        V = []
        W = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    u = np.zeros(4)
                    v = np.zeros(4)
                    w = np.zeros(4)
                    u[i*2+j] = 1
                    v[j*2+k] = 1
                    w[k*2+i] = 1
                    U.append(u)
                    V.append(v)
                    W.append(w)
        return np.array(U).T, np.array(V).T, np.array(W).T

    def get_schoolbook_2x2x1():
        U = []
        V = []
        W = []
        for i in range(2):
            for j in range(2):
                for k in range(2): # k is 0 or 1? No, B is 2x1.
                    # B has 2 rows. j is row of A, k is row of B?
                    # A (2x2) B (2x1).
                    # C_i = sum_j A_ij B_j.
                    # Terms (i, j). j is 0..1.
                    # So 2*2 = 4 terms.
                    pass
        # Let's do it properly.
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        t = 0
        for i in range(2):
            for j in range(2):
                U[i*2+j, t] = 1
                V[j, t] = 1
                W[i, t] = 1
                t += 1
        return U, V, W

    # Part 1: 4x4x4 (Rank 56 with Schoolbook 2x2x2)
    # 7 Strassen steps.
    
    # Strassen Coefficients
    # A blocks: 00, 01, 10, 11
    # B blocks: 00, 01, 10, 11
    
    # Step 0: M1 = (A00+A11)(B00+B11)
    # Step 1: M2 = (A10+A11)B00
    # Step 2: M3 = A00(B01-B11)
    # Step 3: M4 = A11(B10-B00)
    # Step 4: M5 = (A00+A01)B11
    # Step 5: M6 = (A10-A00)(B00+B01)
    # Step 6: M7 = (A01-A11)(B10+B11)
    
    # C blocks:
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    # We will iterate through the 7 steps.
    # For each step, we have alpha (for A) and beta (for B).
    # We use the Schoolbook 2x2x2 decomposition (8 terms).
    # For each of the 8 terms, we add to global U, V, W.
    
    U_s22, V_s22, W_s22 = get_schoolbook_2x2() # Shapes 4x8
    
    steps = [
        ({'00': 1, '11': 1}, {'00': 1, '11': 1}), # M1
        ({'10': 1, '11': 1}, {'00': 1}),           # M2
        ({'00': 1}, {'01': 1, '11': -1}),          # M3
        ({'11': 1}, {'10': 1, '00': -1}),          # M4
        ({'00': 1, '01': 1}, {'11': 1}),           # M5
        ({'10': 1, '00': -1}, {'00': 1, '01': 1}), # M6
        ({'01': 1, '11': -1}, {'10': 1, '11': 1})  # M7
    ]
    
    # C contributions
    c_contribs = [
        {'00': 1, '11': 1},      # M1 -> C00, C11
        {'10': 1},               # M2 -> C10
        {'01': 1},               # M3 -> C01
        {'00': 1, '10': 1},      # M4 -> C00, C10
        {'01': 1, '00': -1},     # M5 -> C01, C00
        {'11': 1},               # M6 -> C11
        {'00': 1, '11': -1}      # M7 -> C00, C11
    ]
    # Corrected C contributions based on standard Strassen:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    c_contribs = [
        {'00': 1, '11': 1},      # M1
        {'10': 1, '11': -1},     # M2
        {'01': 1, '11': 1},      # M3
        {'00': 1, '10': 1},      # M4
        {'01': 1, '00': -1},     # M5
        {'11': 1},               # M6
        {'00': 1, '11': -1}      # M7
    ]

    for s in range(7):
        alpha = steps[s][0]
        beta = steps[s][1]
        gamma = c_contribs[s]
        
        for t in range(8): # 8 terms in schoolbook 2x2x2
            u_local = U_s22[:, t]
            v_local = V_s22[:, t]
            w_local = W_s22[:, t]
            
            u_global = np.zeros(16)
            v_global = np.zeros(28)
            w_global = np.zeros(28)
            
            # Map A
            for block, coeff in alpha.items():
                if coeff == 0: continue
                r, c = int(block[0]), int(block[1])
                for i in range(2):
                    for j in range(2):
                        u_global[(r*2+i)*4 + (c*2+j)] = coeff * u_local[i*2+j]
                        
            # Map B (Cols 0-3)
            for block, coeff in beta.items():
                if coeff == 0: continue
                r, c = int(block[0]), int(block[1])
                for i in range(2):
                    for j in range(2):
                        v_global[(r*2+i)*7 + (c*2+j)] = coeff * v_local[i*2+j]
                        
            # Map C (Cols 0-3)
            for block, coeff in gamma.items():
                if coeff == 0: continue
                r, c = int(block[0]), int(block[1])
                for i in range(2):
                    for k in range(2):
                        w_global[(c*2+k)*4 + (r*2+i)] = coeff * w_local[k*2+i]
                        
            add_term(u_global, v_global, w_global)

    # Part 2: 4x4x2 (Cols 4-5)
    # 4 terms, each 2x2x2.
    # A11 Btop, A12 Bbot, A21 Btop, A22 Bbot
    
    terms_p2 = [
        ({'00': 1}, {'00': 1}, {'00': 1}), # A11 Btop -> Ctop
        ({'01': 1}, {'11': 1}, {'00': 1}), # A12 Bbot -> Ctop
        ({'10': 1}, {'00': 1}, {'10': 1}), # A21 Btop -> Cbot
        ({'11': 1}, {'11': 1}, {'10': 1})  # A22 Bbot -> Cbot
    ]
    
    for alpha, beta, gamma in terms_p2:
        for t in range(8):
            u_local = U_s22[:, t]
            v_local = V_s22[:, t]
            w_local = W_s22[:, t]
            
            u_global = np.zeros(16)
            v_global = np.zeros(28)
            w_global = np.zeros(28)
            
            for block, coeff in alpha.items():
                r, c = int(block[0]), int(block[1])
                for i in range(2):
                    for j in range(2):
                        u_global[(r*2+i)*4 + (c*2+j)] = coeff * u_local[i*2+j]
                        
            # B Cols 4-5
            # Block 00 of 4x2 is rows 0-1, cols 4-5
            # Block 11 of 4x2 is rows 2-3, cols 4-5
            for block, coeff in beta.items():
                r, c = int(block[0]), int(block[1])
                # c is always 0 in local 4x2, but global cols are 4-5
                # Local col 0 -> Global 4, Local col 1 -> Global 5
                for i in range(2):
                    for j in range(2):
                        v_global[(r*2+i)*7 + (4+c*2+j)] = coeff * v_local[i*2+j]
                        
            # C Cols 4-5
            for block, coeff in gamma.items():
                r, c = int(block[0]), int(block[1])
                for i in range(2):
                    for k in range(2):
                        w_global[(4+c*2+k)*4 + (r*2+i)] = coeff * w_local[k*2+i]
                        
            add_term(u_global, v_global, w_global)

    # Part 3: 4x4x1 (Col 6)
    # 4 terms, each 2x2x1.
    
    U_s221, V_s221, W_s221 = get_schoolbook_2x2x1() # 4x4, 2x4, 2x4
    
    terms_p3 = [
        ({'00': 1}, {'00': 1}, {'00': 1}), # A11 Btop -> Ctop
        ({'01': 1}, {'11': 1}, {'00': 1}), # A12 Bbot -> Ctop
        ({'10': 1}, {'00': 1}, {'10': 1}), # A21 Btop -> Cbot
        ({'11': 1}, {'11': 1}, {'10': 1})  # A22 Bbot -> Cbot
    ]
    
    for alpha, beta, gamma in terms_p3:
        for t in range(4):
            u_local = U_s221[:, t]
            v_local = V_s221[:, t]
            w_local = W_s221[:, t]
            
            u_global = np.zeros(16)
            v_global = np.zeros(28)
            w_global = np.zeros(28)
            
            for block, coeff in alpha.items():
                r, c = int(block[0]), int(block[1])
                for i in range(2):
                    for j in range(2):
                        u_global[(r*2+i)*4 + (c*2+j)] = coeff * u_local[i*2+j]
                        
            # B Col 6
            # Block 00 of 4x1 is rows 0-1, col 6
            # Block 11 of 4x1 is rows 2-3, col 6
            for block, coeff in beta.items():
                r, c = int(block[0]), int(block[1])
                for i in range(2):
                    v_global[(r*2+i)*7 + 6] = coeff * v_local[i]
                    
            # C Col 6
            for block, coeff in gamma.items():
                r, c = int(block[0]), int(block[1])
                for i in range(2):
                    w_global[6*4 + (r*2+i)] = coeff * w_local[i]
                    
            add_term(u_global, v_global, w_global)

    rank = len(U_list)
    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
