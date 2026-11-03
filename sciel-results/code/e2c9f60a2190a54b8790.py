import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    n, m, p = DIMENSIONS
    num_A = n * m  # 6
    num_B = m * p  # 18
    num_C = p * n  # 12
    
    # Target rank 33
    rank = 33
    U = np.zeros((num_A, rank))
    V = np.zeros((num_B, rank))
    W = np.zeros((num_C, rank))
    
    col = 0
    
    # Part 1: A_L * B_T
    # A_L is A[:, 0] (2x1)
    # B_T is B[0, :] (1x6)
    # C_part = A_L * B_T (2x6)
    # Term for A_{i,0} * B_{0,k} -> C_{i,k}
    for i in range(n):
        for k in range(p):
            a = i * m + 0
            b = 0 * p + k
            c = k * n + i
            U[a, col] = 1.0
            V[b, col] = 1.0
            W[c, col] = 1.0
            col += 1
            
    # Part 2: A_R * B_B
    # A_R is A[:, 1:3] (2x2)
    # B_B is B[1:3, :] (2x6)
    # Split B_B into 3 blocks of 2x2: B_B[:, 0:2], B_B[:, 2:4], B_B[:, 4:6]
    # Each block multiplication is M_2,2,2 with rank 7 (Strassen)
    
    # Strassen coefficients for M_2,2,2
    # M1 = (X11+X22)(Y11+Y22)
    # M2 = (X21+X22)Y11
    # M3 = X11(Y12-Y22)
    # M4 = X22(Y21-Y11)
    # M5 = (X11+X12)Y22
    # M6 = (X21-X11)(Y11+Y12)
    # M7 = (X12-X22)(Y21+Y22)
    
    # Z11 = M1 + M4 - M5 + M7
    # Z12 = M3 + M5
    # Z21 = M2 + M4
    # Z22 = M1 - M2 + M3 + M6
    
    # X indices: X_{ij} corresponds to A_{i, j+1}
    # Y indices: Y_{jk} corresponds to B_{j+1, 2q+k}
    # Z indices: Z_{ik} corresponds to C_{i, 2q+k}
    
    for q in range(3):
        # Block q: B columns 2q, 2q+1
        
        # M1
        # X: (1,0) + (1,1) -> A_{0,1}, A_{1,2}
        # Y: (1,0) + (1,1) -> B_{1,2q}, B_{2,2q+1}
        # Z: (1,1) coeff 1, (2,2) coeff 1
        # Wait, Z11 and Z22 use M1 with coeff 1.
        
        def add_strassen_term(m_id, x_coeffs, y_coeffs, z_coeffs):
            nonlocal col
            # x_coeffs: dict (i,j) -> coeff
            # y_coeffs: dict (j,k) -> coeff
            # z_coeffs: dict (i,k) -> coeff
            
            for (i, j), cx in x_coeffs.items():
                a = i * m + (j + 1)
                U[a, col] += cx
                
            for (j, k), cy in y_coeffs.items():
                b = (j + 1) * p + (2 * q + k)
                V[b, col] += cy
                
            for (i, k), cz in z_coeffs.items():
                c = (2 * q + k) * n + i
                W[c, col] += cz
                
            col += 1

        # M1: (X11+X22)(Y11+Y22)
        add_strassen_term(1, 
                          {(0,0): 1, (1,1): 1}, 
                          {(0,0): 1, (1,1): 1}, 
                          {(0,0): 1, (1,1): 1})
                          
        # M2: (X21+X22)Y11
        add_strassen_term(2, 
                          {(1,0): 1, (1,1): 1}, 
                          {(0,0): 1}, 
                          {(1,0): 1})
                          
        # M3: X11(Y12-Y22)
        add_strassen_term(3, 
                          {(0,0): 1}, 
                          {(0,1): 1, (1,1): -1}, 
                          {(0,1): 1})
                          
        # M4: X22(Y21-Y11)
        add_strassen_term(4, 
                          {(1,1): 1}, 
                          {(1,0): 1, (0,0): -1}, 
                          {(0,0): 1, (1,0): 1})
                          
        # M5: (X11+X12)Y22
        add_strassen_term(5, 
                          {(0,0): 1, (0,1): 1}, 
                          {(1,1): 1}, 
                          {(0,0): -1, (0,1): 1})
                          
        # M6: (X21-X11)(Y11+Y12)
        add_strassen_term(6, 
                          {(1,0): 1, (0,0): -1}, 
                          {(0,0): 1, (0,1): 1}, 
                          {(1,1): 1})
                          
        # M7: (X12-X22)(Y21+Y22)
        add_strassen_term(7, 
                          {(0,1): 1, (1,1): -1}, 
                          {(1,0): 1, (1,1): 1}, 
                          {(0,0): 1})

    # Verify col is 12 + 3*7 = 33
    if col != 33:
        # Fallback to schoolbook if logic error
        rank = n * m * p
        U = np.zeros((num_A, rank))
        V = np.zeros((num_B, rank))
        W = np.zeros((num_C, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
                    if col == rank: break
                if col == rank: break
            if col == rank: break

    # Quick verification
    U_arr = U
    V_arr = V
    W_arr = W
    
    # Check a few random valid triples
    valid_triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                valid_triples.append((a, b, c))
                
    # Sample check
    import random
    random.seed(42)
    for _ in range(100):
        a, b, c = random.choice(valid_triples)
        val = np.sum(U_arr[a, :] * V_arr[b, :] * W_arr[c, :])
        if abs(val - 1.0) > 1e-9:
             # If verification fails, return schoolbook
             rank = n * m * p
             U = np.zeros((num_A, rank))
             V = np.zeros((num_B, rank))
             W = np.zeros((num_C, rank))
             col = 0
             for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        U[a, col] = 1
                        V[b, col] = 1
                        W[c, col] = 1
                        col += 1
             return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
