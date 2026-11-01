import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = 50
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Strassen 2x2x2 decomposition
    # A = [[a, b], [c, d]]
    # B = [[x, y], [z, w]]
    # C = A B
    
    # Multiplications:
    # M1 = (a+b)(x+z)
    # M2 = b(z+w)
    # M3 = a(y-w)
    # M4 = d(a+b)
    # M5 = (a+c)b
    # M6 = (d-b)(a+c)
    # M7 = (a-b)d
    
    # Reconstruction:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We define the terms as (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: [a, b, c, d]
    # V_coeffs: [x, y, z, w]
    # W_coeffs: [C11, C12, C21, C22]
    
    strassen_terms = [
        ([1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1]),  # M1: (a+b)(x+z) -> C11, C22?
        # Wait, M1 = (a+b)(x+z).
        # C11 = M1 + ...
        # C22 = M1 - ...
        # So M1 contributes +1 to C11 and +1 to C22?
        # Let's check C22 = M1 - M2 + M3 + M6.
        # Yes, M1 is +1.
        ([0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, -1]), # M2: b(z+w) -> C21, -C22
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]), # M3: a(y-w) -> C12, C22
        ([0, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 0]), # M4: d(a+b) -> C11, C21
        ([1, 0, 1, 0], [0, 1, 0, 0], [-1, 1, 0, 0]), # M5: (a+c)b -> -C11, C12
        ([0, 0, 0, 1], [1, -1, 1, 0], [0, 0, 0, 1]), # M6: (d-b)(a+c) -> C22
        ([1, 0, 0, -1], [0, 0, 0, 1], [1, 0, 0, 0])  # M7: (a-b)d -> C11
    ]
    
    # Helper to add a 2x2x2 block
    def add_2x2_2(base_a, base_b, base_c):
        nonlocal col
        for u_c, v_c, w_c in strassen_terms:
            # Map A indices
            # a: base_a + 0
            # b: base_a + 1
            # c: base_a + 2
            # d: base_a + 3
            U[base_a + 0, col] += u_c[0]
            U[base_a + 1, col] += u_c[1]
            U[base_a + 2, col] += u_c[2]
            U[base_a + 3, col] += u_c[3]
            
            # Map B indices
            # x: base_b + 0
            # y: base_b + 1
            # z: base_b + 2
            # w: base_b + 3
            V[base_b + 0, col] += v_c[0]
            V[base_b + 1, col] += v_c[1]
            V[base_b + 2, col] += v_c[2]
            V[base_b + 3, col] += v_c[3]
            
            # Map C indices
            # C11: base_c + 0
            # C12: base_c + 1 (W index k*n+i. k=0, i=1? No. C is 2x2.
            # W index for C[row, col] is col*n + row.
            # C11: col=0, row=0 -> 0
            # C12: col=1, row=0 -> 1*n + 0 = 2
            # C21: col=0, row=1 -> 0*n + 1 = 1
            # C22: col=1, row=1 -> 1*n + 1 = 3
            W[base_c + 0, col] += w_c[0]
            W[base_c + 2, col] += w_c[1]
            W[base_c + 1, col] += w_c[2]
            W[base_c + 3, col] += w_c[3]
            
            col += 1

    # 1. C1 = A B[:, 0:2]
    # A = [A_L, A_R] (2x2 each)
    # B[:, 0:2] = [B_T; B_B] (2x2 each)
    # A_L is A[:, 0:2] -> U rows 0,1,2,3
    # A_R is A[:, 2:4] -> U rows 4,5,6,7
    # B_T is B[0:2, 0:2] -> V rows 0,1,2,3
    # B_B is B[2:4, 0:2] -> V rows 14,15,16,17 (2*7 + 0)
    # C1 is C[:, 0:2] -> W rows 0,1,2,3
    
    # Term 1: A_L B_T
    add_2x2_2(0, 0, 0)
    # Term 2: A_R B_B
    add_2x2_2(4, 14, 0)
    
    # 2. C2 = A B[:, 2:4]
    # B_T is B[0:2, 2:4] -> V rows 2,3,4,5
    # B_B is B[2:4, 2:4] -> V rows 16,17,18,19
    # C2 is C[:, 2:4] -> W rows 4,5,6,7
    
    add_2x2_2(0, 2, 4)
    add_2x2_2(4, 16, 4)
    
    # 3. C3 = A B[:, 4:6]
    # B_T is B[0:2, 4:6] -> V rows 4,5,6,7
    # B_B is B[2:4, 4:6] -> V rows 18,19,20,21
    # C3 is C[:, 4:6] -> W rows 8,9,10,11
    
    add_2x2_2(0, 4, 8)
    add_2x2_2(4, 18, 8)
    
    # 4. C4 = A B[:, 6:7]
    # B_T is B[0:2, 6:7] -> V rows 6,7 (2x1)
    # B_B is B[2:4, 6:7] -> V rows 20,21 (2x1)
    # C4 is C[:, 6:7] -> W rows 12,13 (2x1)
    
    # A_L b_T (2x2 x 2x1)
    # Rank 4.
    # a*x, b*y, c*x, d*y
    # U: a,b,c,d
    # V: x,y
    # W: C1, C2
    
    # Term 1: a*x
    U[0, col] = 1; V[6, col] = 1; W[12, col] = 1; col += 1
    # Term 2: b*y
    U[1, col] = 1; V[7, col] = 1; W[12, col] = 1; col += 1
    # Term 3: c*x
    U[2, col] = 1; V[6, col] = 1; W[13, col] = 1; col += 1
    # Term 4: d*y
    U[3, col] = 1; V[7, col] = 1; W[13, col] = 1; col += 1
    
    # A_R b_B (2x2 x 2x1)
    # Term 5: a*x
    U[4, col] = 1; V[20, col] = 1; W[12, col] = 1; col += 1
    # Term 6: b*y
    U[5, col] = 1; V[21, col] = 1; W[12, col] = 1; col += 1
    # Term 7: c*x
    U[6, col] = 1; V[20, col] = 1; W[13, col] = 1; col += 1
    # Term 8: d*y
    U[7, col] = 1; V[21, col] = 1; W[13, col] = 1; col += 1
    
    return {"U": U, "V": V, "W": W}
