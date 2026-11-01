import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # A = [[a, b], [c, d]]
    # B = [[e, f], [g, h]]
    # M1 = a(e+h)
    # M2 = (a+c)g
    # M3 = (b+d)e
    # M4 = d(h-f)
    # M5 = (a+c)(e+f)
    # M6 = (d-a)(g+h)
    # M7 = (b-c)(g+f)
    
    # C00 = M5 + M4 - M2 + M6
    # C01 = M1 + M2
    # C10 = M3 + M4
    # C11 = M5 - M1 + M3 - M7
    
    # Indices in A (2x2): 00, 01, 10, 11 -> 0, 1, 2, 3
    # Indices in B (2x2): 00, 01, 10, 11 -> 0, 1, 2, 3
    # Indices in C (2x2): 00, 01, 10, 11 -> 0, 1, 2, 3
    
    # U factors (coeffs for A)
    # V factors (coeffs for B)
    # W factors (coeffs for C)
    
    strassen_factors = [
        # M1
        { 'u': {0: 1}, 'v': {0: 1, 3: 1}, 'w': {1: 1, 3: 1} }, # C01 += M1, C11 -= M1
        # M2
        { 'u': {0: 1, 2: 1}, 'v': {2: 1}, 'w': {0: -1, 1: 1} }, # C00 -= M2, C01 += M2
        # M3
        { 'u': {1: 1, 3: 1}, 'v': {0: 1}, 'w': {2: 1, 3: 1} }, # C10 += M3, C11 += M3
        # M4
        { 'u': {3: 1}, 'v': {3: 1, 1: -1}, 'w': {0: 1, 2: 1} }, # C00 += M4, C10 += M4
        # M5
        { 'u': {0: 1, 2: 1}, 'v': {0: 1, 1: 1}, 'w': {0: 1, 3: 1} }, # C00 += M5, C11 += M5
        # M6
        { 'u': {3: 1, 0: -1}, 'v': {2: 1, 3: 1}, 'w': {0: 1} }, # C00 += M6
        # M7
        { 'u': {1: 1, 2: -1}, 'v': {2: 1, 1: 1}, 'w': {3: -1} }, # C11 -= M7
    ]
    
    # Map local 2x2 indices to global indices
    # A is 2x4. A1 is A[:, 0:2], A2 is A[:, 2:4]
    # B is 4x8. B1 is B[0:2, :], B2 is B[2:4, :]
    # B1k is B[0:2, 2k:2k+2], B2k is B[2:4, 2k:2k+2]
    # C is 2x8. Ck is C[:, 2k:2k+2]
    
    # U row index: i * m + j = i * 4 + j
    # V row index: j * p + k = j * 8 + k
    # W row index: k * n + i = k * 2 + i
    
    col = 0
    for k in range(4):
        # Process A1 * B1k
        # A1 corresponds to A rows 0,1 and cols 0,1
        # B1k corresponds to B rows 0,1 and cols 2k, 2k+1
        for s in range(7):
            fac = strassen_factors[s]
            # Update U
            for a_idx, val in fac['u'].items():
                # a_idx 0->00, 1->01, 2->10, 3->11 in A1
                # A1 row 0, col a_idx%2; A1 row 1, col a_idx//2?
                # A1 = [[a00, a01], [a10, a11]]
                # a_idx 0: (0,0) -> U[0*4+0]
                # a_idx 1: (0,1) -> U[0*4+1]
                # a_idx 2: (1,0) -> U[1*4+0]
                # a_idx 3: (1,1) -> U[1*4+1]
                r_u = (a_idx // 2) * 4 + (a_idx % 2)
                U[r_u][col] += val
            
            # Update V
            for b_idx, val in fac['v'].items():
                # b_idx 0->00, 1->01, 2->10, 3->11 in B1k
                # B1k row 0, col b_idx%2 + 2k
                # B1k row 1, col b_idx//2 + 2k
                # V row index: j * 8 + k_global
                # j is 0 or 1 (row in B1k)
                # k_global is 2k + (b_idx%2)
                j = b_idx // 2
                k_g = 2 * k + (b_idx % 2)
                r_v = j * 8 + k_g
                V[r_v][col] += val
            
            # Update W
            for c_idx, val in fac['w'].items():
                # c_idx 0->00, 1->01, 2->10, 3->11 in Ck
                # Ck row 0, col c_idx%2 + 2k
                # Ck row 1, col c_idx//2 + 2k
                # W row index: k_global * 2 + i
                # i is c_idx//2 (0 or 1)
                # k_global is 2k + (c_idx%2)
                i = c_idx // 2
                k_g = 2 * k + (c_idx % 2)
                r_w = k_g * 2 + i
                W[r_w][col] += val
            
            col += 1
        
        # Process A2 * B2k
        # A2 corresponds to A rows 0,1 and cols 2,3
        # B2k corresponds to B rows 2,3 and cols 2k, 2k+1
        for s in range(7):
            fac = strassen_factors[s]
            # Update U
            for a_idx, val in fac['u'].items():
                # A2 row 0, col a_idx%2 + 2
                # A2 row 1, col a_idx//2 + 2
                r_u = (a_idx // 2) * 4 + (a_idx % 2 + 2)
                U[r_u][col] += val
            
            # Update V
            for b_idx, val in fac['v'].items():
                # B2k row 0, col b_idx%2 + 2k -> global row 2, col ...
                # B2k row 1, col b_idx//2 + 2k -> global row 3, col ...
                j = b_idx // 2 + 2
                k_g = 2 * k + (b_idx % 2)
                r_v = j * 8 + k_g
                V[r_v][col] += val
            
            # Update W
            for c_idx, val in fac['w'].items():
                i = c_idx // 2
                k_g = 2 * k + (c_idx % 2)
                r_w = k_g * 2 + i
                W[r_w][col] += val
            
            col += 1

    return {"U": U, "V": V, "W": W}
