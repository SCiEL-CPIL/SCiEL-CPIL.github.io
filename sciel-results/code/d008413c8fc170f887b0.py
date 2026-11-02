import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = 2, 4, 4
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen steps: (x_coeffs, y_coeffs, z_coeffs)
    # x_coeffs: [a, b, c, d]
    # y_coeffs: [e, f, g, h]
    # z_coeffs: [z00, z01, z10, z11]
    steps = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),  # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]), # M3
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]), # M4
        ([1, 0, 1, 0], [0, 0, 0, 1], [-1, 1, 0, 0]), # M5
        ([-1, 0, 0, 1], [1, 1, 0, 0], [0, 0, 0, 1]), # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])  # M7
    ]
    
    # Sub-problems: (a_indices, b_indices, c_indices)
    # a_indices: [a, b, c, d] global indices in U
    # b_indices: [e, f, g, h] global indices in V
    # c_indices: [z00, z01, z10, z11] global indices in W
    
    # Problem 1: A_left * B_TL
    # A_left cols 0,1. A indices:
    # a=A[0,0]=0, b=A[0,1]=1, c=A[1,0]=4, d=A[1,1]=5
    a1 = [0, 1, 4, 5]
    # B_TL rows 0,1 cols 0,1. B indices:
    # e=B[0,0]=0, f=B[0,1]=1, g=B[1,0]=4, h=B[1,1]=5
    b1 = [0, 1, 4, 5]
    # C indices for cols 0,1
    # z00=C[0,0]->W[0], z01=C[0,1]->W[2], z10=C[1,0]->W[1], z11=C[1,1]->W[3]
    c1 = [0, 2, 1, 3]
    
    # Problem 2: A_left * B_TR
    a2 = a1
    # B_TR rows 0,1 cols 2,3.
    # e=B[0,2]=2, f=B[0,3]=3, g=B[1,2]=6, h=B[1,3]=7
    b2 = [2, 3, 6, 7]
    # C indices for cols 2,3
    # z00=C[0,2]->W[4], z01=C[0,3]->W[6], z10=C[1,2]->W[5], z11=C[1,3]->W[7]
    c2 = [4, 6, 5, 7]
    
    # Problem 3: A_right * B_BL
    # A_right cols 2,3.
    # a=A[0,2]=2, b=A[0,3]=3, c=A[1,2]=6, d=A[1,3]=7
    a3 = [2, 3, 6, 7]
    # B_BL rows 2,3 cols 0,1.
    # e=B[2,0]=8, f=B[2,1]=9, g=B[3,0]=12, h=B[3,1]=13
    b3 = [8, 9, 12, 13]
    # C indices same as P1
    c3 = c1
    
    # Problem 4: A_right * B_BR
    a4 = a3
    # B_BR rows 2,3 cols 2,3.
    # e=B[2,2]=10, f=B[2,3]=11, g=B[3,2]=14, h=B[3,3]=15
    b4 = [10, 11, 14, 15]
    # C indices same as P2
    c4 = c2
    
    problems = [
        (a1, b1, c1),
        (a2, b2, c2),
        (a3, b3, c3),
        (a4, b4, c4)
    ]
    
    col = 0
    for a_idx, b_idx, c_idx in problems:
        for x_c, y_c, z_c in steps:
            for i in range(4):
                if x_c[i] != 0:
                    U[a_idx[i]][col] += x_c[i]
                if y_c[i] != 0:
                    V[b_idx[i]][col] += y_c[i]
                if z_c[i] != 0:
                    W[c_idx[i]][col] += z_c[i]
            col += 1
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return {"U": U, "V": V, "W": W}
