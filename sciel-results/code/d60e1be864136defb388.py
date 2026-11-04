def _strassen_rank7():
    # M1..M7 as (a_row, a_col, b_row, b_col, sign) contributions
    # A factors (U): index (i,j)
    # B factors (V): index (j,k)
    # C factors (W): index (k,i)
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    # Each M_c is a product of a sum of A terms and a sum of B terms.
    # Represent each M by list of (i,j,coef) for A and (j,k,coef) for B, then C = sum of M with signs.
    M = {
        0: ([(0,0,1),(1,1,1)], [(0,0,1),(0,1,1),(1,0,1),(1,1,1)]),  # (A11+A22)(B11+B12+B21+B22)
        1: ([(1,0,1),(1,1,1)], [(0,0,1)]),                            # (A21+A22)B11
        2: ([(0,0,1)], [(1,0,1),(1,1,-1)]),                           # A11(B21-B22)
        3: ([(1,1,1)], [(0,1,1),(0,0,-1)]),                           # A22(B12-B11)
        4: ([(0,0,1),(0,1,1)], [(1,1,1)]),                            # (A11+A12)B22
        5: ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)]),                   # (A21-A11)(B11+B12)
        6: ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)]),                   # (A12-A22)(B21+B22)
    }
    for c,(Aterms,Bterms) in M.items():
        for (i,j,s) in Aterms:
            U[i*2+j][c] += s
        for (j,k,s) in Bterms:
            V[j*2+k][c] += s
    # C11=M1+M4-M5+M7 -> W[0,0]: M1,M4,+ ; M5,- ; M7,+
    # C12=M3+M5 -> W[1,0]
    # C21=M2+M4 -> W[0,1]
    # C22=M1-M2+M3+M6 -> W[1,1]
    Csigns = {
        (0,0): {0:1, 3:1, 4:-1, 6:1},   # M1+M4-M5+M7
        (1,0): {2:1, 4:1},               # M3+M5
        (0,1): {1:1, 3:1},               # M2+M4
        (1,1): {0:1, 1:-1, 2:1, 5:1},    # M1-M2+M3+M6
    }
    for (k,i),sg in Csigns.items():
        for c,s in sg.items():
            W[k*2+i][c] += s
    return U, V, W
