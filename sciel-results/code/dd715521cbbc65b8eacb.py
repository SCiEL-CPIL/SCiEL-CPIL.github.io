<<<<<<< SEARCH
    # Get rank-23 decomposition for 3x3x3
    U23, V23, W23 = _rank23_3x3x3()
    
    # Get rank-3 decomposition for 3x3x1
    U3, V3, W3 = _schoolbook_3x3x1()
    
    # Build the full decomposition
    # For columns 0-2 (first 3x3 block):
    # U_full[i*3+j, t] = U23[i*3+j, t] for t=0..22
    # V_full[j*3+k, t] = V23[j*3+k, t] for k=0..2, t=0..22
    # W_full[k*3+i, t] = W23[k*3+i, t] for k=0..2, t=0..22
    
    # For columns 3-5 (second 3x3 block):
    # U_full[i*3+j, t] = U23[i*3+j, t-23] for t=23..45
    # V_full[j*3+(k+3), t] = V23[j*3+k, t-23] for k=0..2, t=23..45
    # W_full[(k+3)*3+i, t] = W23[k*3+i, t-23] for k=0..2, t=23..45
    
    # For columns 6 (3x1 block):
    # U_full[i*3+j, t] = U3[i*3+j, t-46] for t=46..48
    # V_full[j*3+6, t] = V3[j, t-46] for t=46..48
    # W_full[6*3+i, t] = W3[i, t-46] for t=46..48
    
    total_rank = 23 + 23 + 3  # 49
    U_full = np.zeros((n*m, total_rank))
    V_full = np.zeros((m*p, total_rank))
    W_full = np.zeros((p*n, total_rank))
    
    # First 3x3 block (columns 0-2)
    U_full[:, 0:23] = U23
    V_full[:, 0:23] = V23
    W_full[:, 0:23] = W23
    
    # Second 3x3 block (columns 3-5)
    U_full[:, 23:46] = U23
    # V: for j in 0..2, k in 0..2, maps to column k+3
    for j in range(3):
        for k in range(3):
            V_full[j*3 + (k+3), 23:46] = V23[j*3 + k, :]
    # W: for k in 0..2, i in 0..2, maps to row (k+3)*3+i
    for k in range(3):
        for i in range(3):
            W_full[(k+3)*3 + i, 23:46] = W23[k*3 + i, :]
    
    # 3x1 block (column 6)
    U_full[:, 46:49] = U3
    # V: for j in 0..2, column 6
    for j in range(3):
        V_full[j*3 + 6, 46:49] = V3[j, :]
    # W: for i in 0..2, row 6*3+i
    for i in range(3):
        W_full[6*3 + i, 46:49] = W3[i, :]
=======
    # Use schoolbook rank-9 for each 3x3x3 block and rank-3 for 3x3x1
    # Total rank: 9 + 9 + 3 = 21
    
    def _schoolbook_3x3x3():
        U = np.zeros((9, 9))
        V = np.zeros((9, 9))
        W = np.zeros((9, 9))
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U[i*3+j, col] = 1
                    V[j*3+k, col] = 1
                    W[k*3+i, col] = 1
                    col += 1
        return U, V, W
    
    U9, V9, W9 = _schoolbook_3x3x3()
    U3, V3, W3 = _schoolbook_3x3x1()
    
    total_rank = 9 + 9 + 3  # 21
    U_full = np.zeros((n*m, total_rank))
    V_full = np.zeros((m*p, total_rank))
    W_full = np.zeros((p*n, total_rank))
    
    # First 3x3 block (columns 0-2)
    U_full[:, 0:9] = U9
    V_full[:, 0:9] = V9
    W_full[:, 0:9] = W9
    
    # Second 3x3 block (columns 3-5)
    U_full[:, 9:18] = U9
    for j in range(3):
        for k in range(3):
            V_full[j*3 + (k+3), 9:18] = V9[j*3 + k, :]
    for k in range(3):
        for i in range(3):
            W_full[(k+3)*3 + i, 9:18] = W9[k*3 + i, :]
    
    # 3x1 block (column 6)
    U_full[:, 18:21] = U3
    for j in range(3):
        V_full[j*3 + 6, 18:21] = V3[j, :]
    for i in range(3):
        W_full[6*3 + i, 18:21] = W3[i, :]
>>>>>>> REPLACE

