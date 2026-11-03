def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    # Rank calculation:
    # 2 blocks of A (m=4 split into two 2s)
    # Each block has 3 sub-blocks of B:
    #   1. 2x2x2 (Rank 8)
    #   2. 2x2x2 (Rank 8)
    #   3. 2x2x1 (Rank 4)
    # Total Rank = 2 * (8 + 8 + 4) = 40
    
    rank = 40
    
    U = np.zeros((n * m, rank))  # (8, 40)
    V = np.zeros((m * p, rank))  # (20, 40)
    W = np.zeros((p * n, rank))  # (10, 40)
    
    col_offset = 0
    
    for block in range(2):
        for sub in range(3):
            if sub < 2:
                # Schoolbook 2x2x2, B cols 2*sub, 2*sub+1
                sU, sV, sW = _strassen_2x2x2() # Now returns rank 8
                
                for c in range(8):
                    for i in range(2):
                        for j_local in range(2):
                            val = sU[i*2 + j_local, c]
                            if val != 0:
                                U[i*m + (2*block + j_local), col_offset + c] = val
                    
                    for j_local in range(2):
                        for k_local in range(2):
                            val = sV[j_local*2 + k_local, c]
                            if val != 0:
                                V[(2*block + j_local)*p + (2*sub + k_local), col_offset + c] = val
                    
                    for k_local in range(2):
                        for i in range(2):
                            val = sW[k_local*2 + i, c]
                            if val != 0:
                                W[(2*sub + k_local)*n + i, col_offset + c] = val
                
                col_offset += 8
            else:
                # Schoolbook 2x2x1, B col 4
                sU, sV, sW = _schoolbook_2x2x1()
                
                for c in range(4):
                    for i in range(2):
                        for j_local in range(2):
                            val = sU[i*2 + j_local, c]
                            if val != 0:
                                U[i*m + (2*block + j_local), col_offset + c] = val
                    
                    for j_local in range(2):
                        val = sV[j_local, c]
                        if val != 0:
                            V[(2*block + j_local)*p + 4, col_offset + c] = val
                    
                    for i in range(2):
                        val = sW[i, c]
                        if val != 0:
                            W[4*n + i, col_offset + c] = val
                
                col_offset += 4

    U_list = [[int(U[i, j]) if U[i, j] == int(U[i, j]) else float(U[i, j]) for j in range(rank)] for i in range(n*m)]
    V_list = [[int(V[i, j]) if V[i, j] == int(V[i, j]) else float(V[i, j]) for j in range(rank)] for i in range(m*p)]
    W_list = [[int(W[i, j]) if W[i, j] == int(W[i, j]) else float(W[i, j]) for j in range(rank)] for i in range(p*n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
