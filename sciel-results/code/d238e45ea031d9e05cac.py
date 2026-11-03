<<<<<<< SEARCH
    # I'm double-counting. The 2x2x2 blocks cover i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4,5}
    # That's 2*4*6 = 48 terms, using 12*7 = 84 rank.
    # Remaining: i=2 (all j,k): 4*7=28, and i in {0,1}, k=6: 2*4=8
    # Total terms: 48+28+8 = 84 ✓
    # Total rank: 84 + 28 + 8 = 120. Worse than schoolbook!
    
    # The problem is that Strassen for 2x2x2 has rank 7, but we're using it for
    # disjoint blocks, so we don't save anything overall.
    
    # Let me just return the schoolbook solution.
    
    U = np.zeros((12, 84))
    V = np.zeros((28, 84))
    W = np.zeros((21, 84))
    col = 0
    for i in range(3):
        for j in range(4):
            for k in range(7):
                U[i*4 + j, col] = 1
                V[j*7 + k, col] = 1
                W[k*3 + i, col] = 1
                col += 1
    
    U_list = [[float(round(U[i][c], 10)) for c in range(84)] for i in range(12)]
    V_list = [[float(round(V[i][c], 10)) for c in range(84)] for i in range(28)]
    W_list = [[float(round(W[i][c], 10)) for c in range(84)] for i in range(21)]
    return {"U": U_list, "V": V_list, "W": W_list}
=======
    # Use a simpler block decomposition: split 3 into 2+1.
    # Top 2x4*4x7: use schoolbook rank 56
    # Bottom 1x4*4x7: use schoolbook rank 28
    # Total: 84 (same as full schoolbook, but structured)
    
    # This doesn't improve, but it's correct.
    # For a real improvement, we'd need a better 2x4x7 decomposition.
    
    U = np.zeros((12, 84))
    V = np.zeros((28, 84))
    W = np.zeros((21, 84))
    col = 0
    
    # Top part: i in {0,1}
    for i in range(2):
        for j in range(4):
            for k in range(7):
                U[i*4 + j, col] = 1
                V[j*7 + k, col] = 1
                W[k*3 + i, col] = 1
                col += 1
    
    # Bottom part: i=2
    for j in range(4):
        for k in range(7):
            U[2*4 + j, col] = 1
            V[j*7 + k, col] = 1
            W[k*3 + 2, col] = 1
            col += 1
    
    U_list = [[float(round(U[i][c], 10)) for c in range(84)] for i in range(12)]
    V_list = [[float(round(V[i][c], 10)) for c in range(84)] for i in range(28)]
    W_list = [[float(round(W[i][c], 10)) for c in range(84)] for i in range(21)]
    return {"U": U_list, "V": V_list, "W": W_list}
>>>>>>> REPLACE
