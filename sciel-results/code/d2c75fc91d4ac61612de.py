    # j=0 block (r=0 to 5)
    for r in range(6):
        k = r
        # U: i=0,1; j=0 -> indices 0, 2
        U[0][r] = 1
        U[2][r] = 1
        
        # V: j=0; k=r -> index 0*6 + r = r
        V[r][r] = 1
        
        # W: k=r; i=0,1 -> indices r*2+0, r*2+1
        W[r*2][r] = 1
        W[r*2+1][r] = 1
        
    # j=1 block (r=6 to 11)
    for r in range(6, 12):
        k = r - 6
        # U: i=0,1; j=1 -> indices 1, 3
        U[1][r] = 1
        U[3][r] = 1
        
        # V: j=1; k=k -> index 1*6 + k = 6+k
        V[6+k][r] = 1
        
        # W: k=k; i=0,1 -> indices k*2+0, k*2+1
        W[k*2][r] = 1
        W[k*2+1][r] = 1
