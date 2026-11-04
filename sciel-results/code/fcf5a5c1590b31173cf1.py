        # P1
        U[0*3+0][col] = 1; U[1*3+0][col] = -1
        V[0*8+k_end-1][col] = 1; V[1*8+k_end-1][col] = -1
        W[4*b+0][col] = -1
        W[4*b+3][col] = 1
        col += 1
        
        # P2
        U[0*3+0][col] = 1; U[1*3+0][col] = 1
        V[1*8+k_end-1][col] = 1
        W[4*b+1][col] = 1
        W[4*b+3][col] = -1
        col += 1
        
        # P3
        U[0*3+1][col] = 1; U[1*3+1][col] = -1
        V[0*8+k_start][col] = 1
        W[4*b+2][col] = 1
        W[4*b+3][col] = 1
        col += 1
        
        # P4
        U[0*3+0][col] = -1; U[1*3+1][col] = 1
        V[0*8+k_start][col] = 1; V[0*8+k_start+1][col] = 1
        W[4*b+0][col] = -1
        W[4*b+1][col] = 1
        col += 1
        
        # P5
        U[0*3+0][col] = 1; U[0*3+1][col] = 1
        V[1*8+k_end-1][col] = 1
        W[4*b+0][col] = 1
        W[4*b+2][col] = 1
        col += 1
        
        # P6
        U[0*3+0][col] = -1; U[1*3+0][col] = 1
        V[0*8+k_start][col] = 1; V[0*8+k_start+1][col] = 1
        W[4*b+3][col] = 1
        col += 1
        
        # P7
        U[0*3+1][col] = 1; U[1*3+1][col] = -1
        V[1*8+k_start][col] = 1; V[1*8+k_end-1][col] = -1
        W[4*b+0][col] = 1
        W[4*b+3][col] = 1
        col += 1
