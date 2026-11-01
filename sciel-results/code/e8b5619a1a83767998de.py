    for b in range(num_blocks):
        base_rank = b * rank_per_block
        base_k = b * 2 # global column offset for this block
        
        # M1: (A00+A11)(B00+B11)
        r = base_rank
        U[0*m+0][r] = 1
        U[1*m+1][r] = 1
        V[0*p+base_k][r] = 1
        V[1*p+base_k+1][r] = 1
        # W: C00 += 1, C11 += 1
        W[base_k*n+0][r] = 1
        W[(base_k+1)*n+1][r] = 1
        
        # M2: (A10+A11)(B00)
        r = base_rank + 1
        U[1*m+0][r] = 1
        U[1*m+1][r] = 1
        V[0*p+base_k][r] = 1
        # W: C10 += 1, C11 += -1
        W[base_k*n+1][r] = 1
        W[(base_k+1)*n+1][r] = -1
        
        # M3: (A00)(B01-B11)
        r = base_rank + 2
        U[0*m+0][r] = 1
        V[0*p+base_k+1][r] = 1
        V[1*p+base_k+1][r] = -1
        # W: C01 += 1, C11 += 1
        W[(base_k+1)*n+0][r] = 1
        W[(base_k+1)*n+1][r] = 1
        
        # M4: (A11)(B10-B00)
        r = base_rank + 3
        U[1*m+1][r] = 1
        V[1*p+base_k][r] = 1
        V[0*p+base_k][r] = -1
        # W: C00 += 1, C10 += 1
        W[base_k*n+0][r] = 1
        W[base_k*n+1][r] = 1
        
        # M5: (A00+A01)(B11)
        r = base_rank + 4
        U[0*m+0][r] = 1
        U[0*m+1][r] = 1
        V[1*p+base_k+1][r] = 1
        # W: C00 += -1, C01 += 1
        W[base_k*n+0][r] = -1
        W[(base_k+1)*n+0][r] = 1
        
        # M6: (A10-A00)(B00+B01)
        r = base_rank + 5
        U[1*m+0][r] = 1
        U[0*m+0][r] = -1
        V[0*p+base_k][r] = 1
        V[0*p+base_k+1][r] = 1
        # W: C11 += 1
        W[(base_k+1)*n+1][r] = 1
        
        # M7: (A01-A11)(B10+B11)
        r = base_rank + 6
        U[0*m+1][r] = 1
        U[1*m+1][r] = -1
        V[1*p+base_k][r] = 1
        V[1*p+base_k+1][r] = 1
        # W: C00 += 1, C11 += -1
        W[base_k*n+0][r] = 1
        W[(base_k+1)*n+1][r] = -1
