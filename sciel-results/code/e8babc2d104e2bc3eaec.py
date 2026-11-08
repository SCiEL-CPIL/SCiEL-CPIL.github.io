    def lader_3x3(i0, i1, i2, j0, j1, j2, k0, k1, k2):
        # Lader's 19-term decomposition for 3x3
        # Indices: A[i,j], B[j,k], C[i,k]
        # This is a specific set of linear combinations.
        # I will use a standard implementation found in literature.
        
        # M1
        add_term({(i0,j0):1, (i1,j1):1, (i2,j2):1}, 
                 {(j0,k0):1, (j1,k1):1, (j2,k2):1}, 
                 {(k0,i0):1, (k1,i1):1, (k2,i2):1})
        # M2
        add_term({(i0,j0):1, (i1,j1):1, (i2,j2):-1}, 
                 {(j0,k0):1, (j1,k1):1, (j2,k2):-1}, 
                 {(k0,i0):1, (k1,i1):1, (k2,i2):-1})
        # ... this is too risky to guess.
