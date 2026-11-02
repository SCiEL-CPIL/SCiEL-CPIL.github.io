def strassen(i0, j0, k0):
    # P1
    add_term({i0*m+j0: 1, (i0+1)*m+(j0+1): 1},  # A11, A22
             {j0*p+k0: 1, (j0+1)*p+(k0+1): 1},  # B11, B22
             {k0*n+i0: 1, (k0+1)*n+(i0+1): 1})  # C11, C22 -> W[0,0]=1, W[1,1]=1
    # This matches C11 += P1, C22 += P1. Correct.
    
    # P2
    add_term({(i0+1)*m+j0: 1, (i0+1)*m+(j0+1): 1}, # A21, A22
             {j0*p+k0: 1},                          # B11
             {k0*n+(i0+1): 1})                      # C21 -> W[0,1]=1
    # This matches C21 += P2. Correct.
             
    # P3
    add_term({i0*m+j0: 1},                          # A11
             {(j0+1)*p+k0: 1, (j0+1)*p+(k0+1): -1}, # B21, -B22
             {k0*n+i0: 1, (k0+1)*n+i0: 1})          # C11, C12?
    # W[0,0]=1 (C11), W[1,0]=1 (C12? No, k=1, i=0 is C12? No, C is indexed by row i, col k.
    # So C_{ik}. W is indexed by k*n+i.
    # W[0*n+0] = W[0] -> C_{0,0} (C11).
    # W[1*n+0] = W[3] (if n=2) or W[1*3+0]=W[3] (if n=3).
    # In the code, n=3. So W index is k*3+i.
    # W[0*3+0] = W[0] -> C_{0,0}.
    # W[1*3+0] = W[3] -> C_{0,1}.
    # So P3 contributes to C11 and C12.
    # Standard Strassen: C12 = P3 + P5. C11 = ... + P7.
    # Wait, standard Strassen C11 does NOT have P3.
    # Let's check standard formulas again.
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    # My P3 is M3. M3 = A11(B21-B22).
    # So M3 should only contribute to C12.
    # The code has W entries for C11 and C12 for P3. This is WRONG.
