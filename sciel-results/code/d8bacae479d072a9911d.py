<<<<<<< SEARCH
    w1 = zero_vec(9)
    # C00 += M1, C11 += M1
    w1[0] = 1 # k=0, i=0 -> 0*3+0=0
    w1[11] = 1 # k=2? No. C11 is i=1, k=1. W[1*3+1] = W[4].
    # Wait, W index is k*n + i.
    # C00: i=0, k=0. W[0].
    # C11: i=1, k=1. W[1*3+1] = W[4].
    # C01: i=0, k=1. W[1*3+0] = W[3].
    # C10: i=1, k=0. W[0*3+1] = W[1].
    
    # Let's re-verify W indices.
    # W[k*n + i]
    # C[i,k]
    # i=0, k=0 -> W[0]
    # i=0, k=1 -> W[3]
    # i=0, k=2 -> W[6]
    # i=1, k=0 -> W[1]
    # i=1, k=1 -> W[4]
    # i=1, k=2 -> W[7]
    # i=2, k=0 -> W[2]
    # i=2, k=1 -> W[5]
    # i=2, k=2 -> W[8]
    
    # Strassen M1: (A00+A11)(B00+B11)
    # Contributes to C00 (+1), C11 (+1)
    w1[0] = 1
    w1[4] = 1
    add_term(u1, v1, w1)
=======
    w1 = zero_vec(9)
    # Strassen M1: (A00+A11)(B00+B11)
    # Contributes to C00 (+1), C11 (+1)
    # W[k*n + i]: C00 -> W[0], C11 -> W[4]
    w1[0] = 1
    w1[4] = 1
    add_term(u1, v1, w1)
>>>>>>> REPLACE
