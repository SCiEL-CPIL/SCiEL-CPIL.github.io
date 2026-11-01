<<<<<<< SEARCH
    col = 0
    # Part 1: 2x2x10 schoolbook
    for i in range(n):
        for j in range(m1):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Part 2: 2x1x10 rank-1
    # T[i, m1, k] = A[i, m1] * B[m1, k]
    # This is a single rank-1 term
    for i in range(n):
        U[i * m + m1, col] = 1
    for k in range(p):
        V[m1 * p + k, col] = 1
    for i in range(n):
        for k in range(p):
            W[k * n + i, col] = 1  # This needs to be A[i,m1]*B[m1,k]
    
    # Wait, the rank-1 term should be:
    # T[i, m1, k] = A[i, m1] * B[m1, k]
    # In the decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For j=m1, we want: U[i*m+m1, r] * V[m1*p+k, r] * W[k*n+i, r] = A[i,m1]*B[m1,k]
    # If we set U[i*m+m1, r] = 1, V[m1*p+k, r] = 1, W[k*n+i, r] = 1,
    # then we get 1*1*1 = 1, not A[i,m1]*B[m1,k]
    # 
    # The issue is that in the tensor decomposition, the entries of A and B are
    # the "variables", and the decomposition must work for ALL values of A and B.
    # So T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # must equal A[i,j]*B[j,k] for all A,B.
    # 
    # For the rank-1 part: T[i, m1, k] = A[i,m1]*B[m1,k]
    # We need: sum_r U[i*m+m1,r] * V[m1*p+k,r] * W[k*n+i,r] = A[i,m1]*B[m1,k]
    # 
    # This is a product of a function of (i) and a function of (k), which is
    # already rank 1 as a bilinear form. So we can set:
    # U[i*m+m1, r] = 1 for all i, V[m1*p+k, r] = 1 for all k, W[k*n+i, r] = 1 for all i,k
    # But that gives 1, not A[i,m1]*B[m1,k].
    # 
    # I'm confusing the tensor with the actual matrix product. Let me re-read the problem.
    # 
    # The tensor T[i,j,k] represents the coefficient of A[i,j]*B[j,k] in the product.
    # So T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k].
    # 
    # The decomposition is: T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # 
    # For the rank-1 part where j=m1:
    # T[i,m1,k] = 1 (it's the coefficient, not the value)
    # We need: sum_r U[i*m+m1,r] * V[m1*p+k,r] * W[k*n+i,r] = 1 for all i,k
    # 
    # If we set U[i*m+m1,r] = 1, V[m1*p+k,r] = 1, W[k*n+i,r] = 1, we get 1. Correct!
    
    # So the implementation above is actually correct. Let me clean it up.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
=======
    col = 0
    # Part 1: 2x2x10 schoolbook (j=0,1)
    for i in range(n):
        for j in range(m1):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Part 2: 2x1x10 rank-1 (j=2)
    # T[i, 2, k] = 1 for all i,k
    # Single rank term: U[i*3+2, col] = 1, V[20+k, col] = 1, W[k*2+i, col] = 1
    for i in range(n):
        U[i * m + m1, col] = 1
    for k in range(p):
        V[m1 * p + k, col] = 1
    for i in range(n):
        for k in range(p):
            W[k * n + i, col] = 1
    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
>>>>>>> REPLACE
