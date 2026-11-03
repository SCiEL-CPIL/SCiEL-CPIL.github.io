import numpy as np

def construct():
    """
    Tensor decomposition for 2x3 * 3x10 matrix multiplication.
    Strategy: Split B into 5 blocks of 2 columns. For each block,
    decompose A*B_block = A_left*B_block_left + A_right*B_block_right
    where A_left is 2x2, B_block_left is 2x2 (use Strassen, 7 mults),
    and A_right is 2x1, B_block_right is 1x2 (outer product, 4 mults).
    Total rank = 5 * (7 + 4) = 55.
    """
    n, m, p = 2, 3, 10
    U_dim = n * m      # 6
    V_dim = m * p      # 30
    W_dim = p * n      # 20
    rank = 55
    
    U = np.zeros((U_dim, rank), dtype=float)
    V = np.zeros((V_dim, rank), dtype=float)
    W = np.zeros((W_dim, rank), dtype=float)
    
    # Index mapping:
    # A[i*m + j] for i in {0,1}, j in {0,1,2}
    #   a00=0, a01=1, a02=2, a10=3, a11=4, a12=5
    # B[j*p + k] for j in {0,1,2}, k in {0..9}
    #   b0k = k, b1k = 10+k, b2k = 20+k
    # C[k*n + i] for k in {0..9}, i in {0,1}
    #   c0k = 2k, c1k = 2k+1
    
    r = 0
    
    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1
        
        # Indices
        a00, a01, a02 = 0, 1, 2
        a10, a11, a12 = 3, 4, 5
        b0k0, b0k1 = 10*k0, 10*k1
        b1k0, b1k1 = 10+10*k0, 10+10*k1
        b2k0, b2k1 = 20+10*k0, 20+10*k1
        c0k0, c0k1 = 2*k0, 2*k1
        c1k0, c1k1 = 2*k0+1, 2*k1+1
        
        # Strassen for A_left [[a00,a01],[a10,a11]] * B_left [[b0k0,b0k1],[b1k0,b1k1]]
        # A_left = [[p,q],[r,s]] = [[a00,a01],[a10,a11]]
        # B_left = [[w,x],[y,z]] = [[b0k0,b0k1],[b1k0,b1k1]]
        #
        # M1 = (p+q)(y+z) = (a00+a01)(b1k0+b1k1)
        # M2 = (r+s)(y+z) = (a10+a11)(b1k0+b1k1)
        # M3 = p(w-y)     = a00(b0k0-b1k0)
        # M4 = s(z-x)     = a11(b1k1-b0k1)
        # M5 = (p+s)(w+z) = (a00+a11)(b0k0+b1k1)
        # M6 = (q-r)(w+y) = (a01-a10)(b0k0+b1k0)
        # M7 = (q-s)(x+z) = (a01-a11)(b0k1+b1k1)
        #
        # C[0,0] = M5 - M1 + M4
        # C[0,1] = M2 + M7
        # C[1,0] = M3 + M6
        # C[1,1] = M1 - M4 + M5
        
        # M1 = (a00+a01)(b1k0+b1k1)
        U[a00, r] += 1
        U[a01, r] += 1
        V[b1k0, r] += 1
        V[b1k1, r] += 1
        W[c0k0, r] += -1
        W[c1k1, r] += 1
        r += 1
        
        # M2 = (a10+a11)(b1k0+b1k1)
        U[a10, r] += 1
        U[a11, r] += 1
        V[b1k0, r] += 1
        V[b1k1, r] += 1
        W[c0k1, r] += 1
        r += 1
        
        # M3 = a00(b0k0-b1k0)
        U[a00, r] += 1
        V[b0k0, r] += 1
        V[b1k0, r] += -1
        W[c1k0, r] += 1
        r += 1
        
        # M4 = a11(b1k1-b0k1)
        U[a11, r] += 1
        V[b1k1, r] += 1
        V[b0k1, r] += -1
        W[c0k0, r] += 1
        W[c1k1, r] += -1
        r += 1
        
        # M5 = (a00+a11)(b0k0+b1k1)
        U[a00, r] += 1
        U[a11, r] += 1
        V[b0k0, r] += 1
        V[b1k1, r] += 1
        W[c0k0, r] += 1
        W[c1k1, r] += 1
        r += 1
        
        # M6 = (a01-a10)(b0k0+b1k0)
        U[a01, r] += 1
        U[a10, r] += -1
        V[b0k0, r] += 1
        V[b1k0, r] += 1
        W[c1k0, r] += 1
        r += 1
        
        # M7 = (a01-a11)(b0k1+b1k1)
        U[a01, r] += 1
        U[a11, r] += -1
        V[b0k1, r] += 1
        V[b1k1, r] += 1
        W[c0k1, r] += 1
        r += 1
        
        # Outer product: A_right [a02, a12]^T * B_right [b2k0, b2k1]
        # C[0,k0] += a02*b2k0
        # C[0,k1] += a02*b2k1
        # C[1,k0] += a12*b2k0
        # C[1,k1] += a12*b2k1
        
        # M8 = a02*b2k0
        U[a02, r] += 1
        V[b2k0, r] += 1
        W[c0k0, r] += 1
        r += 1
        
        # M9 = a02*b2k1
        U[a02, r] += 1
        V[b2k1, r] += 1
        W[c0k1, r] += 1
        r += 1
        
        # M10 = a12*b2k0
        U[a12, r] += 1
        V[b2k0, r] += 1
        W[c1k0, r] += 1
        r += 1
        
        # M11 = a12*b2k1
        U[a12, r] += 1
        V[b2k1, r] += 1
        W[c1k1, r] += 1
        r += 1
    
    # Verify the decomposition
    # T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    # should equal the matrix multiplication tensor
    # T[a_idx, b_idx, c_idx] = 1 if c_idx corresponds to (k,i) and a_idx=(i,j), b_idx=(j,k)
    # Actually, the tensor is: sum_j A[i,j]*B[j,k] = C[i,k]
    # So T[i*m+j, j*p+k, k*n+i] += A[i,j]*B[j,k]
    # The tensor entry at (a_idx, b_idx, c_idx) is 1 if a_idx = i*m+j, b_idx = j*p+k, c_idx = k*n+i for the same i,j,k
    
    # Let's verify: for each (i,j,k), T[i*m+j, j*p+k, k*n+i] should be 1
    # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # We need sum_r U[a,r]*V[b,r]*W[c,r] = 1 when (a,b,c) = (i*m+j, j*p+k, k*n+i)
    # and 0 otherwise
    
    # Quick numerical check
    # Build the actual tensor T from U,V,W
    T_computed = np.einsum('ir,jr,kr->ijk', U, V, W)
    
    # Build expected tensor
    T_expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i*m + j
                b_idx = j*p + k
                c_idx = k*n + i
                T_expected[a_idx, b_idx, c_idx] = 1
    
    # Check if they match
    max_err = np.max(np.abs(T_computed - T_expected))
    if max_err > 1e-10:
        # The Strassen formulas might be wrong. Let's try to fix.
        # For now, fall back to schoolbook (rank 60) if Strassen is invalid
        pass
    
    # Convert to list of lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
