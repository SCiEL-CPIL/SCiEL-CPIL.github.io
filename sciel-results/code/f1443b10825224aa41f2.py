"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 7
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_term(u_val, v_val, w_val):
        U_list.append(u_val)
        V_list.append(v_val)
        W_list.append(w_val)
    
    def add_schoolbook(i, j, k):
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        u[i * m + j] = 1
        v[j * p + k] = 1
        w[k * n + i] = 1
        add_term(u, v, w)
    
    def add_strassen(i0, j0, k0):
        # Strassen decomposition for 2x2x2 starting at (i0, j0, k0)
        # A = [[a,b],[c,d]] = A[i0:i0+2, j0:j0+2]
        # B = [[e,f],[g,h]] = B[j0:j0+2, k0:k0+2]
        # C = [[p,q],[r,s]] = C[i0:i0+2, k0:k0+2]
        
        a = (i0, j0)
        b = (i0, j0+1)
        c = (i0+1, j0)
        d = (i0+1, j0+1)
        e = (j0, k0)
        f = (j0, k0+1)
        g = (j0+1, k0)
        h = (j0+1, k0+1)
        
        # M1 = (a+d)(e+h)
        u1 = [0]*(n*m); v1 = [0]*(m*p); w1 = [0]*(p*n)
        u1[a[0]*m+a[1]] = 1; u1[d[0]*m+d[1]] = 1
        v1[e[0]*p+e[1]] = 1; v1[h[0]*p+h[1]] = 1
        w1[(k0)*n+i0] = 1; w1[(k0)*n+i0+1] = 1
        w1[(k0+1)*n+i0] = 1; w1[(k0+1)*n+i0+1] = 1
        # Wait, this is wrong. Strassen computes products, not sums.
        
        # I'm confusing myself. Let me look up the correct Strassen decomposition.
        
        # Strassen: 
        # M1 = (A11 + A22) * (B11 + B22)
        # M2 = (A21 + A22) * B11
        # M3 = A11 * (B12 - B22)
        # M4 = A22 * (B21 - B11)
        # M5 = (A11 + A12) * B22
        # M6 = (A21 - A11) * (B11 + B12)
        # M7 = (A12 - A22) * (B21 + B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Each M is a product of a linear form in A and a linear form in B.
        # So each M corresponds to one rank-1 term in the tensor decomposition.
        
        # For M1: U has 1 at (i0,j0) and (i0+1,j0+1), V has 1 at (j0,k0) and (j0+1,k0+1)
        # But W needs to distribute to the 4 outputs.
        
        # Actually, in the tensor decomposition, each rank-1 term is:
        # U[i,j,r] * V[j,k,r] * W[k,i,r]
        
        # For M1 = (A11+A22)*(B11+B22), this contributes to all 4 outputs of C.
        # Specifically:
        # C11 += M1, C12 += 0, C21 += 0, C22 += M1 (from the formulas above, M1 appears in C11 and C22)
        
        # Wait, from the formulas:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # So M1 contributes +1 to C11 and +1 to C22.
        
        # In the tensor decomposition, for the term corresponding to M1:
        # U[i,j] = 1 if (i,j) in {(i0,j0), (i0+1,j0+1)}
        # V[j,k] = 1 if (j,k) in {(j0,k0), (j0+1,k0+1)}
        # W[k,i] = 1 if (k,i) in {(k0,i0), (k0+1,i0+1)}  [for C11 and C22]
        # But W also needs to be 0 for C12 and C21.
        
        # This is not a single rank-1 term! M1 contributes to two different outputs.
        
        # I think I'm misunderstanding the tensor decomposition.
        
        # In the tensor decomposition, each rank-1 term is a product of three vectors.
        # The Strassen decomposition has 7 such terms, each contributing to multiple outputs.
        
        # Let me define each of the 7 terms explicitly.
        
        # Term 1 (M1): U1[i,j] = 1 if (i,j) = (i0,j0) or (i0+1,j0+1)
        #              V1[j,k] = 1 if (j,k) = (j0,k0) or (j0+1,k0+1)
        #              W1[k,i] = 1 if (k,i) = (k0,i0) or (k0+1,i0+1)
        # This gives: sum over (i,j) in {(i0,j0),(i0+1,j0+1)} of sum over (j,k) in {(j0,k0),(j0+1,k0+1)} of A[i,j]*B[j,k] * W1[k,i]
        # = (A[i0,j0]+A[i0+1,j0+1]) * (B[j0,k0]+B[j0+1,k0+1]) * (W1[k0,i0] + W1[k0+1,i0+1] + cross terms)
        
        # This is getting too messy. Let me use a different approach.
        
        # I'll just use schoolbook for now and accept rank 84.
        pass
    
    # Just use schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
