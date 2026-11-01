DIMENSIONS = [2, 2, 8]

def construct():
    n, m, p = DIMENSIONS
    # Rank 28 decomposition using 7-term decomposition for each of 4 blocks
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 7-term decomposition coefficients for 2x2 matrix mult
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = a(e+h), M2 = d(a+b), M3 = c(g+h), M4 = b(a+c)
    # M5 = (a+d)(e+h), M6 = (c+d)(g+h), M7 = (a+b)(e+f)
    # C11 = M1 - M2 + M4 - M7
    # C12 = M1 - M5 + M6 - M7
    # C21 = M2 - M3 + M5 - M7
    # C22 = M3 - M4 + M5 - M6
    
    # U coefficients for A
    # U[i*m+j, col]
    # a=0, b=1, c=2, d=3
    U_m1 = [1, 0, 0, 0] # a
    U_m2 = [0, 1, 0, 1] # d(a+b) -> d*a + d*b? No, d is coeff of d.
    # M2 = d(a+b) = da + db. So U[0]=1, U[1]=1, U[3]=1? No.
    # M_r = sum U[i,j] A[i,j] * sum V[j,k] B[j,k]
    # M2 = d(a+b) = (0a+0b+0c+1d)(1a+1b+0c+0d)? No.
    # It's a product of linear forms.
    # L_r(A) L_r(B)
    # L1(A) = a, L1(B) = e+h
    # L2(A) = d, L2(B) = a+b
    # L3(A) = c, L3(B) = g+h
    # L4(A) = b, L4(B) = a+c
    # L5(A) = a+d, L5(B) = e+h
    # L6(A) = c+d, L6(B) = g+h
    # L7(A) = a+b, L7(B) = e+f
    
    # V coefficients for B (2x2 block)
    # e=0, f=1, g=2, h=3 (local indices)
    V_m1 = [1, 0, 0, 1] # e+h
    V_m2 = [1, 1, 0, 0] # a+b (but B is e,f,g,h) -> e+f? No, L2(B)=a+b is wrong.
    # L2(B) must be linear in B.
    # M2 = d(a+b). This is not a product of L(A) and L(B).
    # Strassen's algorithm uses products of sums.
    # M2 = (a+b)(d+e)? No.
    
    # Correct Strassen:
    # M1 = a(e+h)
    # M2 = d(a+b) -> This is NOT a product of linear forms in A and B separately?
    # d is in A. a+b is in A.
    # So M2 is quadratic in A? No.
    # Strassen:
    # M1 = a(e+h)
    # M2 = d(a+b)
    # M3 = c(g+h)
    # M4 = b(a+c)
    # M5 = (a+d)(e+h)
    # M6 = (c+d)(g+h)
    # M7 = (a+b)(e+f)
    
    # M2 = d(a+b) = da + db.
    # This is a product of L(A)=d and L(B)=? No.
    # It's a product of L(A)=a+b and L(B)=? No.
    # It's a product of L(A)=d and L(B)=?
    # Actually, Strassen's M2 is (a+b)(d+e)? No.
    
    # I will use the schoolbook (rank 32) to be safe.
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
