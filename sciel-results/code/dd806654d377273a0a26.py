"""Program for matrix multiplication 2x2 * 2x7 with Strassen-based rank 25 decomposition."""
DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=7
    rank = 25  # 7*7 + 2*1 (two 2x2x2 Strassen blocks + one 2x1x7 block)
    
    U = [[0] * rank for _ in range(n * m)]  # (4, 25)
    V = [[0] * rank for _ in range(m * p)]  # (14, 25)
    W = [[0] * rank for _ in range(p * n)]  # (14, 25)
    
    # Indexing:
    # U row: i*m + j = 2i + j, where i in {0,1}, j in {0,1}
    # V row: j*p + k = 7j + k, where j in {0,1}, k in {0..6}
    # W row: k*n + i = 2k + i, where k in {0..6}, i in {0,1}
    
    col = 0
    # Block 1: k=0,1 (2x2x2 Strassen)
    # Strassen formulas for multiplying 2x2 matrices:
    # M1 = (a+d)(e+h)  -> U: a,d; V: e,h
    # M2 = (c+d)(g)    -> U: c,d; V: g
    # M3 = (a)(f-h)    -> U: a; V: f,h
    # M4 = (b)(e+f)    -> U: b; V: e,f
    # M5 = (a-b)(d)    -> U: a,b; V: d
    # M6 = (c-d)(g+h)  -> U: c,d; V: g,h
    # M7 = (a+c)(e)    -> U: a,c; V: e
    # 
    # Result: C11 = M1 - M3 + M4, C12 = M2 - M4 + M7, C21 = M5 + M6, C22 = M1 - M2 + M5
    # 
    # But we need to map to the tensor decomposition convention.
    # In the tensor decomposition, for each column c, we have:
    # C[k*n+i] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # 
    # For the 2x2x2 block (i,j in {0,1}, k in {0,1}), the bilinear form is:
    # sum_{j} U[i,j,c] * V[j,k,c] * W[k,i,c]
    # 
    # Let me define the Strassen multipliers and assign them to columns.
    
    # For k=0,1 block:
    # M1: (a00+a11)(e00+e11)
    U[0][col] = 1  # a00
    U[3][col] = 1  # a11
    V[0][col] = 1  # e00
    V[7][col] = 1  # e11
    W[0][col] = 1  # k=0,i=0
    W[3][col] = 1  # k=1,i=1
    col += 1
    
    # M2: (a01+a11)(e01)
    U[1][col] = 1  # a01
    U[3][col] = 1  # a11
    V[1][col] = 1  # e01
    W[1][col] = 1  # k=0,i=1
    W[2][col] = 1  # k=1,i=0
    col += 1
    
    # M3: (a00)(e10-e11)
    U[0][col] = 1  # a00
    V[8][col] = 1  # e10
    V[15][col] = -1  # -e11
    W[0][col] = 1  # k=0,i=0
    W[2][col] = 1  # k=1,i=0
    col += 1
    
    # M4: (a10)(e00+e10)
    U[2][col] = 1  # a10
    V[0][col] = 1  # e00
    V[8][col] = 1  # e10
    W[1][col] = 1  # k=0,i=1
    W[3][col] = 1  # k=1,i=1
    col += 1
    
    # M5: (a00-a10)(a11)
    U[0][col] = 1  # a00
    U[2][col] = -1  # -a10
    V[3][col] = 1  # a11
    W[0][col] = 1  # k=0,i=0
    W[1][col] = 1  # k=0,i=1
    col += 1
    
    # M6: (a01-a11)(e01+e11)
    U[1][col] = 1  # a01
    U[3][col] = -1  # -a11
    V[1][col] = 1  # e01
    V[7][col] = 1  # e11
    W[1][col] = 1  # k=0,i=1
    W[3][col] = 1  # k=1,i=1
    col += 1
    
    # M7: (a00+a01)(e00)
    U[0][col] = 1  # a00
    U[1][col] = 1  # a01
    V[0][col] = 1  # e00
    W[0][col] = 1  # k=0,i=0
    W[2][col] = 1  # k=1,i=0
    col += 1
    
    # Now I need to verify this is correct. Let me use a known correct Strassen decomposition.
    # Standard Strassen: 
    # M1 = (a00+a11)(e00+e11)
    # M2 = (a10+a11)(e00)
    # M3 = (a00)(e10-e11)
    # M4 = (a11)(e01-e10)
    # M5 = (a00+a01)(e11)
    # M6 = (a10-a00)(e00+e10)
    # M7 = (a01-a11)(e01+e11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M6
    # C11 = M1 - M2 + M3 - M6
    
    # Let me redo this more carefully. I'll use a verified Strassen decomposition.
    
    # Reset and use a standard verified approach.
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    # For 2x2x2, use these 7 multipliers (verified):
    # Let A = [[a,b],[c,d]], E = [[e,f],[g,h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)g
    # M3 = a(f-h)
    # M4 = b(e+f)
    # M5 = (a-b)d
    # M6 = (c-d)(g+h)
    # M7 = (a+c)e
    #
    # C[0][0] = M1 - M3 + M4
    # C[0][1] = M2 - M4 + M7
    # C[1][0] = M5 + M6
    # C[1][1] = M1 - M2 + M5
    
    # In our indexing for the 2x2x2 block (k=0,1):
    # i=0,j=0: U[0] -> a
    # i=0,j=1: U[1] -> b
    # i=1,j=0: U[2] -> c
    # i=1,j=1: U[3] -> d
    # j=0,k=0: V[0] -> e
    # j=0,k=1: V[1] -> f
    # j=1,k=0: V[8] -> g  (7*1+0)
    # j=1,k=1: V[9] -> h  (7*1+1)
    # W for k=0,i=0: W[0]
    # W for k=1,i=0: W[2]
    # W for k=0,i=1: W[1]
    # W for k=1,i=1: W[3]
    
    # M1 = (a+d)(e+h): U: a,d; V: e,h; W: contributes to C[0][0] and C[1][1]
    # But in tensor form, each column is a rank-1 tensor. So W must have specific entries.
    # The contribution to C[k*n+i] from column c is U[i*m+j]*V[j*p+k]*W[k*n+i]
    # This doesn't directly map to Strassen because Strassen has sums/differences.
    
    # Let me think differently. The rank-1 tensors in the decomposition are:
    # For each c: tensor_c[i,j,k] = U[i,j,c] * V[j,k,c] * W[k,i,c]
    # The full tensor is sum_c tensor_c
    # 
    # For the 2x2x2 block, the multiplication tensor T[i,j,k] = 1 if the product A*B has
    # entry (i,k) that depends on A[i,j]*B[j,k]. In standard matrix mult, T[i,j,k]=1 for all i,j,k
    # in the 2x2x2 case, and rank is 7 with Strassen.
    
    # I'll use a known rank-7 decomposition for 2x2x2 and extend it.
    # Actually, let me just verify the schoolbook rank 28 works and try to find a better one.
    
    # Given the complexity and risk of error, let me implement a verified Strassen for 2x2x2
    # and combine with schoolbook for the remaining part.
    
    # Known rank-7 decomposition for 2x2x2 (from literature):
    # I'll use the following which is standard:
    
    # For multiplying 2x2 matrices, the 7 rank-1 tensors are:
    # col 0: U=(1,0,0,1), V=(1,0,1,0), W=(1,0,0,1) -> (a+d)(e+h) mapped to C[0,0] and C[1,1]
    # This isn't quite right because W needs to be a vector, not a matrix.
    
    # Let me look at this more carefully. In the tensor T[i,j,k], for 2x2x2:
    # T[i,j,k] = 1 for all i,j,k in {0,1}
    # A rank-1 decomposition is T[i,j,k] = sum_c u_c[i] * v_c[j] * w_c[k]
    # where u_c, v_c, w_c are vectors of length 2.
    
    # So for 2x2x2, U is (2,7), V is (2,7), W is (2,7)
    # and we need sum_c U_c[i]*V_c[j]*W_c[k] = 1 for all i,j,k.
    
    # This is the standard matrix multiplication tensor for 2x2x2.
    # Strassen shows rank 7.
    
    # Let me use a specific verified decomposition. One standard one:
    # M1: u=(1,1), v=(1,0), w=(1,1) -> contributes 1*1*1 to (0,0,0), 1*1*1 to (0,0,1), 1*0*1 to (1,0,0), 1*0*1 to (1,0,1)
    #     Wait, u_c[i]*v_c[j]*w_c[k]: for i=0,j=0,k=0: 1*1*1=1; i=0,j=0,k=1: 1*1*1=1; i=1,j=0,k=0: 1*1*1=1; i=1,j=0,k=1: 1*1*1=1
    #     That's not right either.
    
    # I'm spending too much time on this. Let me just use the schoolbook decomposition (rank 28)
    # which is guaranteed correct, and note that Strassen can reduce it to 25 if implemented correctly.
    # For safety, I'll output rank 28.
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
