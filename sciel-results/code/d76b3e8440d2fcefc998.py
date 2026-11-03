"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    # U: (n*m, rank) = (16, rank)
    # V: (m*p, rank) = (24, rank)
    # W: (p*n, rank) = (24, rank)
    # W[k*n+i, r] convention: W row index = k*n + i where k is col-block, i is row-block
    
    # We use Strassen for each 2x2x2x2 sub-product
    # A is 4x4, split into 2x2 blocks of 2x2: A[i,k] for i,k in {0,1}
    # B is 4x6, split into 2x3 blocks of 2x2: B[k,j] for k in {0,1}, j in {0,1,2}
    # C is 4x6, split into 2x3 blocks of 2x2: C[i,j] for i in {0,1}, j in {0,1,2}
    # C[i,j] = A[i,0]*B[0,j] + A[i,1]*B[1,j] (2x2 matrix mult)
    
    # Strassen 7 terms for 2x2x2:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1=(a+d)(e+h), M2=(c+d)e, M3=a(f-h), M4=d(g-e), M5=(a+b)h, M6=(c-a)(e+f), M7=(b-d)(g+h)
    # C[0,0]=M1+M4-M5+M7, C[0,1]=M3+M5, C[1,0]=M2+M4, C[1,1]=M1-M2+M3+M6
    
    # For each (i, k, j, s): one rank-1 term
    # i in {0,1}, k in {0,1}, j in {0,1,2}, s in {0..6}
    
    total_rank = 2 * 2 * 3 * 7  # 84
    
    U = [[0] * total_rank for _ in range(n * m)]  # 16 x 84
    V = [[0] * total_rank for _ in range(m * p)]  # 24 x 84
    W = [[0] * total_rank for _ in range(p * n)]  # 24 x 84
    
    col = 0
    for i in range(2):  # row-block of A and C
        for k in range(2):  # col-block of A, row-block of B
            for j in range(3):  # col-block of B and C
                # A[i,k] block: rows 2i, 2i+1; cols 2k, 2k+1
                # B[k,j] block: rows 2k, 2k+1; cols 2j, 2j+1
                # C[i,j] block: rows 2i, 2i+1; cols 2j, 2j+1
                
                # a = A[2i, 2k], b = A[2i, 2k+1], c = A[2i+1, 2k], d = A[2i+1, 2k+1]
                # e = B[2k, 2j], f = B[2k, 2j+1], g = B[2k+1, 2j], h = B[2k+1, 2j+1]
                # p = C[2i, 2j], q = C[2i, 2j+1], r = C[2i+1, 2j], s = C[2i+1, 2j+1]
                
                # W convention: W[k*n + i, r] where k is output col index, i is output row index
                # So W row for C[2i, 2j] is (2j)*n + 2i = (2j)*4 + 2i
                # W row for C[2i, 2j+1] is (2j+1)*4 + 2i
                # W row for C[2i+1, 2j] is (2j)*4 + (2i+1)
                # W row for C[2i+1, 2j+1] is (2j+1)*4 + (2i+1)
                
                wp = (2*j) * n + (2*i)      # C[2i, 2j]
                wq = (2*j+1) * n + (2*i)    # C[2i, 2j+1]
                wr = (2*j) * n + (2*i+1)    # C[2i+1, 2j]
                ws = (2*j+1) * n + (2*i+1)  # C[2i+1, 2j+1]
                
                # U row for A[2i, 2k] = 2i*m + 2k = 2i*4 + 2k
                # U row for A[2i, 2k+1] = 2i*4 + 2k+1
                # U row for A[2i+1, 2k] = (2i+1)*4 + 2k
                # U row for A[2i+1, 2k+1] = (2i+1)*4 + 2k+1
                
                ua = 2*i * m + 2*k
                ub = 2*i * m + 2*k + 1
                uc = (2*i+1) * m + 2*k
                ud = (2*i+1) * m + 2*k + 1
                
                # V row for B[2k, 2j] = 2k*p + 2j = 2k*6 + 2j
                # V row for B[2k, 2j+1] = 2k*6 + 2j+1
                # V row for B[2k+1, 2j] = (2k+1)*6 + 2j
                # V row for B[2k+1, 2j+1] = (2k+1)*6 + 2j+1
                
                ve = 2*k * p + 2*j
                vf = 2*k * p + 2*j + 1
                vg = (2*k+1) * p + 2*j
                vh = (2*k+1) * p + 2*j + 1
                
                # Strassen terms
                # M1: (a+d)(e+h)
                U[ua][col] += 1; U[ud][col] += 1
                V[ve][col] += 1; V[vh][col] += 1
                W[wp][col] += 1; W[ws][col] += 1
                col += 1
                
                # M2: (c+d)e
                U[uc][col] += 1; U[ud][col] += 1
                V[ve][col] += 1
                W[wr][col] += 1
                col += 1
                
                # M3: a(f-h)
                U[ua][col] += 1
                V[vf][col] += 1; V[vh][col] -= 1
                W[wq][col] += 1; W[ws][col] += 1
                col += 1
                
                # M4: d(g-e)
                U[ud][col] += 1
                V[vg][col] += 1; V[ve][col] -= 1
                W[wp][col] += 1; W[wr][col] += 1
                col += 1
                
                # M5: (a+b)h
                U[ua][col] += 1; U[ub][col] += 1
                V[vh][col] += 1
                W[wp][col] -= 1; W[wq][col] += 1
                col += 1
                
                # M6: (c-a)(e+f)
                U[uc][col] += 1; U[ua][col] -= 1
                V[ve][col] += 1; V[vf][col] += 1
                W[ws][col] += 1
                col += 1
                
                # M7: (b-d)(g+h)
                U[ub][col] += 1; U[ud][col] -= 1
                V[vg][col] += 1; V[vh][col] += 1
                W[wp][col] += 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
