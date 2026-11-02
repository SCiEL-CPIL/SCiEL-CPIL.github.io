"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 16
    rank = 56  # 8 pairs * 7 Strassen terms
    
    U = [[0] * rank for _ in range(n * m)]  # (4, 56)
    V = [[0] * rank for _ in range(m * p)]  # (32, 56)
    W = [[0] * rank for _ in range(p * n)]  # (32, 56)
    
    # Strassen 7-term decomposition for 2x2 matrix multiplication
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1=(a+d)(e+h), M2=(c+d)e, M3=a(f-h), M4=d(g-e), M5=(a+b)h, M6=(c-a)(e+g), M7=(b-d)(f+h)
    # C[0,0]=M1+M4-M5+M7, C[0,1]=M3+M5, C[1,0]=M2+M4, C[1,1]=M1-M2+M3+M6
    
    # U coefficients for each Strassen term (a,b,c,d)
    U_coeffs = [
        [1, 0, 0, 1],   # M1: a+d
        [0, 0, 1, 1],   # M2: c+d
        [1, 0, 0, 0],   # M3: a
        [0, 0, 0, 1],   # M4: d
        [1, 1, 0, 0],   # M5: a+b
        [0, -1, 1, 0],  # M6: c-a
        [0, 1, 0, -1],  # M7: b-d
    ]
    
    # V coefficients for each Strassen term (e,f,g,h) where e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
    V_coeffs = [
        [1, 1, 0, 0],   # M1: e+h
        [1, 0, 0, 0],   # M2: e
        [0, 1, 0, -1],  # M3: f-h
        [0, 0, 1, -1],  # M4: g-e
        [0, 0, 0, 1],   # M5: h
        [1, 0, 1, 0],   # M6: e+g
        [0, 1, 0, 1],   # M7: f+h
    ]
    
    # W contributions: for each term, which C entries it affects and with what coefficient
    # C[0,0]=M1+M4-M5+M7, C[0,1]=M3+M5, C[1,0]=M2+M4, C[1,1]=M1-M2+M3+M6
    # W index: k*n + i = 2*k + i
    W_contribs = [
        [(0, 1), (3, 1)],      # M1: C[0,0]+=1, C[1,1]+=1
        [(1, -1), (2, 1)],     # M2: C[0,1]+=-1, C[1,0]+=1
        [(1, 1), (3, 1)],      # M3: C[0,1]+=1, C[1,1]+=1
        [(0, 1), (2, 1)],      # M4: C[0,0]+=1, C[1,0]+=1
        [(0, -1), (1, 1)],     # M5: C[0,0]+=-1, C[0,1]+=1
        [(3, 1)],              # M6: C[1,1]+=1
        [(0, 1)],              # M7: C[0,0]+=1
    ]
    
    num_pairs = p // 2  # 8 pairs
    
    for t in range(num_pairs):
        col0 = 2 * t
        col1 = 2 * t + 1
        for s in range(7):
            col = t * 7 + s
            
            # Fill U
            for ij in range(4):
                U[ij][col] = U_coeffs[s][ij]
            
            # Fill V: B[j, k] is at row j*p + k = j*16 + k
            # e=B[0,col0], f=B[0,col1], g=B[1,col0], h=B[1,col1]
            V[0 * p + col0][col] = V_coeffs[s][0]  # e
            V[0 * p + col1][col] = V_coeffs[s][1]  # f
            V[1 * p + col0][col] = V_coeffs[s][2]  # g
            V[1 * p + col1][col] = V_coeffs[s][3]  # h
            
            # Fill W: C[i, k] is at W[k*n + i] = W[2*k + i]
            for (ci, coeff) in W_contribs[s]:
                # ci encodes (i, k_offset) where k = col0 + k_offset
                i = ci // 2
                k_offset = ci % 2
                k = col0 + k_offset
                W[k * n + i][col] += coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
