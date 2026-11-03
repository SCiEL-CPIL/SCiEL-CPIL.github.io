"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6
    # Rank 33 decomposition:
    # Split j into 0,1 (use 2x2x6 algo) and 2 (use 2x1x6 algo)
    # 2x2x6 = 3 * 2x2x2. Rank 3*7 = 21.
    # 2x1x6 = 6 * 2x1x1. Rank 12.
    # Total 33.
    
    rank = 33
    U = [[0] * rank for _ in range(n * m)]  # 6 x 33
    V = [[0] * rank for _ in range(m * p)]  # 18 x 33
    W = [[0] * rank for _ in range(p * n)]  # 12 x 33
    
    col = 0
    # Part 1: j in {0, 1}, k in {0..5}
    # Split k into 3 blocks of 2: k0={0,1}, k1={2,3}, k2={4,5}
    for k_block in range(3):
        k_offset = k_block * 2
        # 2x2x2 Strassen for A[:, 0:2], B[0:2, k_offset:k_offset+2]
        # Strassen terms for 2x2x2:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B10
        # M3 = A00(B01-B10)
        # M4 = A11(B11-B00)
        # M5 = (A00+A01)B00
        # M6 = (A10+A11)(B01-B11) -> Wait, standard Strassen
        # Let's use a standard 7-term Strassen for 2x2 matrices.
        # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        # M1 = (a+d)(e+h)
        # M2 = (c+d)h
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)e
        # M6 = (c+d)(f-g)
        # M7 = (b-d)(g+h)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Indices in U: i*m + j. j is 0 or 1.
        # Indices in V: j*p + k. j is 0 or 1. k is local 0 or 1.
        # Indices in W: k*n + i. k is local 0 or 1. i is 0 or 1.
        
        terms = [
            # (u_indices, v_indices, w_coefficients)
            # M1
            ([0, 3], [0, 1], {(0,0):1, (0,1):-1, (1,0):1, (1,1):1}), # (a+d)(e+h)
            # M2
            ([3], [1], {(1,0):1, (1,1):-1}), # (c+d)h -> c=U3, d=U3? No.
            # Let's map carefully.
            # U rows: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) for j=0,1?
            # No, U index is i*m+j.
            # j=0: i=0->0, i=1->2.
            # j=1: i=0->1, i=1->3.
            # So:
            # A00=U0, A01=U1, A10=U2, A11=U3.
            # V rows: j*p+k.
            # j=0: k=0->0, k=1->1.
            # j=1: k=0->p, k=1->p+1.
            # So:
            # B00=V0, B01=V1, B10=Vp, B11=V(p+1).
            # W rows: k*n+i.
            # k=0: i=0->0, i=1->1.
            # k=1: i=0->2, i=1->3.
            # So:
            # C00=W0, C01=W1, C10=W2, C11=W3.
            
            # M1 = (A00+A11)(B00+B11)
            # U: 0, 3. V: 0, p.
            # W: C00+=M1, C10-=M1? No.
            # C00 = M1 - M5 + M7 + M4
            # C10 = M1 + M2 + M4
            # C01 = M3 + M5
            # C11 = M1 - M2 - M3 + M6
            
            # Let's just define the 7 terms and add them to W.
        ]
        
        # Define Strassen terms for local 2x2x2
        # M1: (A00+A11)(B00+B11)
        U[0][col] += 1; U[3][col] += 1
        V[0][col] += 1; V[p+k_offset][col] += 1
        # W contributions for M1: C00+=1, C10+=1, C01? No.
        # C00 = M1 + M4 - M5 + M7
        # C10 = M1 + M2 + M4
        # C01 = M3 + M5
        # C11 = M1 - M2 - M3 + M6
        W[0][col] += 1
        W[2][col] += 1
        W[3][col] += 1 # C11 gets M1? Yes.
        col += 1

        # M2: (A10+A11)B10
        U[2][col] += 1; U[3][col] += 1
        V[p+k_offset][col] += 1
        W[2][col] += 1 # C10 += M2
        W[3][col] -= 1 # C11 -= M2
        col += 1

        # M3: A00(B01-B10)
        U[0][col] += 1
        V[1+k_offset][col] += 1; V[p+k_offset][col] -= 1
        W[1][col] += 1 # C01 += M3
        W[3][col] -= 1 # C11 -= M3
        col += 1

        # M4: A11(B11-B00)
        U[3][col] += 1
        V[p+1+k_offset][col] += 1; V[0+k_offset][col] -= 1
        W[0][col] += 1 # C00 += M4
        W[2][col] += 1 # C10 += M4
        col += 1

        # M5: (A00+A01)B00
        U[0][col] += 1; U[1][col] += 1
        V[0+k_offset][col] += 1
        W[0][col] -= 1 # C00 -= M5
        W[1][col] += 1 # C01 += M5
        col += 1

        # M6: (A10+A11)(B01-B11)
        U[2][col] += 1; U[3][col] += 1
        V[1+k_offset][col] += 1; V[p+1+k_offset][col] -= 1
        W[3][col] += 1 # C11 += M6
        col += 1

        # M7: (A01-A11)(B10+B11)
        U[1][col] += 1; U[3][col] -= 1
        V[p+k_offset][col] += 1; V[p+1+k_offset][col] += 1
        W[0][col] += 1 # C00 += M7
        col += 1

    # Part 2: j=2, k in {0..5}
    # 2x1x6. Rank 12.
    # C_i,k = A_i2 B_2k.
    # Terms: A_i2 * B_2k.
    # 2 * 6 = 12 terms.
    for i in range(2):
        for k in range(6):
            U[i*m + 2][col] = 1
            V[2*p + k][col] = 1
            W[k*n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
