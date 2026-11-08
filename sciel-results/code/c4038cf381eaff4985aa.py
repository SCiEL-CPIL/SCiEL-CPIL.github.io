import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # We use two blocks of 2x2x2 Strassen decompositions.
    # Total rank = 7 + 7 = 14.
    rank = 14
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define Strassen decomposition for 2x2x2
    # Indices i, j, k in {0, 1}
    # Condition: sum_r W[k,i,r] * U[i,j,r] * V[j,k,r] = 1 if i,j,k valid, else 0?
    # Actually, the condition is:
    # sum_r W[k,i,r] U[a,b,r] V[c,d,r] = delta(a,i) delta(d,k) delta(b,c)
    
    # Strassen Products:
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    
    # Output reconstruction:
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # We define U, V, W for one block (indices 0,1)
    # r is 0..6
    # U[i,j,r], V[j,k,r], W[k,i,r]
    
    U_block = [[0]*7 for _ in range(2*2)] # i*2+j
    V_block = [[0]*7 for _ in range(2*2)] # j*2+k
    W_block = [[0]*7 for _ in range(2*2)] # k*2+i
    
    # P1 = (a+d)(e+h)
    # U: a (0,0), d (1,1)
    # V: e (0,0), h (1,1)
    # W: 1 for all k,i? No.
    # W[k,i] contributes to C[i,k].
    # C00 gets P1. C11 gets P1.
    # So W[0,0]=1, W[1,1]=1.
    U_block[0*2+0][0] = 1 # a
    U_block[1*2+1][0] = 1 # d
    V_block[0*2+0][0] = 1 # e
    V_block[1*2+1][0] = 1 # h
    W_block[0*2+0][0] = 1 # C00
    W_block[1*2+1][0] = 1 # C11
    
    # P2 = (c+d)e
    # U: c (1,0), d (1,1)
    # V: e (0,0)
    # W: C10 gets P2.
    U_block[1*2+0][1] = 1 # c
    U_block[1*2+1][1] = 1 # d
    V_block[0*2+0][1] = 1 # e
    W_block[0*2+1][1] = 1 # C10 (k=0, i=1)
    
    # P3 = a(f-h)
    # U: a (0,0)
    # V: f (0,1), -h (1,1)
    # W: C01 gets P3, C11 gets P3.
    U_block[0*2+0][2] = 1 # a
    V_block[0*2+1][2] = 1 # f
    V_block[1*2+1][2] = -1 # -h
    W_block[1*2+0][2] = 1 # C01 (k=1, i=0)
    W_block[1*2+1][2] = 1 # C11 (k=1, i=1)
    
    # P4 = d(g-e)
    # U: d (1,1)
    # V: g (1,0), -e (0,0)
    # W: C00 gets P4, C10 gets P4.
    U_block[1*2+1][3] = 1 # d
    V_block[1*2+0][3] = 1 # g
    V_block[0*2+0][3] = -1 # -e
    W_block[0*2+0][3] = 1 # C00
    W_block[0*2+1][3] = 1 # C10
    
    # P5 = (a+b)h
    # U: a (0,0), b (0,1)
    # V: h (1,1)
    # W: C00 gets -P5, C01 gets P5.
    U_block[0*2+0][4] = 1 # a
    U_block[0*2+1][4] = 1 # b
    V_block[1*2+1][4] = 1 # h
    W_block[0*2+0][4] = -1 # C00
    W_block[1*2+0][4] = 1 # C01
    
    # P6 = (c-a)(e+f)
    # U: c (1,0), -a (0,0)
    # V: e (0,0), f (0,1)
    # W: C11 gets P6.
    U_block[1*2+0][5] = 1 # c
    U_block[0*2+0][5] = -1 # -a
    V_block[0*2+0][5] = 1 # e
    V_block[0*2+1][5] = 1 # f
    W_block[1*2+1][5] = 1 # C11
    
    # P7 = (b-d)(g+h)
    # U: b (0,1), -d (1,1)
    # V: g (1,0), h (1,1)
    # W: C00 gets P7.
    U_block[0*2+1][6] = 1 # b
    U_block[1*2+1][6] = -1 # -d
    V_block[1*2+0][6] = 1 # g
    V_block[1*2+1][6] = 1 # h
    W_block[0*2+0][6] = 1 # C00

    # Now map to full matrices
    # Block 1: j in {0, 1}
    # Block 2: j in {2, 3}
    
    for i in range(n):
        for j in range(m):
            # U index: i*m + j
            row_u = i * m + j
            if j < 2:
                # Block 1
                local_j = j
                for r in range(7):
                    U[row_u][r] = U_block[i*2 + local_j][r]
            else:
                # Block 2
                local_j = j - 2
                for r in range(7):
                    U[row_u][r + 7] = U_block[i*2 + local_j][r]
                    
    for j in range(m):
        for k in range(p):
            # V index: j*p + k
            row_v = j * p + k
            if j < 2:
                # Block 1
                local_j = j
                for r in range(7):
                    V[row_v][r] = V_block[local_j*2 + k][r] # k is 0..1 for 2x2? 
                    # Wait, p=7. k is 0..6.
                    # But B is 4x7.
                    # My Strassen block is for 2x2 B.
                    # Here B is 4x7.
                    # We split A (2x4) into two 2x2.
                    # B is 4x7.
                    # B1 is 2x7, B2 is 2x7.
                    # Strassen works for 2x2 * 2x2.
                    # Here we have 2x2 * 2x7.
                    # We can treat 2x7 as 3.5 blocks of 2x2? No.
                    # We can just use the linearity.
                    # The rank of 2x2 * 2xp is 7 * (p/2)?
                    # Actually, R(2,2,p) = 7 * ceil(p/2)?
                    # For p=7, we can split into 2x2, 2x2, 2x1.
                    # R(2,2,2)=7. R(2,2,1)=2? (Schoolbook 2, Strassen not applicable for 1)
                    # So R(2,2,7) = 7 + 7 + 2 = 16?
                    # Then total rank = 16 + 16 = 32?
                    # This is worse than 56? No, 32 < 56.
                    # But can we do better?
                    
                    # Let's re-evaluate R(2,4,7).
                    # A is 2x4. B is 4x7.
                    # We can view this as 2x2 * 2x7 + 2x2 * 2x7.
                    # R(2,2,7) = R(2,2,2) + R(2,2,2) + R(2,2,1) = 7 + 7 + 2 = 16.
                    # Total rank = 16 + 16 = 32.
                    
                    # Is there a better way?
                    # Maybe R(2,4,7) < 32?
                    # R(2,4,4) = 2 * R(2,2,2) = 14.
                    # R(2,4,7) = R(2,4,4) + R(2,4,3)?
                    # R(2,4,3) = R(2,2,3) + R(2,2,1)?
                    # R(2,2,3) = 7 + 2 = 9.
                    # R(2,4,3) = 9 + 9 = 18?
                    # Total 14 + 18 = 32.
                    
                    # It seems 32 is a solid upper bound.
                    # Let's implement rank 32.
                    
                    pass
            else:
                local_j = j - 2
                for r in range(7):
                    V[row_v][r + 7] = V_block[local_j*2 + k][r]
                    
    # This approach is getting complicated with the dimensions.
    # Let's stick to the simplest improvement: Rank 32.
    
    return {"U": U, "V": V, "W": W}

# The above code is a sketch. I will write the full rank 32 implementation.
