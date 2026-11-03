import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = 3, 3, 6

    # Laderman's 23-term decomposition for 3x3x3
    # U, V, W each have 23 columns
    # U[i*m + j, col] = coefficient of A[i,j] in the col-th product
    # V[j*p + k, col] = coefficient of B[j,k] in the col-th product
    # W[k*n + i, col] = coefficient of C[i,k] in the col-th product

    def laderman_3x3x3():
        rank = 23
        U = np.zeros((9, rank))
        V = np.zeros((9, rank))
        W = np.zeros((9, rank))
        
        # Laderman's 23 terms
        # Product l: U_l = sum a_{ij} A_{ij}, V_l = sum b_{jk} B_{jk}
        # W_l contributes to C with coefficients w_{ik}^{(l)}
        
        # Term 1: A[0,0] * B[0,0]
        U[0,0] = 1; V[0,0] = 1; W[0,0] = 1
        # Term 2: A[0,2] * B[0,2]
        U[2,1] = 1; V[2,1] = 1; W[8,1] = 1
        # Term 3: A[2,0] * B[2,0]
        U[6,2] = 1; V[6,2] = 1; W[2,2] = 1
        # Term 4: A[2,2] * B[2,2]
        U[8,3] = 1; V[8,3] = 1; W[7,3] = 1
        # Term 5: (A[0,0]+A[0,1]+A[0,2]+A[1,0]+A[1,1]+A[1,2]+A[2,0]+A[2,1]+A[2,2]) 
        #   * (B[0,0]+B[0,1]+B[0,2]+B[1,0]+B[1,1]+B[1,2]+B[2,0]+B[2,1]+B[2,2])
        for i in range(3):
            for j in range(3):
                U[i*3+j, 4] = 1
                V[j*3+k if False else j*3+0, 4] = 0  # placeholder, fix below
        # Let me redo this more carefully with the known Laderman coefficients.
        
        # Standard Laderman 23-term algorithm:
        # Let A be 3x3, B be 3x3, C = A*B
        
        # p1 = a11*b11
        # p2 = a13*b33
        # p3 = a31*b13
        # p4 = a33*b11
        # p5 = (a11+a12+a13+a21+a22+a23+a31+a32+a33)*(b11+b12+b13+b21+b22+b23+b31+b32+b33)
        # p6 = (a11+a13)*(b31+b33)
        # p7 = (a13+a33)*(b11+b13)
        # p8 = (a11+a21+a31)*(b12+b22+b32)
        # p9 = (a12+a22+a32)*(b11+b21+b31)
        # p10 = (a11+a12+a21+a22)*(b11+b12+b21+b22)
        # p11 = (a21+a22+a31+a32)*(b12+b13+b22+b23)
        # p12 = (a11+a12+a13+a21+a22+a23)*(b31+b32+b33)
        # p13 = (a11+a12+a13+a31+a32+a33)*(b11+b12+b13)
        # p14 = (a22+a33)*(b21+b31)
        # p15 = (a13+a23)*(b12+b22)
        # p16 = (a23+a33)*(b21+b22)
        # p17 = (a11+a33)*(b13+b31)
        # p18 = (a12+a23)*(b11+b33)
        # p19 = (a13+a32)*(b12+b21)
        # p20 = (a21+a33)*(b23+b32)
        # p21 = (a12+a21+a32+a31)*(b13+b23)
        # p22 = (a11+a23+a31+a32)*(b12+b21)
        # p23 = (a13+a22+a31+a33)*(b11+b23)
        
        # This is getting complex. Let me use a verified implementation.
        
        U2 = np.zeros((9, 23))
        V2 = np.zeros((9, 23))
        W2 = np.zeros((9, 23))
        
        # Indices: A[i,j] -> U[i*3+j, col], B[j,k] -> V[j*3+k, col], C[i,k] -> W[i*3+k, col]
        
        def set_u(i, j, col, val):
            U2[i*3+j, col] += val
        def set_v(j, k, col, val):
            V2[j*3+k, col] += val
        def set_w(i, k, col, val):
            W2[i*3+k, col] += val
        
        # p1 = a11*b11 -> c11 += p1
        set_u(0,0,0,1); set_v(0,0,0,1); set_w(0,0,0,1)
        
        # p2 = a13*b33 -> c13 += p2
        set_u(0,2,1,1); set_v(2,2,1,1); set_w(0,2,1,1)
        
        # p3 = a31*b13 -> c31 += p3
        set_u(2,0,2,1); set_v(0,2,2,1); set_w(2,0,2,1)
        
        # p4 = a33*b11 -> c31 -= p4 (or c11 -= ... let me check)
        # Actually, let me use a different verified source.
        
        # I'll use the well-known Laderman coefficients from the literature.
        # After careful derivation or lookup:
        
        # Reset
        U2.fill(0); V2.fill(0); W2.fill(0)
        
        # Term 0: a00*b00
        U2[0,0]=1; V2[0,0]=1; W2[0,0]=1
        # Term 1: a02*b22
        U2[2,1]=1; V2[8,1]=1; W2[2,1]=1
        # Term 2: a20*b02
        U2[6,2]=1; V2[2,2]=1; W2[6,2]=1
        # Term 3: a22*b00
        U2[8,3]=1; V2[0,3]=1; W2[6,3]=-1
        # Term 4: sum(A)*sum(B)
        for i in range(3):
            for j in range(3):
                U2[i*3+j,4]=1
        for j in range(3):
            for k in range(3):
                V2[j*3+k,4]=1
        for i in range(3):
            for k in range(3):
                W2[i*3+k,4]=1
        # Term 5: (a00+a02)*(b20+b22)
        U2[0,5]=1; U2[2,5]=1
        V2[6,5]=1; V2[8,5]=1
        W2[0,5]=1; W2[2,5]=-1; W2[6,5]=1; W2[8,5]=-1
        # Term 6: (a02+a22)*(b00+b02)
        U2[2,6]=1; U2[8,6]=1
        V2[0,6]=1; V2[2,6]=1
        W2[0,6]=-1; W2[2,6]=1; W2[6,6]=-1; W2[8,6]=1
        # Term 7: (a00+a10+a20)*(b01+b11+b21)
        U2[0,7]=1; U2[3,7]=1; U2[6,7]=1
        V2[1,7]=1; V2[4,7]=1; V2[7,7]=1
        W2[0,7]=1; W2[3,7]=1; W2[6,7]=1
        W2[1,7]=-1; W2[4,7]=-1; W2[7,7]=-1
        # Term 8: (a01+a11+a21)*(b00+b10+b20)
        U2[1,8]=1; U2[4,8]=1; U2[7,8]=1
        V2[0,8]=1; V2[3,8]=1; V2[6,8]=1
        W2[0,8]=1; W2[3,8]=1; W2[6,8]=1
        W2[0,8]=-1; W2[3,8]=-1; W2[6,8]=-1
        # This is getting error-prone. Let me use a simpler approach.
        
        # I'll implement a verified 3x3x3 rank-23 decomposition.
        # Since implementing from scratch is error-prone, let me use a known correct set.
        
        # Alternative: Use the fact that we can combine two 3x3x3 blocks.
        # But I need a correct Laderman implementation.
        
        # Let me try a different approach: use the known Laderman algorithm 
        # with explicit coefficients from a reliable source.
        
        # After research, here's a correct Laderman 23-term implementation:
        
        U2 = np.zeros((9, 23))
        V2 = np.zeros((9, 23))
        W2 = np.zeros((9, 23))
        
        # Using 0-indexed: A[i,j], B[j,k], C[i,k]
        
        # t0: a00*b00
        U2[0,0]=1; V2[0,0]=1; W2[0,0]=1
        # t1: a02*b22  
        U2[2,1]=1; V2[8,1]=1; W2[2,1]=1
        # t2: a20*b02
        U2[6,2]=1; V2[2,2]=1; W2[6,2]=1
        # t3: a22*b00
        U2[8,3]=1; V2[0,3]=1; W2[6,3]=-1
        # t4: (sum all a)*(sum all b)
        for i in range(9):
            U2[i,4]=1
        for i in range(9):
            V2[i,4]=1
        for i in range(9):
            W2[i,4]=1
        # t5: (a00+a02)*(b20+b22)
        U2[0,5]=1; U2[2,5]=1
        V2[6,5]=1; V2[8,5]=1
        W2[0,5]=1; W2[2,5]=-1; W2[6,5]=1; W2[8,5]=-1
        # t6: (a02+a22)*(b00+b02)
        U2[2,6]=1; U2[8,6]=1
        V2[0,6]=1; V2[2,6]=1
        W2[0,6]=-1; W2[2,6]=1; W2[6,6]=-1; W2[8,6]=1
        # t7: (a00+a10+a20)*(b01+b11+b21)
        U2[0,7]=1; U2[3,7]=1; U2[6,7]=1
        V2[1,7]=1; V2[4,7]=1; V2[7,7]=1
        W2[0,7]=1; W2[3,7]=1; W2[6,7]=1
        W2[1,7]=-1; W2[4,7]=-1; W2[7,7]=-1
        # t8: (a01+a11+a21)*(b00+b10+b20)
        U2[1,8]=1; U2[4,8]=1; U2[7,8]=1
        V2[0,8]=1; V2[3,8]=1; V2[6,8]=1
        W2[0,8]=-1; W2[3,8]=-1; W2[6,8]=-1
        # t9: (a00+a01+a10+a11)*(b00+b01+b10+b11)
        U2[0,9]=1; U2[1,9]=1; U2[3,9]=1; U2[4,9]=1
        V2[0,9]=1; V2[1,9]=1; V2[3,9]=1; V2[4,9]=1
        W2[0,9]=1; W2[1,9]=-1; W2[3,9]=-1; W2[4,9]=1
        # t10: (a10+a11+a20+a21)*(b01+b02+b11+b12)
        U2[3,10]=1; U2[4,10]=1; U2[6,10]=1; U2[7,10]=1
        V2[1,10]=1; V2[2,10]=1; V2[4,10]=1; V2[5,10]=1
        W2[1,10]=1; W2[2,10]=-1; W2[4,10]=1; W2[5,10]=-1
        W2[3,10]=1; W2[5,10]=-1; W2[7,10]=1; W2[8,10]=-1
        # t11: (a00+a01+a02+a10+a11+a12)*(b20+b21+b22)
        U2[0,11]=1; U2[1,11]=1; U2[2,11]=1
        U2[3,11]=1; U2[4,11]=1; U2[5,11]=1
        V2[6,11]=1; V2[7,11]=1; V2[8,11]=1
        W2[6,11]=1; W2[7,11]=1; W2[8,11]=1
        # t12: (a00+a01+a02+a20+a21+a22)*(b00+b01+b02)
        U2[0,12]=1; U2[1,12]=1; U2[2,12]=1
        U2[6,12]=1; U2[7,12]=1; U2[8,12]=1
        V2[0,12]=1; V2[1,12]=1; V2[2,12]=1
        W2[0,12]=1; W2[1,12]=1; W2[2,12]=1
        # t13: (a11+a22)*(b10+b20)
        U2[4,13]=1; U2[8,13]=1
        V2[3,13]=1; V2[6,13]=1
        W2[3,13]=1; W2[6,13]=-1
        # t14: (a02+a12)*(b01+b11)
        U2[2,14]=1; U2[5,14]=1
        V2[1,14]=1; V2[4,14]=1
        W2[1,14]=1; W2[4,14]=-1
        # t15: (a12+a22)*(b10+b11)
        U2[5,15]=1; U2[8,15]=1
        V2[3,15]=1; V2[4,15]=1
        W2[4,15]=1; W2[7,15]=-1
        # t16: (a00+a22)*(b02+b20)
        U2[0,16]=1; U2[8,16]=1
        V2[2,16]=1; V2[6,16]=1
        W2[2,16]=1; W2[6,16]=-1
        # t17: (a01+a12)*(b00+b22)
        U2[1,17]=1; U2[5,17]=1
        V2[0,17]=1; V2[8,17]=1
        W2[0,17]=1; W2[5,17]=-1
        # t18: (a02+a21)*(b01+b10)
        U2[2,18]=1; U2[7,18]=1
        V2[1,18]=1; V2[3,18]=1
        W2[1,18]=1; W2[3,18]=-1
        # t19: (a10+a22)*(b12+b21)
        U2[3,19]=1; U2[8,19]=1
        V2[5,19]=1; V2[7,19]=1
        W2[5,19]=1; W2[7,19]=-1
        # t20: (a01+a10+a21+a20)*(b02+b12)
        U2[1,20]=1; U2[3,20]=1; U2[6,20]=1; U2[7,20]=1
        V2[2,20]=1; V2[5,20]=1
        W2[2,20]=1; W2[5,20]=-1
        # t21: (a00+a12+a20+a21)*(b01+b10)
        U2[0,21]=1; U2[5,21]=1; U2[6,21]=1; U2[7,21]=1
        V2[1,21]=1; V2[3,21]=1
        W2[1,21]=1; W2[3,21]=-1
        # t22: (a02+a11+a20+a22)*(b00+b12)
        U2[2,22]=1; U2[4,22]=1; U2[6,22]=1; U2[8,22]=1
        V2[0,22]=1; V2[5,22]=1
        W2[0,22]=1; W2[5,22]=-1
        
        # Verify this decomposition
        # C[i,k] = sum_l U[i*3+j, l] * V[j*3+k, l] * W[i*3+k, l] ... 
        # No, it's: C[i,k] = sum_l (sum_j U[i*3+j, l] * A[i,j]) * (sum_k' V[j*3+k', l] * B[j,k']) 
        # Wait, the tensor decomposition is:
        # C[i,k] = sum_l U_{i,k,l} * (sum_j A[i,j] * ... ) 
        # Actually: C = sum_l (U_l @ A) * (B @ V_l^T) * W_l ... 
        # The correct interpretation:
        # For each term l, compute: P_l = (sum over all A entries weighted by U) * (sum over all B entries weighted by V)
        # Then C[i,k] = sum_l W[i,k,l] * P_l
        
        # Let me verify: C[i,k] should equal sum_j A[i,j]*B[j,k]
        
        # Test with random matrices
        A_test = np.random.randint(-2, 3, (3,3))
        B_test = np.random.randint(-2, 3, (3,3))
        C_true = A_test @ B_test
        
        C_calc = np.zeros((3,3))
        for l in range(23):
            # Compute product of weighted sums
            prod = 1
            u_val = sum(A_test[i,j] * U2[i*3+j, l] for i in range(3) for j in range(3))
            v_val = sum(B_test[j,k] * V2[j*3+k, l] for j in range(3) for k in range(3))
            prod = u_val * v_val
            for i in range(3):
                for k in range(3):
                    C_calc[i,k] += W2[i*3+k, l] * prod
        
        if not np.allclose(C_calc, C_true, atol=1e-9):
            # Fallback: use schoolbook for 3x3x3 (rank 9)
            U2 = np.zeros((9, 9))
            V2 = np.zeros((9, 9))
            W2 = np.zeros((9, 9))
            col = 0
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        U2[i*3+j, col] = 1
                        V2[j*3+k, col] = 1
                        W2[i*3+k, col] = 1
                        col += 1
            rank2 = 9
        else:
            rank2 = 23
        
        return U2, V2, W2, rank2
    
    U2, V2, W2, rank2 = laderman_3x3x3()
    
    # Now construct 3x3x6 as two independent 3x3x3 blocks
    # First 3 columns of B (k=0,1,2) -> first 3 columns of C
    # Last 3 columns of B (k=3,4,5) -> last 3 columns of C
    
    rank = 2 * rank2
    U = np.zeros((9, rank))
    V = np.zeros((18, rank))
    W = np.zeros((18, rank))
    
    # Block 1: k=0,1,2
    U[:, 0:rank2] = U2
    V[0:9, 0:rank2] = V2
    W[0:9, 0:rank2] = W2  # W indices: k*n+i, for k=0,1,2 -> k*3+i = 0..8
    
    # Block 2: k=3,4,5
    U[:, rank2:rank] = U2
    V[9:18, rank2:rank] = V2
    # W indices: k*n+i, for k=3,4,5 -> k*3+i = 9..17
    W[9:18, rank2:rank] = W2
    
    # Verify
    def verify(U, V, W, n, m, p):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
