import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = 2, 3, 8
    
    # Build the expected tensor
    T_exp = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T_exp[a_idx, b_idx, c_idx] = 1
    
    def verify(U, V, W):
        r = U.shape[1]
        T_check = np.einsum('ad,bd,cd->abc', U, V, W)
        return np.array_equal(T_check, T_exp)
    
    # Rank-44 solution: 4 blocks of rank-11 (2x3x2)
    # Each block: 4 (for j=0 part) + 7 (Strassen for j=1,2 part)
    def build_rank44():
        U = np.zeros((n*m, 44), dtype=np.int64)
        V = np.zeros((m*p, 44), dtype=np.int64)
        W = np.zeros((p*n, 44), dtype=np.int64)
        col_offset = 0
        
        for block in range(4):
            k0 = block * 2
            k1 = block * 2 + 1
            
            # Part 1: A[:,0] * B[0,k0:k1] (2x1 * 1x2) - rank 4
            # t0 = A[0,0]*B[0,k0] -> C[0,k0]
            U[0, col_offset] = 1; V[k0, col_offset] = 1; W[k0*n+0, col_offset] = 1
            col_offset += 1
            # t1 = A[0,0]*B[0,k1] -> C[0,k1]
            U[0, col_offset] = 1; V[k1, col_offset] = 1; W[k1*n+0, col_offset] = 1
            col_offset += 1
            # t2 = A[1,0]*B[0,k0] -> C[1,k0]
            U[3, col_offset] = 1; V[k0, col_offset] = 1; W[k0*n+1, col_offset] = 1
            col_offset += 1
            # t3 = A[1,0]*B[0,k1] -> C[1,k1]
            U[3, col_offset] = 1; V[k1, col_offset] = 1; W[k1*n+1, col_offset] = 1
            col_offset += 1
            
            # Part 2: A[:,1:3] * B[1:3,k0:k1] (2x2 * 2x2) - Strassen rank 7
            # A_sub: [[A[0,1], A[0,2]], [A[1,1], A[1,2]]] = indices [1,2,4,5]
            # B_sub: [[B[1,k0], B[1,k0+1]], [B[2,k0], B[2,k0+1]]] = indices [p+k0, p+k1, 2p+k0, 2p+k1]
            
            # M1 = (A[0,1]+A[1,2])*(B[1,k0]+B[2,k1])
            U[1, col_offset] = 1; U[5, col_offset] = 1
            V[p+k0, col_offset] = 1; V[2*p+k1, col_offset] = 1
            W[k0*n+0, col_offset] = 1; W[k1*n+1, col_offset] = 1
            col_offset += 1
            
            # M2 = (A[1,1]+A[1,2])*B[1,k0]
            U[4, col_offset] = 1; U[5, col_offset] = 1
            V[p+k0, col_offset] = 1
            W[k0*n+1, col_offset] = 1; W[k1*n+1, col_offset] = -1
            col_offset += 1
            
            # M3 = A[0,1]*(B[1,k1]-B[2,k1])
            U[1, col_offset] = 1
            V[p+k1, col_offset] = 1; V[2*p+k1, col_offset] = -1
            W[k1*n+0, col_offset] = 1; W[k1*n+1, col_offset] = 1
            col_offset += 1
            
            # M4 = A[1,2]*(B[2,k0]-B[1,k0])
            U[5, col_offset] = 1
            V[2*p+k0, col_offset] = 1; V[p+k0, col_offset] = -1
            W[k0*n+0, col_offset] = 1; W[k0*n+1, col_offset] = 1
            col_offset += 1
            
            # M5 = (A[0,1]+A[0,2])*B[2,k1]
            U[1, col_offset] = 1; U[2, col_offset] = 1
            V[2*p+k1, col_offset] = 1
            W[k0*n+0, col_offset] = -1; W[k1*n+0, col_offset] = 1
            col_offset += 1
            
            # M6 = (A[1,1]-A[0,1])*(B[1,k0]+B[1,k1])
            U[4, col_offset] = 1; U[1, col_offset] = -1
            V[p+k0, col_offset] = 1; V[p+k1, col_offset] = 1
            W[k1*n+1, col_offset] = 1
            col_offset += 1
            
            # M7 = (A[0,2]-A[1,2])*(B[2,k0]+B[2,k1])
            U[2, col_offset] = 1; U[5, col_offset] = -1
            V[2*p+k0, col_offset] = 1; V[2*p+k1, col_offset] = 1
            W[k0*n+0, col_offset] = 1
            col_offset += 1
        
        return U, V, W
    
    U44, V44, W44 = build_rank44()
    
    if verify(U44, V44, W44):
        best_U, best_V, best_W = U44, V44, W44
        best_rank = 44
    else:
        # Fallback to schoolbook
        rank_sb = n * m * p
        U_sb = np.zeros((n*m, rank_sb), dtype=np.int64)
        V_sb = np.zeros((m*p, rank_sb), dtype=np.int64)
        W_sb = np.zeros((p*n, rank_sb), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    U_sb[a_idx, col] = 1
                    V_sb[b_idx, col] = 1
                    W_sb[c_idx, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Try to improve: attempt rank-40 (10 per block for 2x3x2)
    # For 2x3x2: C[0,0]=a0.b0, C[0,1]=a0.b1, C[1,0]=a1.b0, C[1,1]=a1.b1
    # where a0=(a00,a01,a02), a1=(a10,a11,a12), b0=(b00,b10,b20), b1=(b01,b11,b21)
    # 
    # Try 10 multiplications:
    # P1 = (a00+a11)*(b00+b11)
    # P2 = (a01+a12)*(b01+b12)
    # P3 = (a02+a10)*(b02+b10)
    # P4 = a00*b11
    # P5 = a11*b00
    # P6 = a01*b12
    # P7 = a12*b01
    # P8 = a02*b10
    # P9 = a10*b02
    # P10 = (a00+a12)*(b01+b10)  -- need to check
    
    # Let me try a systematic approach with 10 terms
    # C[0,0] = a00*b00 + a01*b10 + a02*b20
    # C[0,1] = a00*b01 + a01*b11 + a02*b21
    # C[1,0] = a10*b00 + a11*b10 + a12*b20
    # C[1,1] = a10*b01 + a11*b11 + a12*b21
    
    # 10 products:
    # P1 = (a00+a11)*(b00+b11) = a00*b00 + a00*b11 + a11*b00 + a11*b11
    # P2 = (a01+a12)*(b01+b12) = a01*b01 + a01*b12 + a12*b01 + a12*b12  -- wait, b12 doesn't exist
    # I mean b[1,k] where k is the column. Let me re-index.
    # b0 = (B[0,k0], B[1,k0], B[2,k0]), b1 = (B[0,k1], B[1,k1], B[2,k1])
    
    # Let me denote:
    # a0 = (a00, a01, a02) = A[0,0], A[0,1], A[0,2]
    # a1 = (a10, a11, a12) = A[1,0], A[1,1], A[1,2]
    # b0 = (b00, b10, b20) = B[0,k0], B[1,k0], B[2,k0]
    # b1 = (b01, b11, b21) = B[0,k1], B[1,k1], B[2,k1]
    
    # C[0,0] = a00*b00 + a01*b10 + a02*b20
    # C[0,1] = a00*b01 + a01*b11 + a02*b21
    # C[1,0] = a10*b00 + a11*b10 + a12*b20
    # C[1,1] = a10*b01 + a11*b11 + a12*b21
    
    # 10 products attempt:
    # P1 = (a00+a11)*(b00+b11)
    # P2 = (a01+a12)*(b01+b12) -- b12 doesn't exist, should be b[1,k1]=b11... 
    # I'm confusing myself. Let me use clearer notation.
    
    # For a single 2x3x2 block with k0, k1:
    # A entries: A[0,0], A[0,1], A[0,2], A[1,0], A[1,1], A[1,2]
    # B entries: B[0,k0], B[1,k0], B[2,k0], B[0,k1], B[1,k1], B[2,k1]
    # C entries: C[0,k0], C[0,k1], C[1,k0], C[1,k1]
    
    # C[0,k0] = A[0,0]*B[0,k0] + A[0,1]*B[1,k0] + A[0,2]*B[2,k0]
    # C[0,k1] = A[0,0]*B[0,k1] + A[0,1]*B[1,k1] + A[0,2]*B[2,k1]
    # C[1,k0] = A[1,0]*B[0,k0] + A[1,1]*B[1,k0] + A[1,2]*B[2,k0]
    # C[1,k1] = A[1,0]*B[0,k1] + A[1,1]*B[1,k1] + A[1,2]*B[2,k1]
    
    # Try 10 products:
    # P1 = (A[0,0]+A[1,1])*(B[0,k0]+B[1,k1])
    # P2 = (A[0,1]+A[1,2])*(B[0,k1]+B[1,k0])
    # P3 = (A[0,2]+A[1,0])*(B[1,k0]+B[2,k1])
    # P4 = A[0,0]*B[1,k1]
    # P5 = A[1,1]*B[0,k0]
    # P6 = A[0,1]*B[1,k0]
    # P7 = A[1,2]*B[0,k1]
    # P8 = A[0,2]*B[2,k1]
    # P9 = A[1,0]*B[1,k0]
    # P10 = (A[0,0]+A[0,2])*(B[2,k0]+B[2,k1])
    
    # This is getting complicated. Let me just verify the rank-44 solution and return it.
    # The rank-44 solution is verified working and gives score 1.0909.
    
    if time.time() > deadline - 20:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Try a rank-40 construction with 10 per block
    # Using a known rank-10 scheme for 2x3x2 if it exists
    # For now, return the rank-44 solution
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
