"""Strassen-based decomposition for 4x4x8 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Strassen 2x2x2 decomposition (rank 7)
    # Input A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # C = A*B, C[i,j] = sum_k A[i,k]*B[k,j]
    # 
    # The 7 Strassen terms:
    # M1 = (a11+a22)*(b11+b22)
    # M2 = (a21+a22)*b11
    # M3 = a11*(b12-b22)
    # M4 = a22*(b21-b11)
    # M5 = (a11+a12)*b22
    # M6 = (a21-a11)*(b11+b12)
    # M7 = (a12-a22)*(b21+b22)
    #
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M7
    
    # For 4x4x4 as Kronecker product of two 2x2x2:
    # Index A[i,j] where i=2*i1+i0, j=2*j1+j0
    # Index B[j,k] where j=2*j1+j0, k=2*k1+k0
    # Index C[i,k] where i=2*i1+i0, k=2*k1+k0
    #
    # The rank-49 decomposition: for each pair of Strassen indices (s1, s2) in [0..6]x[0..6]
    # U index = (i1, i0, j1, j0) mapped to 4x4 input
    # V index = (j1, j0, k1, k0) mapped to 4x4 input  
    # W index = (k1, k0, i1, i0) mapped to 4x4 output (trace convention: W[k*n+i])
    
    # Build Strassen 2x2x2 factors
    # U2: 4 entries (i,j for 2x2 A), 7 columns
    # V2: 4 entries (j,k for 2x2 B), 7 columns  
    # W2: 4 entries (k,i for 2x2 C in trace convention), 7 columns
    
    # Strassen coefficients for U (input A part)
    # M1: a11 + a22 -> U: [1,0,0,1]
    # M2: a21 + a22 -> U: [0,0,1,1]
    # M3: a11 -> U: [1,0,0,0]
    # M4: a22 -> U: [0,0,0,1]
    # M5: a11 + a12 -> U: [1,1,0,0]
    # M6: a21 - a11 -> U: [-1,0,1,0]
    # M7: a12 - a22 -> U: [0,1,-1,0]
    
    U2 = [
        [1, 0, 1, 0, 1, -1, 0],  # a11
        [0, 0, 0, 0, 1, 0, 1],   # a12
        [0, 1, 0, 0, 0, 1, 0],   # a21
        [1, 1, 0, 1, 0, 0, -1]   # a22
    ]
    
    # Strassen coefficients for V (input B part)
    # M1: b11 + b22 -> V: [1,0,0,1]
    # M2: b11 -> V: [1,0,0,0]
    # M3: b12 - b22 -> V: [0,1,0,-1]
    # M4: b21 - b11 -> V: [-1,0,1,0]
    # M5: b22 -> V: [0,0,0,1]
    # M6: b11 + b12 -> V: [1,1,0,0]
    # M7: b21 + b22 -> V: [0,0,1,1]
    
    V2 = [
        [1, 1, 0, -1, 0, 1, 0],  # b11
        [0, 0, 1, 0, 0, 1, 0],   # b12
        [0, 0, 0, 1, 0, 0, 1],   # b21
        [1, 0, -1, 0, 1, 0, 1]   # b22
    ]
    
    # Strassen coefficients for W (output C part, trace convention: C[k,i] = C[i,k]^T)
    # C11 = M1 - M4 + M5 - M7 -> W[C11]: [1,0,1,0,-1,0,1]
    # C12 = M3 + M5 -> W[C12]: [0,0,1,0,1,0,0]
    # C21 = M2 + M4 -> W[C21]: [0,1,0,1,0,0,0]
    # C22 = M1 - M2 + M3 + M7 -> W[C22]: [1,-1,1,0,0,0,1]
    
    # Trace convention: W[k*n+i, col] where output is C[i,k], so W index = k*2+i for 2x2
    # C[i,k] where i=row, k=col. W[k*2+i, col]
    # So W[0, col] = C[0,0] coeff, W[1, col] = C[0,1] coeff, W[2, col] = C[1,0] coeff, W[3, col] = C[1,1] coeff
    
    W2 = [
        [1, 0, 0, 1, 0, 0, 1],   # C[0,0] = M1 - M4 + M5 - M7
        [0, 0, 1, 0, 1, 0, 0],   # C[0,1] = M3 + M5
        [0, 1, 0, 1, 0, 0, 0],   # C[1,0] = M2 + M4
        [1, -1, 1, 0, 0, 0, 1]   # C[1,1] = M1 - M2 + M3 + M7
    ]
    
    # Now build 4x4x4 rank-49 decomposition via Kronecker product
    # Input A is 4x4: A[i,j] with i=2*i1+i0, j=2*j1+j0
    # Input B is 4x4: B[j,k] with j=2*j1+j0, k=2*k1+k0
    # Output C is 4x4: C[i,k] with i=2*i1+i0, k=2*k1+k0
    # Trace convention: W[k*4+i, col]
    
    # For Kronecker product of two 2x2x2:
    # The 49 terms are indexed by (s1, s2) where s1, s2 in 0..6
    # For term (s1, s2):
    #   U_A[i,j] = U2[i1, s1] * U2[i0, s2]  where i=2*i1+i0, j=2*j1+j0... 
    #   Wait, I need to be more careful.
    
    # Actually for the Kronecker product structure:
    # A[i,j] = A[i1,j1] * A[i0,j0] in a block sense
    # The tensor for 4x4x4 is the Kronecker product of the 2x2x2 tensor with itself
    # 
    # U_4x4x4[i, j, col] where col = s1*7 + s2
    # i = 2*i1 + i0, j = 2*j1 + j0
    # U_4x4x4[2*i1+i0, 2*j1+j0, s1*7+s2] = U2[i1, s1] * U2[j1, s1] * U2[i0, s2] * U2[j0, s2]
    # Hmm, that's not quite right either.
    
    # Let me think differently. The 4x4x4 tensor T[i,j,k] = delta (for C[i,k] = sum_j A[i,j]B[j,k])
    # This tensor is the Kronecker product T_4 = T_2 ⊗ T_2
    # where T_2[i,j,k] is the 2x2x2 multiplication tensor.
    #
    # A rank-r decomposition of T_2 gives U2[i, r], V2[j, r], W2[k, r] such that
    # T_2[i,j,k] = sum_r U2[i,r]*V2[j,r]*W2[k,r]
    #
    # Then T_4[i1*2+i0, j1*2+j0, k1*2+k0] = T_2[i1,j1,k1] * T_2[i0,j0,k0]
    # = (sum_r1 U2[i1,r1]*V2[j1,r1]*W2[k1,r1]) * (sum_r2 U2[i0,r2]*V2[j0,r2]*W2[k0,r2])
    # = sum_{r1,r2} U2[i1,r1]*U2[i0,r2] * V2[j1,r1]*V2[j0,r2] * W2[k1,r1]*W2[k0,r2]
    #
    # So the rank-49 decomposition:
    # col = r1*7 + r2
    # U4[2*i1+i0, 2*j1+j0, r1*7+r2] = U2[i1, r1] * U2[i0, r2]
    # V4[2*j1+j0, 2*k1+k0, r1*7+r2] = V2[j1, r1] * V2[j0, r2]
    # W4[2*k1+k0, 2*i1+i0, r1*7+r2] = W2[k1, r1] * W2[k0, r2]
    #   (note: W index is k*4+i in trace convention, so W4[k*4+i, col])
    
    rank_4x4x4 = 49
    U4 = [[0] * rank_4x4x4 for _ in range(16)]  # 4*4 = 16 rows
    V4 = [[0] * rank_4x4x4 for _ in range(16)]  # 4*4 = 16 rows
    W4 = [[0] * rank_4x4x4 for _ in range(16)]  # 4*4 = 16 rows (trace: k*4+i)
    
    for r1 in range(7):
        for r2 in range(7):
            col = r1 * 7 + r2
            for i1 in range(2):
                for i0 in range(2):
                    for j1 in range(2):
                        for j0 in range(2):
                            for k1 in range(2):
                                for k0 in range(2):
                                    i = 2 * i1 + i0
                                    j = 2 * j1 + j0
                                    k = 2 * k1 + k0
                                    u_val = U2[i1][r1] * U2[i0][r2]
                                    v_val = V2[j1][r1] * V2[j0][r2]
                                    w_val = W2[k1][r1] * W2[k0][r2]
                                    U4[i * 4 + j][col] = u_val
                                    V4[j * 4 + k][col] = v_val
                                    W4[k * 4 + i][col] = w_val  # trace convention: k*4+i
    
    # Now for 4x4x8: C = A * B where A is 4x4, B is 4x8, C is 4x8
    # Split B = [B1 | B2] where B1, B2 are 4x4
    # C = [C1 | C2] where C1 = A*B1, C2 = A*B2
    # Each Ci uses the rank-49 decomposition
    # Total rank = 98
    
    rank = 98
    U = [[0] * rank for _ in range(16)]   # n*m = 16
    V = [[0] * rank for _ in range(32)]   # m*p = 4*8 = 32
    W = [[0] * rank for _ in range(32)]   # p*n = 8*4 = 32 (trace: k*4+i, k in 0..7, i in 0..3)
    
    # First 49 terms: for C1 = A * B1 (columns 0-3 of B)
    for col in range(49):
        for i in range(4):
            for j in range(4):
                U[i * 4 + j][col] = U4[i * 4 + j][col]
        for j in range(4):
            for k in range(4):
                V[j * 8 + k][col] = V4[j * 4 + k][col]  # B1 is columns 0-3
        for k in range(4):
            for i in range(4):
                W[k * 4 + i][col] = W4[k * 4 + i][col]  # C1 is columns 0-3
    
    # Next 49 terms: for C2 = A * B2 (columns 4-7 of B)
    for col in range(49):
        for i in range(4):
            for j in range(4):
                U[i * 4 + j][col + 49] = U4[i * 4 + j][col]
        for j in range(4):
            for k in range(4):
                V[j * 8 + (k + 4)][col + 49] = V4[j * 4 + k][col]  # B2 is columns 4-7
        for k in range(4):
            for i in range(4):
                W[(k + 4) * 4 + i][col + 49] = W4[k * 4 + i][col]  # C2 is columns 4-7
    
    # Verify the decomposition (quick check with random matrices)
    if time.time() < deadline - 5:
        try:
            A_test = np.random.randint(-2, 3, (4, 4))
            B_test = np.random.randint(-2, 3, (4, 8))
            C_expected = A_test @ B_test
            
            U_arr = np.array(U)
            V_arr = np.array(V)
            W_arr = np.array(W)
            
            # C[i,k] = sum_{j,r} U[i,j,r] * V[j,k,r] * W[k,i,r]
            # In trace convention: W[k*n+i, r]
            C_computed = np.zeros((4, 8))
            for i in range(4):
                for k in range(8):
                    s = 0
                    for j in range(4):
                        for r in range(rank):
                            s += U_arr[i * 4 + j, r] * V_arr[j * 8 + k, r] * W_arr[k * 4 + i, r]
                    C_computed[i, k] = s
            
            if not np.allclose(C_computed, C_expected):
                # Fallback to schoolbook if verification fails
                rank_fb = 128
                U_fb = [[0] * rank_fb for _ in range(16)]
                V_fb = [[0] * rank_fb for _ in range(32)]
                W_fb = [[0] * rank_fb for _ in range(32)]
                c = 0
                for i in range(4):
                    for j in range(4):
                        for k in range(8):
                            U_fb[i * 4 + j][c] = 1
                            V_fb[j * 8 + k][c] = 1
                            W_fb[k * 4 + i][c] = 1
                            c += 1
                return {"U": U_fb, "V": V_fb, "W": W_fb}
        except Exception:
            pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
