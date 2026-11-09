"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Decompose M_{2,5,5} into 4 blocks:
    # m = 2 + 3, p = 2 + 3
    # A (2x5) -> [A1 (2x2), A2 (2x3)]
    # B (5x5) -> [[B11 (2x2), B12 (2x3)], [B21 (3x2), B22 (3x3)]]
    # C (2x5) -> [C1 (2x2), C2 (2x3)]
    # C1 = A1*B11 + A2*B21
    # C2 = A1*B12 + A2*B22
    
    # Block ranks:
    # M_{2,2,2}: Strassen rank 7
    # M_{2,2,3}: schoolbook rank 12
    # M_{2,3,2}: schoolbook rank 12
    # M_{2,3,3}: schoolbook rank 18
    # Total: 7 + 12 + 12 + 18 = 49
    
    total_rank = 49
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Block 1: M_{2,2,2} - Strassen (rank 7)
    # A1 is rows 0-1 of A (indices 0,1 in A)
    # B11 is rows 0-1, cols 0-1 of B
    # C1 part is rows 0-1, cols 0-1 of C
    
    # Strassen for 2x2x2:
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)b00
    # m3 = a00(b01-b11)
    # m4 = a11(b10-b00)
    # m5 = (a00+a01)b11
    # m6 = (a10-a00)(b00+b10)
    # m7 = (a01-a11)(b01+b11)
    # c00 = m1 + m4 - m5 + m7
    # c01 = m3 + m5
    # c10 = m2 + m4
    # c11 = m1 - m2 + m3 + m6
    
    # U, V, W for Strassen (local indices: i in {0,1} for A rows, j in {0,1} for B rows/A cols, k in {0,1} for C cols)
    # U[i*m+j] where m=2 locally, but we're in global coordinates
    # For block 1: A rows 0,1; B rows 0,1; C cols 0,1
    
    # Strassen U vectors (for A part, size 2x2 = 4 entries):
    # u1 = [1, 0, 0, 1]  (a00+a11)
    # u2 = [0, 1, 0, 1]  (a10+a11)
    # u3 = [1, 0, 0, 0]  (a00)
    # u4 = [0, 0, 0, 1]  (a11)
    # u5 = [1, 0, 1, 0]  (a00+a01)
    # u6 = [0, 1, -1, 0] (a10-a00)
    # u7 = [0, 0, 1, -1] (a01-a11)
    
    # Strassen V vectors (for B part, size 2x2 = 4 entries):
    # v1 = [1, 0, 0, 1]  (b00+b11)
    # v2 = [1, 0, 0, 0]  (b00)
    # v3 = [0, 1, 0, -1] (b01-b11)
    # v4 = [0, 0, 1, -1] (b10-b00)
    # v5 = [0, 0, 0, 1]  (b11)
    # v6 = [1, 1, 0, 0]  (b00+b10)
    # v7 = [0, 0, 1, 1]  (b01+b11)
    
    # Strassen W vectors (for C part, size 2x2 = 4 entries):
    # w1 = [1, 0, 0, -1]  (contributes to c00 with +1, c11 with -1)
    # w2 = [0, 0, 1, -1]  (contributes to c10 with +1, c11 with -1)
    # w3 = [0, 1, 0, 1]   (contributes to c01 with +1, c11 with +1)
    # w4 = [1, 0, 1, 0]   (contributes to c00 with +1, c10 with +1)
    # w5 = [-1, 1, 0, 0]  (contributes to c00 with -1, c01 with +1)
    # w6 = [0, 0, 0, 1]   (contributes to c11 with +1)
    # w7 = [1, 0, 0, 0]   (contributes to c00 with +1)
    
    # Let me re-derive more carefully.
    # C = AB where A is 2x2, B is 2x2, C is 2x2
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    
    # Strassen:
    # m1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    # m2 = (A[1,0]+A[1,1])*B[0,0]
    # m3 = A[0,0]*(B[0,1]-B[1,1])
    # m4 = A[1,1]*(B[1,0]-B[0,0])
    # m5 = (A[0,0]+A[0,1])*B[1,1]
    # m6 = (A[1,0]-A[0,0])*(B[0,0]+B[1,0])
    # m7 = (A[0,1]-A[1,1])*(B[0,1]+B[1,1])
    
    # C[0,0] = m1 + m4 - m5 + m7
    # C[0,1] = m3 + m5
    # C[1,0] = m2 + m4
    # C[1,1] = m1 - m2 + m3 + m6
    
    # Now express as sum of rank-1 tensors:
    # T = sum_l u_l ⊗ v_l ⊗ w_l
    
    # For each m_l, we have:
    # m1: u1=[1,0,0,1], v1=[1,0,0,1], w1=[1,0,0,-1] (coefficients for C[0,0]=+1, C[0,1]=0, C[1,0]=0, C[1,1]=-1)
    # m2: u2=[0,1,0,1], v2=[1,0,0,0], w2=[0,0,1,-1] (C[0,0]=0, C[0,1]=0, C[1,0]=+1, C[1,1]=-1)
    # m3: u3=[1,0,0,0], v3=[0,1,0,-1], w3=[0,1,0,1] (C[0,0]=0, C[0,1]=+1, C[1,0]=0, C[1,1]=+1)
    # m4: u4=[0,0,0,1], v4=[0,0,1,-1], w4=[1,0,1,0] (C[0,0]=+1, C[0,1]=0, C[1,0]=+1, C[1,1]=0)
    # m5: u5=[1,0,1,0], v5=[0,0,0,1], w5=[-1,1,0,0] (C[0,0]=-1, C[0,1]=+1, C[1,0]=0, C[1,1]=0)
    # m6: u6=[0,1,-1,0], v6=[1,1,0,0], w6=[0,0,0,1] (C[0,0]=0, C[0,1]=0, C[1,0]=0, C[1,1]=+1)
    # m7: u7=[0,0,1,-1], v7=[0,0,1,1], w7=[1,0,0,0] (C[0,0]=+1, C[0,1]=0, C[1,0]=0, C[1,1]=0)
    
    # Let me verify m1:
    # m1 = (A00+A11)(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11
    # This should contribute to:
    # C00: A00*B00 (yes) and A11*B11 (no, A11*B11 goes to C11)
    # Actually wait, let me recheck.
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # m1 = (A00+A11)(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11
    # In C00: A00*B00 appears. Coefficient in m1 is +1.
    # In C01: A00*B11 appears. Coefficient in m1 is +1.
    # In C10: A11*B00 appears. Coefficient in m1 is +1.
    # In C11: A11*B11 appears. Coefficient in m1 is +1.
    # But we need C00 to get +1 from m1, C01 to get 0, C10 to get 0, C11 to get -1 (from the formula C11 = m1 - m2 + m3 + m6)
    # So w1 should be [1, 0, 0, -1] for C indices [00, 01, 10, 11]
    
    # Let me redo this more carefully by checking each term.
    
    # I'll use a different approach: just hardcode the known Strassen U,V,W matrices.
    
    # For 2x2 matrix multiplication, the Strassen decomposition has:
    # U (4x7), V (4x7), W (4x7) where 4 = 2*2
    
    U_strassen = [
        [1, 0, 1, 0, 1, 0, 0],   # A[0,0]
        [0, 1, 0, 0, 0, 1, 0],   # A[0,1]
        [0, 0, 0, 1, 0, -1, 1],  # A[1,0]
        [1, 1, 0, 1, 0, 0, -1],  # A[1,1]
    ]
    
    V_strassen = [
        [1, 1, 0, 0, 0, 1, 0],   # B[0,0]
        [0, 0, 1, 0, 0, 1, 1],   # B[0,1]
        [0, 0, 0, 1, 1, 0, 0],   # B[1,0]
        [1, 0, -1, -1, 1, 0, 1], # B[1,1]
    ]
    
    W_strassen = [
        [1, 0, 0, 1, -1, 0, 1],  # C[0,0]
        [0, 0, 1, 0, 1, 0, 0],   # C[0,1]
        [0, 1, 0, 1, 0, 0, 0],   # C[1,0]
        [-1, -1, 1, 0, 0, 1, 0], # C[1,1]
    ]
    
    # Verify: sum_l U[i,l]*V[j,l]*W[k,l] should equal delta_{ik}... no wait.
    # For matrix mult, T[i,j,k] = 1 for all i,j,k (each A[i,j]*B[j,k] contributes to C[i,k])
    # So we need sum_l U[i*2+j, l] * V[j*2+k, l] * W[k*2+i, l] = 1 for all i,j,k
    
    # Let me verify with a simple case.
    # Actually, the tensor for mat mult is: C[i,k] = sum_j A[i,j] B[j,k]
    # So T[i,j,k] = 1 (coefficient of A[i,j]B[j,k] in C[i,k])
    # We need: sum_l U[i*2+j, l] * V[j*2+k, l] * W[k*2+i, l] = 1
    
    # Let me verify for i=0, j=0, k=0:
    # sum_l U[0,l] * V[0,l] * W[0,l]
    # = 1*1*1 + 0*1*0 + 1*0*0 + 0*0*1 + 1*0*(-1) + 0*1*0 + 0*0*1
    # = 1 + 0 + 0 + 0 + 0 + 0 + 0 = 1 ✓
    
    # For i=0, j=0, k=1:
    # sum_l U[0,l] * V[1,l] * W[1,l]
    # = 1*0*0 + 0*0*0 + 1*1*1 + 0*0*0 + 1*0*1 + 0*1*0 + 0*0*0
    # = 0 + 0 + 1 + 0 + 0 + 0 + 0 = 1 ✓
    
    # For i=0, j=1, k=0:
    # sum_l U[1,l] * V[2,l] * W[0,l]
    # = 0*0*1 + 1*0*0 + 0*0*0 + 0*1*1 + 0*1*(-1) + 1*0*0 + 0*0*1
    # = 0 + 0 + 0 + 0 + 0 + 0 + 0 = 0 ✗ Should be 1!
    
    # My Strassen matrices are wrong. Let me look up the correct ones.
    
    # Correct Strassen for 2x2:
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)b00
    # m3 = a00(b01-b11)
    # m4 = a11(b10-b00)
    # m5 = (a00+a01)b11
    # m6 = (a10-a00)(b00+b10)
    # m7 = (a01-a11)(b01+b11)
    
    # c00 = m1 + m4 - m5 + m7
    # c01 = m3 + m5
    # c10 = m2 + m4
    # c11 = m1 - m2 + m3 + m6
    
    # U vectors (for A, indexed as [a00, a01, a10, a11]):
    # u1 = [1, 0, 0, 1]
    # u2 = [0, 0, 1, 1]
    # u3 = [1, 0, 0, 0]
    # u4 = [0, 0, 0, 1]
    # u5 = [1, 1, 0, 0]
    # u6 = [0, 0, 1, -1]
    # u7 = [0, 1, 0, -1]
    
    # V vectors (for B, indexed as [b00, b01, b10, b11]):
    # v1 = [1, 0, 0, 1]
    # v2 = [1, 0, 0, 0]
    # v3 = [0, 1, 0, -1]
    # v4 = [0, 0, 1, -1]
    # v5 = [0, 0, 0, 1]
    # v6 = [1, 0, 1, 0]
    # v7 = [0, 1, 0, 1]
    
    # W vectors (for C, indexed as [c00, c01, c10, c11]):
    # w1 = [1, 0, 0, -1]
    # w2 = [0, 0, 1, -1]
    # w3 = [0, 1, 0, 1]
    # w4 = [1, 0, 1, 0]
    # w5 = [-1, 1, 0, 0]
    # w6 = [0, 0, 0, 1]
    # w7 = [1, 0, 0, 0]
    
    U_strassen = [
        [1, 0, 1, 0, 1, 0, 0],   # a00
        [0, 0, 0, 0, 1, 0, 1],   # a01
        [0, 1, 0, 0, 0, 1, 0],   # a10
        [1, 1, 0, 1, 0, -1, -1], # a11
    ]
    
    V_strassen = [
        [1, 1, 0, 0, 0, 1, 0],   # b00
        [0, 0, 1, 0, 0, 0, 1],   # b01
        [0, 0, 0, 1, 0, 1, 0],   # b10
        [1, 0, -1, -1, 1, 0, 1], # b11
    ]
    
    W_strassen = [
        [1, 0, 0, 1, -1, 0, 1],  # c00
        [0, 0, 1, 0, 1, 0, 0],   # c01
        [0, 1, 0, 1, 0, 0, 0],   # c10
        [-1, -1, 1, 0, 0, 1, 0], # c11
    ]
    
    # Verify i=0, j=1, k=0:
    # sum_l U[1,l] * V[2,l] * W[0,l]
    # U[1] = [0, 0, 0, 0, 1, 0, 1]
    # V[2] = [0, 0, 0, 1, 0, 1, 0]
    # W[0] = [1, 0, 0, 1, -1, 0, 1]
    # = 0*0*1 + 0*0*0 + 0*0*0 + 0*1*1 + 1*0*(-1) + 0*1*0 + 1*0*1
    # = 0 + 0 + 0 + 0 + 0 + 0 + 0 = 0 ✗
    
    # Still wrong. Let me just use schoolbook for all blocks to be safe.
    # Rank 50 is guaranteed correct.
    
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
