"""Rewritten program: exploit structure of 2x2 by 2x7 multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x2 * 2x7 matrix multiplication.
    
    Strategy: 
    - The multiplication tensor has dimensions (n*m, m*p, p*n) = (4, 14, 14).
    - We use a structured approach based on known fast matrix multiplication.
    - For 2x2 * 2x2, Strassen gives rank 7 (vs schoolbook 8).
    - For 2x2 * 2x7, we can decompose into 2x2 * 2x4 + 2x2 * 2x3.
    - 2x2 * 2x4 can be seen as 2x2 * 2x2 + 2x2 * 2x2, so rank 7+7=14 with Strassen.
    - 2x2 * 2x3: schoolbook is 12, but we can do better.
    
    Actually, let me think more carefully. The tensor is T_{i,j,k} where:
    - i indexes rows of A (n=2) and columns of A (m=2): so i in [0,3] for (row_A, col_A)
    - j indexes rows of B (m=2) and columns of B (p=7): so j in [0,13] for (row_B, col_B)  
    - k indexes rows of output (p=7) and columns of output (n=2): so k in [0,13] for (row_out, col_out)
    
    Wait, the convention is W[k*n+i, r] where k is row of B/output and i is col of A/output.
    Let me re-read: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    
    The tensor entry T[a,b,c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k.
    So T[i*m+j, j*p+k, k*n+i] = 1.
    
    For a rank-r decomposition, we need U, V, W such that:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i*m+j, j*p+k, k*n+i]
    
    Schoolbook: rank = n*m*p = 2*2*7 = 28.
    
    Let me try to use Strassen-like ideas. For the 2x2 * 2x2 part, Strassen uses 7 multiplications.
    
    Approach: Split the 7 columns of B into groups.
    B is 2x7. Let's write B = [B1 | B2 | B3] where B1 is 2x2, B2 is 2x2, B3 is 2x3.
    
    Then A*B = A*B1 + A*B2 + A*B3.
    
    For A*B1 and A*B2, each is 2x2 * 2x2, which can be done with rank 7 (Strassen).
    For A*B3, which is 2x2 * 2x3, the schoolbook rank is 12.
    
    Total: 7 + 7 + 12 = 26.
    
    But we can do better. Let me think about 2x2 * 2x3.
    
    Actually, let me just implement a search for a good decomposition.
    Given the time constraint, let me use a heuristic approach.
    
    Alternative: Use the fact that this is a small tensor and try to find a low-rank decomposition
    via numerical optimization, then round to integers.
    
    Let me try a different approach: use known results.
    The rank of the 2x2 * 2x7 multiplication tensor... 
    
    Actually, let me just do a careful structured construction.
    
    For 2x2 * 2x7:
    Let A = [[a11, a12], [a21, a22]], B = [[b11, b12, b13, b14, b15, b16, b17],
                                            [b21, b22, b23, b24, b25, b26, b27]]
    
    C = A*B is 2x7:
    C[i,j] = sum_k A[i,k] * B[k,j]
    
    C[0,0] = a11*b11 + a12*b21
    C[0,1] = a11*b12 + a12*b22
    ...
    C[1,0] = a21*b11 + a22*b21
    ...
    
    The tensor has 4*14*14 = 784 entries, mostly zero.
    
    Let me try to use a decomposition based on Strassen for the 2x2 blocks.
    
    For the first 4 columns of B (columns 0-3), we can use two Strassen decompositions.
    For the last 3 columns, we use schoolbook.
    
    Strassen for 2x2 * 2x2:
    p1 = (a11+a22)*(b11+b22)
    p2 = (a21+a22)*b11
    p3 = a11*(b12-b22)
    p4 = a22*(b21-b11)
    p5 = (a11+a12)*b22
    p6 = (a21-a11)*(b11+b12)
    p7 = (a12-a22)*(b21+b22)
    
    c11 = p1 + p4 - p5 + p7
    c12 = p3 + p5
    c21 = p2 + p4
    c22 = p1 - p2 + p3 + p6
    
    So for A * B[:, 0:2], we use Strassen with 7 multiplications.
    For A * B[:, 2:4], we use another Strassen with 7 multiplications.
    For A * B[:, 4:7], we use schoolbook with 12 multiplications.
    
    Total rank: 7 + 7 + 12 = 26.
    
    Let me implement this.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=7
    rank = 26
    U = [[0.0] * rank for _ in range(n * m)]  # 4 x 26
    V = [[0.0] * rank for _ in range(m * p)]  # 14 x 26
    W = [[0.0] * rank for _ in range(p * n)]  # 14 x 26
    
    # U indexes: i*m + j where i in [0,1] (row of A), j in [0,1] (col of A)
    # V indexes: j*p + k where j in [0,1] (row of B), k in [0,6] (col of B)
    # W indexes: k*n + i where k in [0,6] (row of C), i in [0,1] (col of C)
    
    # Strassen for columns 0-1 of B (first 2x2 block)
    # Multiplication indices: 0-6
    # p1 = (a11+a22)*(b11+b22)
    # p2 = (a21+a22)*b11
    # p3 = a11*(b12-b22)
    # p4 = a22*(b21-b11)
    # p5 = (a11+a12)*b22
    # p6 = (a21-a11)*(b11+b12)
    # p7 = (a12-a22)*(b21+b22)
    
    # A indices: a11=U[0], a12=U[1], a21=U[2], a22=U[3]
    # B indices: b11=V[0], b12=V[1], b21=V[2], b22=V[3] (for first 2 cols)
    # C indices: c11=W[0], c12=W[1], c21=W[2], c22=W[3] (for first 2 cols)
    
    # p1
    U[0][0] += 1; U[3][0] += 1  # a11 + a22
    V[0][0] += 1; V[3][0] += 1  # b11 + b22
    W[0][0] += 1; W[1][0] += 1; W[2][0] += 1; W[3][0] += 1  # c11,c12,c21,c22 get +1
    
    # Wait, W should only contribute to specific C entries. Let me be more careful.
    # c11 = p1 + p4 - p5 + p7
    # c12 = p3 + p5
    # c21 = p2 + p4
    # c22 = p1 - p2 + p3 + p6
    
    # So for p1: contributes +1 to c11, +1 to c22
    # For p2: contributes +1 to c21, -1 to c22
    # For p3: contributes +1 to c12, +1 to c22
    # For p4: contributes +1 to c11, +1 to c21
    # For p5: contributes -1 to c11, +1 to c12
    # For p6: contributes +1 to c22
    # For p7: contributes +1 to c11
    
    # Let me redo this properly.
    
    # Reset U, V, W
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen block 1: B columns 0-1
    # p1 = (a11+a22)*(b11+b22)
    U[0][0] += 1; U[3][0] += 1
    V[0][0] += 1; V[3][0] += 1
    W[0][0] += 1  # c11
    W[3][0] += 1  # c22
    
    # p2 = (a21+a22)*b11
    U[2][1] += 1; U[3][1] += 1
    V[0][1] += 1
    W[2][1] += 1  # c21
    W[3][1] -= 1  # c22
    
    # p3 = a11*(b12-b22)
    U[0][2] += 1
    V[1][2] += 1; V[3][2] -= 1
    W[1][2] += 1  # c12
    W[3][2] += 1  # c22
    
    # p4 = a22*(b21-b11)
    U[3][3] += 1
    V[2][3] += 1; V[0][3] -= 1
    W[0][3] += 1  # c11
    W[2][3] += 1  # c21
    
    # p5 = (a11+a12)*b22
    U[0][4] += 1; U[1][4] += 1
    V[3][4] += 1
    W[0][4] -= 1  # c11
    W[1][4] += 1  # c12
    
    # p6 = (a21-a11)*(b11+b12)
    U[2][5] += 1; U[0][5] -= 1
    V[0][5] += 1; V[1][5] += 1
    W[3][5] += 1  # c22
    
    # p7 = (a12-a22)*(b21+b22)
    U[1][6] += 1; U[3][6] -= 1
    V[2][6] += 1; V[3][6] += 1
    W[0][6] += 1  # c11
    
    # Strassen block 2: B columns 2-3
    # p1' = (a11+a22)*(b13+b23)  [b13=B[0,2], b23=B[1,2]]
    # p2' = (a21+a22)*b13
    # p3' = a11*(b14-b24)  [b14=B[0,3], b24=B[1,3]]
    # p4' = a22*(b23-b13)
    # p5' = (a11+a12)*b24
    # p6' = (a21-a11)*(b13+b14)
    # p7' = (a12-a22)*(b23+b24)
    
    # C entries for cols 2-3: c13=W[4], c14=W[5], c23=W[6], c24=W[7]
    
    # p1'
    U[0][7] += 1; U[3][7] += 1
    V[2][7] += 1; V[5][7] += 1  # b13=V[2], b23=V[5]
    W[4][7] += 1  # c13
    W[7][7] += 1  # c24
    
    # p2'
    U[2][8] += 1; U[3][8] += 1
    V[2][8] += 1  # b13
    W[6][8] += 1  # c23
    W[7][8] -= 1  # c24
    
    # p3'
    U[0][9] += 1
    V[3][9] += 1; V[5][9] -= 1  # b14=V[3], b24=V[5]
    W[5][9] += 1  # c14
    W[7][9] += 1  # c24
    
    # p4'
    U[3][10] += 1
    V[5][10] += 1; V[2][10] -= 1  # b23=V[5], b13=V[2]
    W[4][10] += 1  # c13
    W[6][10] += 1  # c23
    
    # p5'
    U[0][11] += 1; U[1][11] += 1
    V[5][11] += 1  # b24
    W[4][11] -= 1  # c13
    W[5][11] += 1  # c14
    
    # p6'
    U[2][12] += 1; U[0][12] -= 1
    V[2][12] += 1; V[3][12] += 1  # b13, b14
    W[7][12] += 1  # c24
    
    # p7'
    U[1][13] += 1; U[3][13] -= 1
    V[5][13] += 1; V[4][13] += 1  # b23=V[5], b24=V[4]
    W[4][13] += 1  # c13
    
    # Schoolbook for B columns 4-6 (2x3 block)
    # 12 multiplications: for each (i,j,k) with i in [0,1], j in [0,1], k in [4,5,6]
    # U[i*2+j], V[j*7+k], W[k*2+i]
    
    col = 14
    for i in range(n):
        for j in range(m):
            for k in range(4, p):
                U[i * m + j][col] += 1
                V[j * p + k][col] += 1
                W[k * n + i][col] += 1
                col += 1
    
    # Verify the decomposition
    # Build the tensor from U, V, W
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # Should equal 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k, else 0
    
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    # Compute T from decomposition
    T_decomp = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    # Build expected tensor
    T_expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_expected[a, b, c] = 1
    
    # Check
    if not np.allclose(T_decomp, T_expected, atol=1e-10):
        # Fallback to schoolbook if Strassen construction is wrong
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
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
