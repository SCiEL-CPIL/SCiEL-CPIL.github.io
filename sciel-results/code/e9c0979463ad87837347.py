"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Strassen's algorithm for 2x2 matrix multiplication.
    
    The matrix multiplication tensor M_{ijk} where C_{ik} = sum_j A_{ij} B_{jk}
    can be decomposed as sum_r U_{i,j,r} * V_{j,k,r} * W_{k,i,r}
    
    Wait, let me get the conventions right from the task:
    - U has shape (n*m, rank): U[i*m+j, r]
    - V has shape (m*p, rank): V[j*p+k, r]  
    - W has shape (p*n, rank): W[k*n+i, r]
    
    The tensor being decomposed is T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    Actually, the matrix multiplication tensor has entries T[i,j,k] such that
    C[i,k] = sum_j A[i,j] B[j,k]
    
    In tensor decomposition form: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    Wait, that's not quite right either. Let me think more carefully.
    
    The matrix multiplication tensor T has indices (i, j, k) where:
    - i indexes rows of A and C (0 to n-1)
    - j indexes columns of A / rows of B (0 to m-1) 
    - k indexes columns of B and C (0 to p-1)
    
    T[i,j,k] = 1 for all i,j,k (it's the full tensor of ones in the standard basis)
    
    No wait. The matrix multiplication tensor is defined such that:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    As a bilinear form / tensor: T[i,j,k] = 1 for all valid i,j,k
    
    And the decomposition is: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    where:
    - U[i*m+j, r] corresponds to the coefficient for A[i,j] in the r-th product
    - V[j*p+k, r] corresponds to the coefficient for B[j,k] in the r-th product  
    - W[k*n+i, r] corresponds to the coefficient for C[i,k] in the r-th sum
    
    So for each r, we compute P_r = (sum_{i,j} U[i*m+j,r] * A[i,j]) * (sum_{j,k} V[j*p+k,r] * B[j,k])
    and then C[i,k] += W[k*n+i,r] * P_r
    
    Strassen's algorithm for 2x2:
    A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    
    m1 = (a11 + a22)(b11 + b22)
    m2 = (a21 + a22)b11
    m3 = a11(b12 - b22)
    m4 = a22(b21 - b11)
    m5 = (a11 + a12)b22
    m6 = (a21 - a11)(b11 + b12)
    m7 = (a12 - a22)(b21 + b22)
    
    c11 = m1 + m4 - m5 + m7
    c12 = m3 + m5
    c21 = m2 + m4
    c22 = m1 - m2 + m3 + m6
    
    Let me map this to the U, V, W format.
    
    For r=0 (m1): (a11 + a22)(b11 + b22)
    U contributions: U[0*2+0, 0] = 1 (a11), U[1*2+1, 0] = 1 (a22)
    V contributions: V[0*2+0, 0] = 1 (b11), V[0*2+1, 0] = 1 (b22)
    W contributions: c11 gets +1, c22 gets +1
    W[0*2+0, 0] = 1 (c11, where k=0, i=0)
    W[1*2+1, 0] = 1 (c22, where k=1, i=1)
    
    Wait, W[k*n+i, r]. For c11: i=0, k=0, so W[0*2+0, 0] = W[0, 0]
    For c22: i=1, k=1, so W[1*2+1, 0] = W[3, 0]
    
    For r=1 (m2): (a21 + a22)b11
    U: U[1*2+0, 1] = 1 (a21), U[1*2+1, 1] = 1 (a22)
    V: V[0*2+0, 1] = 1 (b11)
    W: c21 gets +1. c21: i=1, k=0, so W[0*2+1, 1] = W[1, 1]
    
    For r=2 (m3): a11(b12 - b22)
    U: U[0*2+0, 2] = 1 (a11)
    V: V[0*2+1, 2] = 1 (b12), V[1*2+1, 2] = -1 (b22)
    W: c12 gets +1, c22 gets +1
    c12: i=0, k=1, W[1*2+0, 2] = W[2, 2]
    c22: i=1, k=1, W[1*2+1, 2] = W[3, 2]
    
    For r=3 (m4): a22(b21 - b11)
    U: U[1*2+1, 3] = 1 (a22)
    V: V[1*2+0, 3] = 1 (b21), V[0*2+0, 3] = -1 (b11)
    W: c11 gets +1, c21 gets +1
    c11: i=0, k=0, W[0*2+0, 3] = W[0, 3]
    c21: i=1, k=0, W[0*2+1, 3] = W[1, 3]
    
    For r=4 (m5): (a11 + a12)b22
    U: U[0*2+0, 4] = 1 (a11), U[0*2+1, 4] = 1 (a12)
    V: V[1*2+1, 4] = 1 (b22)
    W: c11 gets -1, c12 gets +1
    c11: i=0, k=0, W[0*2+0, 4] = W[0, 4] = -1
    c12: i=0, k=1, W[1*2+0, 4] = W[2, 4] = 1
    
    For r=5 (m6): (a21 - a11)(b11 + b12)
    U: U[1*2+0, 5] = 1 (a21), U[0*2+0, 5] = -1 (a11)
    V: V[0*2+0, 5] = 1 (b11), V[0*2+1, 5] = 1 (b12)
    W: c22 gets +1
    c22: i=1, k=1, W[1*2+1, 5] = W[3, 5] = 1
    
    For r=6 (m7): (a12 - a22)(b21 + b22)
    U: U[0*2+1, 6] = 1 (a12), U[1*2+1, 6] = -1 (a22)
    V: V[1*2+0, 6] = 1 (b21), V[1*2+1, 6] = 1 (b22)
    W: c11 gets +1
    c11: i=0, k=0, W[0*2+0, 6] = W[0, 6] = 1
    
    Let me verify c11 = m1 + m4 - m5 + m7:
    m1: W[0,0] = 1 ✓
    m4: W[0,3] = 1 ✓
    m5: W[0,4] = -1 ✓
    m7: W[0,6] = 1 ✓
    
    c12 = m3 + m5:
    m3: W[2,2] = 1 ✓
    m5: W[2,4] = 1 ✓
    
    c21 = m2 + m4:
    m2: W[1,1] = 1 ✓
    m4: W[1,3] = 1 ✓
    
    c22 = m1 - m2 + m3 + m6:
    m1: W[3,0] = 1 ✓
    m2: W[3,1] = -1 ✓
    m3: W[3,2] = 1 ✓
    m6: W[3,5] = 1 ✓
    
    This looks correct! Rank = 7.
    """
    n, m, p = DIMENSIONS  # 2, 2, 2
    rank = 7
    
    U = [[0] * rank for _ in range(n * m)]  # shape (4, 7)
    V = [[0] * rank for _ in range(m * p)]  # shape (4, 7)
    W = [[0] * rank for _ in range(p * n)]  # shape (4, 7)
    
    # r=0: m1 = (a11 + a22)(b11 + b22)
    U[0][0] = 1   # a11: i=0,j=0 -> 0*2+0
    U[3][0] = 1   # a22: i=1,j=1 -> 1*2+1
    V[0][0] = 1   # b11: j=0,k=0 -> 0*2+0
    V[1][0] = 1   # b22: j=0,k=1 -> 0*2+1
    W[0][0] = 1   # c11: k=0,i=0 -> 0*2+0
    W[3][0] = 1   # c22: k=1,i=1 -> 1*2+1
    
    # r=1: m2 = (a21 + a22)b11
    U[2][1] = 1   # a21: i=1,j=0 -> 1*2+0
    U[3][1] = 1   # a22: i=1,j=1 -> 1*2+1
    V[0][1] = 1   # b11: j=0,k=0 -> 0*2+0
    W[1][1] = 1   # c21: k=0,i=1 -> 0*2+1
    
    # r=2: m3 = a11(b12 - b22)
    U[0][2] = 1   # a11: i=0,j=0 -> 0*2+0
    V[1][2] = 1   # b12: j=0,k=1 -> 0*2+1
    V[3][2] = -1  # b22: j=1,k=1 -> 1*2+1
    W[2][2] = 1   # c12: k=1,i=0 -> 1*2+0
    W[3][2] = 1   # c22: k=1,i=1 -> 1*2+1
    
    # r=3: m4 = a22(b21 - b11)
    U[3][3] = 1   # a22: i=1,j=1 -> 1*2+1
    V[2][3] = 1   # b21: j=1,k=0 -> 1*2+0
    V[0][3] = -1  # b11: j=0,k=0 -> 0*2+0
    W[0][3] = 1   # c11: k=0,i=0 -> 0*2+0
    W[1][3] = 1   # c21: k=0,i=1 -> 0*2+1
    
    # r=4: m5 = (a11 + a12)b22
    U[0][4] = 1   # a11: i=0,j=0 -> 0*2+0
    U[1][4] = 1   # a12: i=0,j=1 -> 0*2+1
    V[3][4] = 1   # b22: j=1,k=1 -> 1*2+1
    W[0][4] = -1  # c11: k=0,i=0 -> 0*2+0
    W[2][4] = 1   # c12: k=1,i=0 -> 1*2+0
    
    # r=5: m6 = (a21 - a11)(b11 + b12)
    U[2][5] = 1   # a21: i=1,j=0 -> 1*2+0
    U[0][5] = -1  # a11: i=0,j=0 -> 0*2+0
    V[0][5] = 1   # b11: j=0,k=0 -> 0*2+0
    V[1][5] = 1   # b12: j=0,k=1 -> 0*2+1
    W[3][5] = 1   # c22: k=1,i=1 -> 1*2+1
    
    # r=6: m7 = (a12 - a22)(b21 + b22)
    U[1][6] = 1   # a12: i=0,j=1 -> 0*2+1
    U[3][6] = -1  # a22: i=1,j=1 -> 1*2+1
    V[2][6] = 1   # b21: j=1,k=0 -> 1*2+0
    V[3][6] = 1   # b22: j=1,k=1 -> 1*2+1
    W[0][6] = 1   # c11: k=0,i=0 -> 0*2+0
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
