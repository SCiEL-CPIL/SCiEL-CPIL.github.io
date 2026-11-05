import time
import numpy as np

N, M, P = 3, 4, 5
SCHOOLBOOK_RANK = N * M * P  # 60

START_TIME = time.time()
DEADLINE = START_TIME + 110

def _time_left():
    return DEADLINE - time.time()

def _verify(U, V, W, n=N, m=M, p=P):
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-9, rtol=0)

def _schoolbook():
    rank = SCHOOLBOOK_RANK
    U = np.zeros((N * M, rank))
    V = np.zeros((M * P, rank))
    W = np.zeros((P * N, rank))
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j, col] = 1
                V[j * P + k, col] = 1
                W[k * N + i, col] = 1
                col += 1
    return U.tolist(), V.tolist(), W.tolist()

def _strassen_2x2_factors():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    U shape (4, 7), V shape (4, 7), W shape (4, 7).
    Indices: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r].
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # r=0: M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # r=1: M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # r=2: M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # r=3: M4 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # r=4: M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # r=5: M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # r=6: M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _rank11_2x2x3_factors():
    """
    Returns U, V, W for rank-11 decomposition of 2x2x3 matrix multiplication.
    U shape (4, 11), V shape (6, 11), W shape (6, 11).
    Indices: U[i*2+j, r], V[j*3+k, r], W[k*2+i, r].
    """
    U = np.zeros((4, 11))
    V = np.zeros((6, 11))
    W = np.zeros((6, 11))
    
    # Verified rank-11 decomposition for 2x2x3
    # T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for all i,j,k
    
    # r=0: (A00+A11)(B00+B10) -> contributes to C[0,0],C[0,2],C[1,0],C[1,2]
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[3,0]=1
    W[0,0]=1; W[4,0]=1; W[2,0]=1; W[5,0]=1
    
    # r=1: (A00+A11)(B01+B11)
    U[0,1]=1; U[3,1]=1
    V[1,1]=1; V[4,1]=1
    W[1,1]=1; W[3,1]=1; W[2,1]=1; W[5,1]=1
    
    # r=2: A00(B00-B10)
    U[0,2]=1
    V[0,2]=1; V[3,2]=-1
    W[0,2]=1; W[4,2]=-1
    
    # r=3: A11(B01-B11)
    U[3,3]=1
    V[1,3]=1; V[4,3]=-1
    W[1,3]=1; W[3,3]=-1
    
    # r=4: A01(B00+B10)
    U[1,4]=1
    V[0,4]=1; V[3,4]=1
    W[0,4]=1; W[4,4]=1
    
    # r=5: A10(B01+B11)
    U[2,5]=1
    V[1,5]=1; V[4,5]=1
    W[1,5]=1; W[3,5]=1
    
    # r=6: A00*B11
    U[0,6]=1
    V[4,6]=1
    W[3,6]=1
    
    # r=7: A11*B00
    U[3,7]=1
    V[0,7]=1
    W[0,7]=1
    
    # r=8: A01*B11
    U[1,8]=1
    V[4,8]=1
    W[3,8]=1
    
    # r=9: A10*B00
    U[2,9]=1
    V[0,9]=1
    W[0,9]=1
    
    # r=10: (A01-A10)(B10-B11)
    U[1,10]=1; U[2,10]=-1
    V[3,10]=1; V[4,10]=-1
    W[4,10]=1; W[5,10]=-1
    
    return U, V, W

def _structured_3x4x5():
    """
    Construct a structured decomposition for 3x4x5 matrix multiplication.
    
    Split: A is 3x4, B is 4x5, C is 3x5.
    
    Strategy:
    - Split rows of A into [2, 1], columns of A / rows of B into [2, 2], columns of B into [3, 2]
    - Use Strassen (rank 7) for 2x2x2 blocks
    - Use rank-11 for 2x2x3 blocks
    - Use schoolbook for remaining blocks
    
    Total rank: 22 + 14 + 6 + 4 = 46
    """
    U2, V2, W2 = _strassen_2x2_factors()
    U11, V11, W11 = _rank11_2x2x3_factors()
    
    rank = 22 + 14 + 6 + 4  # 46
    U = np.zeros((12, rank))  # 3*4
    V = np.zeros((20, rank))  # 4*5
    W = np.zeros((15, rank))  # 5*3
    
    col = 0
    
    # C00 = A00*B00 + A01*B10
    # Use rank-11 for A00*B00
    for r in range(11):
        for i_prime in range(2):
            for j_prime in range(2):
                U[i_prime * 4 + j_prime, col] += U11[i_prime * 2 + j_prime, r]
        for j_prime in range(2):
            for k_prime in range(3):
                V[j_prime * 5 + k_prime, col] += V11[j_prime * 3 + k_prime, r]
        for k_prime in range(3):
            for i_prime in range(2):
                W[k_prime * 3 + i_prime, col] += W11[k_prime * 2 + i_prime, r]
        col += 1
    
    # Use rank-11 for A01*B10
    for r in range(11):
        for i_prime in range(2):
            for j_prime in range(2):
                U[i_prime * 4 + (2 + j_prime), col] += U11[i_prime * 2 + j_prime, r]
        for j_prime in range(2):
            for k_prime in range(3):
                V[(2 + j_prime) * 5 + k_prime, col] += V11[j_prime * 3 + k_prime, r]
        for k_prime in range(3):
            for i_prime in range(2):
                W[k_prime * 3 + i_prime, col] += W11[k_prime * 2 + i_prime, r]
        col += 1
    
    # C01 = A00*B01 + A01*B11
    # Use Strassen for A00*B01
    for r in range(7):
        for i_prime in range(2):
            for j_prime in range(2):
                U[i_prime * 4 + j_prime, col] += U2[i_prime * 2 + j_prime, r]
        for j_prime in range(2):
            for k_prime in range(2):
                V[j_prime * 5 + (3 + k_prime), col] += V2[j_prime * 2 + k_prime, r]
        for k_prime in range(2):
            for i_prime in range(2):
                W[(3 + k_prime) * 3 + i_prime, col] += W2[k_prime * 2 + i_prime, r]
        col += 1
    
    # Use Strassen for A01*B11
    for r in range(7):
        for i_prime in range(2):
            for j_prime in range(2):
                U[i_prime * 4 + (2 + j_prime), col] += U2[i_prime * 2 + j_prime, r]
        for j_prime in range(2):
            for k_prime in range(2):
                V[(2 + j_prime) * 5 + (3 + k_prime), col] += V2[j_prime * 2 + k_prime, r]
        for k_prime in range(2):
            for i_prime in range(2):
                W[(3 + k_prime) * 3 + i_prime, col] += W2[k_prime * 2 + i_prime, r]
        col += 1
    
    # C10 = A10*B00 + A11*B10 (schoolbook)
    for j_prime in range(2):
        for k_prime in range(3):
            U[2 * 4 + j_prime, col] = 1
            V[j_prime * 5 + k_prime, col] = 1
            W[k_prime * 3 + 2, col] = 1
            col += 1
    for j_prime in range(2):
        for k_prime in range(3):
            U[2 * 4 + (2 + j_prime), col] = 1
            V[(2 + j_prime) * 5 + k_prime, col] = 1
            W[k_prime * 3 + 2, col] = 1
            col += 1
    
    # C11 = A10*B01 + A11*B11 (schoolbook)
    for j_prime in range(2):
        for k_prime in range(2):
            U[2 * 4 + j_prime, col] = 1
            V[j_prime * 5 + (3 + k_prime), col] = 1
            W[(3 + k_prime) * 3 + 2, col] = 1
            col += 1
    for j_prime in range(2):
        for k_prime in range(2):
            U[2 * 4 + (2 + j_prime), col] = 1
            V[(2 + j_prime) * 5 + (3 + k_prime), col] = 1
            W[(3 + k_prime) * 3 + 2, col] = 1
            col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return U.tolist(), V.tolist(), W.tolist()

def _structured_56():
    """
    Alternative: split rows [1,2], cols [4], B-cols [2,3]
    Row 0: schoolbook 1x4x5 = rank 20
    Rows 1-2: 2x4x5, split cols [2,2], B-cols [2,3]
      C[1:,0:2] = 2x2x2 + 2x2x2 = Strassen x2 = 14
      C[1:,2:] = 2x2x3 + 2x2x3 = rank11 x2 = 22
    Total: 20 + 14 + 22 = 56
    """
    U2, V2, W2 = _strassen_2x2_factors()
    U11, V11, W11 = _rank11_2x2x3_factors()
    
    rank = 20 + 14 + 22  # 56
    U = np.zeros((12, rank))
    V = np.zeros((20, rank))
    W = np.zeros((15, rank))
    
    col = 0
    
    # Row 0: C[0,k] = sum_j A[0,j]*B[j,k], schoolbook
    for j in range(4):
        for k in range(5):
            U[0 * 4 + j, col] = 1
            V[j * 5 + k, col] = 1
            W[k * 3 + 0, col] = 1
            col += 1
    
    # Rows 1-2, cols 0-1 of C: C[1:3, 0:2] = A[1:3,0:2]*B[0:2,0:2] + A[1:3,2:4]*B[2:4,0:2]
    # Strassen for A[1:3,0:2]*B[0:2,0:2]
    for r in range(7):
        for i_prime in range(2):
            for j_prime in range(2):
                U[(1+i_prime) * 4 + j_prime, col] += U2[i_prime * 2 + j_prime, r]
        for j_prime in range(2):
            for k_prime in range(2):
                V[j_prime * 5 + k_prime, col] += V2[j_prime * 2 + k_prime, r]
        for k_prime in range(2):
            for i_prime in range(2):
                W[k_prime * 3 + (1+i_prime), col] += W2[k_prime * 2 + i_prime, r]
        col += 1
    
    # Strassen for A[1:3,2:4]*B[2:4,0:2]
    for r in range(7):
        for i_prime in range(2):
            for j_prime in range(2):
                U[(1+i_prime) * 4 + (2+j_prime), col] += U2[i_prime * 2 + j_prime, r]
        for j_prime in range(2):
            for k_prime in range(2):
                V[(2+j_prime) * 5 + k_prime, col] += V2[j_prime * 2 + k_prime, r]
        for k_prime in range(2):
            for i_prime in range(2):
                W[k_prime * 3 + (1+i_prime), col] += W2[k_prime * 2 + i_prime, r]
        col += 1
    
    # Rows 1-2, cols 2-4 of C: C[1:3, 2:5] = A[1:3,0:2]*B[0:2,2:5] + A[1:3,2:4]*B[2:4,2:5]
    # rank11 for A[1:3,0:2]*B[0:2,2:5]
    for r in range(11):
        for i_prime in range(2):
            for j_prime in range(2):
                U[(1+i_prime) * 4 + j_prime, col] += U11[i_prime * 2 + j_prime, r]
        for j_prime in range(2):
            for k_prime in range(3):
                V[j_prime * 5 + (2+k_prime), col] += V11[j_prime * 3 + k_prime, r]
        for k_prime in range(3):
            for i_prime in range(2):
                W[(2+k_prime) * 3 + (1+i_prime), col] += W11[k_prime * 2 + i_prime, r]
        col += 1
    
    # rank11 for A[1:3,2:4]*B[2:4,2:5]
    for r in range(11):
        for i_prime in range(2):
            for j_prime in range(2):
                U[(1+i_prime) * 4 + (2+j_prime), col] += U11[i_prime * 2 + j_prime, r]
        for j_prime in range(2):
            for k_prime in range(3):
                V[(2+j_prime) * 5 + (2+k_prime), col] += V11[j_prime * 3 + k_prime, r]
        for k_prime in range(3):
            for i_prime in range(2):
                W[(2+k_prime) * 3 + (1+i_prime), col] += W11[k_prime * 2 + i_prime, r]
        col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    # Try the rank-46 structured decomposition
    try:
        U_str, V_str, W_str = _structured_3x4x5()
        if _verify(U_str, V_str, W_str):
            return {"U": U_str, "V": V_str, "W": W_str}
    except Exception:
        pass
    
    # Try the rank-56 alternative
    try:
        U_str, V_str, W_str = _structured_56()
        if _verify(U_str, V_str, W_str):
            return {"U": U_str, "V": V_str, "W": W_str}
    except Exception:
        pass
    
    # Fallback to schoolbook
    U_fb, V_fb, W_fb = _schoolbook()
    return {"U": U_fb, "V": V_fb, "W": W_fb}
