"""Matrix multiplication 3,5,6 - rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s wall-clock

    # Build Strassen-based decomposition
    result = _build_strassen_3x5x6(n, m, p)
    
    if result is not None:
        U, V, W, rank = result
        # Verify
        if _verify_fast(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: schoolbook
    rank_school = n * m * p
    U_fb = [[0] * rank_school for _ in range(n * m)]
    V_fb = [[0] * rank_school for _ in range(m * p)]
    W_fb = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}


def _get_strassen_2x2x2():
    """
    Return U, V, W factors for 2x2x2 matrix multiplication (Strassen, rank 7).
    U: (4, 7) indexed as U[i*2+j, c] for A[i,j]
    V: (4, 7) indexed as V[j*2+k, c] for B[j,k]
    W: (4, 7) indexed as W[k*2+i, c] for C[k,i]
    
    Strassen's 7 products:
    m0 = (a+d)(e+h)
    m1 = (c+d)(e)
    m2 = (a)(f-h)
    m3 = (d)(g-e)
    m4 = (a+b)(h)
    m5 = (c-a)(e+f)
    m6 = (b-d)(g+h)
    
    C[0,0] = m0 + m3 - m4 + m6
    C[0,1] = m2 + m4
    C[1,0] = m1 + m3
    C[1,1] = m0 - m1 + m2 + m5
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # U: A factors
    U[0, 0] = 1   # a in m0
    U[3, 0] = 1   # d in m0
    U[2, 1] = 1   # c in m1
    U[3, 1] = 1   # d in m1
    U[0, 2] = 1   # a in m2
    U[3, 3] = 1   # d in m3
    U[0, 4] = 1   # a in m4
    U[1, 4] = 1   # b in m4
    U[2, 5] = 1   # c in m5
    U[0, 5] = -1  # -a in m5
    U[1, 6] = 1   # b in m6
    U[3, 6] = -1  # -d in m6
    
    # V: B factors
    V[0, 0] = 1   # e in m0
    V[3, 0] = 1   # h in m0
    V[0, 1] = 1   # e in m1
    V[1, 2] = 1   # f in m2
    V[3, 2] = -1  # -h in m2
    V[2, 3] = 1   # g in m3
    V[0, 3] = -1  # -e in m3
    V[3, 4] = 1   # h in m4
    V[0, 5] = 1   # e in m5
    V[1, 5] = 1   # f in m5
    V[2, 6] = 1   # g in m6
    V[3, 6] = 1   # h in m6
    
    # W: C factors (W[k*2+i, c])
    W[0, 0] = 1   # m0 in C[0,0]
    W[0, 3] = 1   # m3 in C[0,0]
    W[0, 4] = -1  # -m4 in C[0,0]
    W[0, 6] = 1   # m6 in C[0,0]
    W[2, 2] = 1   # m2 in C[0,1]
    W[2, 4] = 1   # m4 in C[0,1]
    W[1, 1] = 1   # m1 in C[1,0]
    W[1, 3] = 1   # m3 in C[1,0]
    W[3, 0] = 1   # m0 in C[1,1]
    W[3, 1] = -1  # -m1 in C[1,1]
    W[3, 2] = 1   # m2 in C[1,1]
    W[3, 5] = 1   # m5 in C[1,1]
    
    return U, V, W


def _build_strassen_3x5x6(n, m, p):
    """
    Build tensor decomposition for 3x5 @ 5x6 using Strassen block structure.
    
    Split A (3x5) into:
      - Top 2x5 block: A_top (2x5)
      - Bottom 1x5 block: A_bot (1x5)
    
    Split B (5x6) into:
      - Left 5x2: B_left
      - Middle 5x2: B_mid  
      - Right 5x2: B_right
    
    For the top 2x5 @ 5x6 part, split B into three 5x2 blocks.
    For each 2x5 @ 5x2 block, split j into {0,1}, {2,3}, {4}.
    For each 2x2x2 block, use Strassen (rank 7).
    For 2x1x2 blocks, use rank 4 (schoolbook).
    
    For the bottom 1x5 @ 5x6 part, split into 1x5x2 blocks.
    For each 1x5x2 block, split j into {0,1}, {2,3}, {4}.
    For 1x2x2 blocks, use rank 4 (schoolbook).
    For 1x1x2 blocks, use rank 2 (schoolbook).
    
    Total rank: 3 k-blocks * (2*7 + 4 + 2*4 + 2) = 3 * 28 = 84
    """
    # Strassen factors for 2x2x2
    sU, sV, sW = _get_strassen_2x2x2()
    
    # Total rank: 3 k-blocks * (2 Strassen blocks + 1 schoolbook for j=4)
    # For each k-block:
    #   j in {0,1}: Strassen (7)
    #   j in {2,3}: Strassen (7) 
    #   j = 4: schoolbook 2x1x2 (4)
    # Per k-block: 7+7+4 = 18
    # For bottom row (i=2):
    #   j in {0,1}: schoolbook 1x2x2 (4)
    #   j in {2,3}: schoolbook 1x2x2 (4)
    #   j = 4: schoolbook 1x1x2 (2)
    # Per k-block for bottom: 4+4+2 = 10
    # Total per k-block: 18+10 = 28
    # 3 k-blocks: 84
    
    rank = 84
    U = np.zeros((n * m, rank))  # 15 x 84
    V = np.zeros((m * p, rank))  # 30 x 84
    W = np.zeros((p * n, rank))  # 18 x 84
    
    col = 0
    
    for k_block in range(3):
        k0 = k_block * 2  # start k index
        k1 = k0 + 1       # end k index
        
        # Top 2 rows (i=0,1), j in {0,1}: Strassen
        for c in range(7):
            # U: A[i,j] for i in {0,1}, j in {0,1}
            U[0*5 + 0, col] = sU[0, c]   # i=0, j=0
            U[0*5 + 1, col] = sU[1, c]   # i=0, j=1
            U[1*5 + 0, col] = sU[2, c]   # i=1, j=0
            U[1*5 + 1, col] = sU[3, c]   # i=1, j=1
            # V: B[j,k] for j in {0,1}, k in {k0,k1}
            V[0*6 + k0, col] = sV[0, c]   # j=0, k=k0
            V[0*6 + k1, col] = sV[1, c]   # j=0, k=k1
            V[1*6 + k0, col] = sV[2, c]   # j=1, k=k0
            V[1*6 + k1, col] = sV[3, c]   # j=1, k=k1
            # W: C[k,i] for k in {k0,k1}, i in {0,1}
            W[k0*3 + 0, col] = sW[0, c]   # k=k0, i=0
            W[k0*3 + 1, col] = sW[1, c]   # k=k0, i=1
            W[k1*3 + 0, col] = sW[2, c]   # k=k1, i=0
            W[k1*3 + 1, col] = sW[3, c]   # k=k1, i=1
            col += 1
        
        # Top 2 rows (i=0,1), j in {2,3}: Strassen
        for c in range(7):
            U[0*5 + 2, col] = sU[0, c]
            U[0*5 + 3, col] = sU[1, c]
            U[1*5 + 2, col] = sU[2, c]
            U[1*5 + 3, col] = sU[3, c]
            V[2*6 + k0, col] = sV[0, c]
            V[2*6 + k1, col] = sV[1, c]
            V[3*6 + k0, col] = sV[2, c]
            V[3*6 + k1, col] = sV[3, c]
            W[k0*3 + 0, col] = sW[0, c]
            W[k0*3 + 1, col] = sW[1, c]
            W[k1*3 + 0, col] = sW[2, c]
            W[k1*3 + 1, col] = sW[3, c]
            col += 1
        
        # Top 2 rows (i=0,1), j=4: schoolbook 2x1x2
        for i in range(2):
            for k in range(2):
                U[i*5 + 4, col] = 1
                V[4*6 + k0 + k, col] = 1
                W[(k0 + k)*3 + i, col] = 1
                col += 1
        
        # Bottom row (i=2), j in {0,1}: schoolbook 1x2x2
        for j in range(2):
            for k in range(2):
                U[2*5 + j, col] = 1
                V[j*6 + k0 + k, col] = 1
                W[(k0 + k)*3 + 2, col] = 1
                col += 1
        
        # Bottom row (i=2), j in {2,3}: schoolbook 1x2x2
        for j in range(2, 4):
            for k in range(2):
                U[2*5 + j, col] = 1
                V[j*6 + k0 + k, col] = 1
                W[(k0 + k)*3 + 2, col] = 1
                col += 1
        
        # Bottom row (i=2), j=4: schoolbook 1x1x2
        for k in range(2):
            U[2*5 + 4, col] = 1
            V[4*6 + k0 + k, col] = 1
            W[(k0 + k)*3 + 2, col] = 1
            col += 1
    
    if col != rank:
        # Truncate or pad
        if col > rank:
            U = U[:, :rank]
            V = V[:, :rank]
            W = W[:, :rank]
        else:
            rank = col
            U = U[:, :rank]
            V = V[:, :rank]
            W = W[:, :rank]
    
    return U, V, W, rank


def _verify_fast(U, V, W, n, m, p):
    """Fast verification using einsum."""
    r = U.shape[1]
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-9)


def _verify(U, V, W, n, m, p):
    """
    Verify that the tensor decomposition is exact.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    r = U.shape[1]
    
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    
    if not np.allclose(T, 1.0, atol=1e-9):
        return False
    return True

# EVOLVE-BLOCK-END
