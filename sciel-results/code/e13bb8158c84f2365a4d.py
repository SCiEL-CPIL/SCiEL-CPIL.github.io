import time
import random

N, M, P = 3, 3, 8
SCHOOLBOOK = N * M * P  # 72

def _verify(U, V, W, tol=1e-9):
    """Verify that sum_c U[a,c] V[b,c] W[c] == 1 iff a,b,c correspond to a product."""
    try:
        rank = len(U[0])
        # Check all (a, b) pairs
        for a in range(N * M):
            for b in range(M * P):
                acc = [0.0] * (P * N)
                for c in range(rank):
                    u = U[a][c]
                    if u == 0:
                        continue
                    v = V[b][c]
                    if v == 0:
                        continue
                    uv = u * v
                    if uv == 0:
                        continue
                    wc = W[c]
                    for k in range(P * N):
                        acc[k] += uv * wc[k]
                # Check if it matches the expected pattern
                i, j = divmod(a, M)  # i=row, j=col of A
                j2, k2 = divmod(b, P)  # j2=row, k2=col of B
                # For each output position (k, i) in W
                for kk in range(P * N):
                    k_out, i_out = divmod(kk, N)
                    expected = 1.0 if (i_out == i and k_out == k2) else 0.0
                    if abs(acc[kk] - expected) > tol:
                        return False
        return True
    except Exception:
        return False

def _schoolbook():
    """Return the schoolbook rank-72 decomposition."""
    rank = SCHOOLBOOK
    U = [[0] * rank for _ in range(N * M)]
    V = [[0] * rank for _ in range(M * P)]
    W = [[0] * rank for _ in range(P * N)]
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j][col] = 1
                V[j * P + k][col] = 1
                W[k * N + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2_2x2(u_base, v_base, w_base):
    """
    Add Strassen 2x2*2x2 decomposition to global U, V, W.
    u_base: [a00, a01, a10, a11] - 4 U indices
    v_base: [b00, b01, b10, b11] - 4 V indices
    w_base: [c00, c01, c10, c11] - 4 W indices
    Returns number of columns added.
    """
    nonlocal col
    a00, a01, a10, a11 = u_base
    b00, b01, b10, b11 = v_base
    c00, c01, c10, c11 = w_base
    
    # M1 = (A00+A11)(B00+B11)
    U[a00][col] += 1; U[a11][col] += 1
    V[b00][col] += 1; V[b11][col] += 1
    W[c00][col] += 1; W[c11][col] += 1
    W[c01][col] -= 1; W[c10][col] -= 1
    col += 1
    
    # M2 = (A10+A11)B00
    U[a10][col] += 1; U[a11][col] += 1
    V[b00][col] += 1
    W[c10][col] += 1; W[c11][col] -= 1
    col += 1
    
    # M3 = A00(B01-B11)
    U[a00][col] += 1
    V[b01][col] += 1; V[b11][col] -= 1
    W[c01][col] += 1; W[c11][col] += 1
    col += 1
    
    # M4 = A11(B10-B00)
    U[a11][col] += 1
    V[b10][col] += 1; V[b00][col] -= 1
    W[c00][col] += 1; W[c10][col] += 1
    col += 1
    
    # M5 = (A00+A01)B11
    U[a00][col] += 1; U[a01][col] += 1
    V[b11][col] += 1
    W[c00][col] -= 1; W[c01][col] += 1
    col += 1
    
    # M6 = (A10-A00)(B00+B01)
    U[a10][col] += 1; U[a00][col] -= 1
    V[b00][col] += 1; V[b01][col] += 1
    W[c11][col] += 1
    col += 1
    
    # M7 = (A01-A11)(B10+B11)
    U[a01][col] += 1; U[a11][col] -= 1
    V[b10][col] += 1; V[b11][col] += 1
    W[c00][col] += 1; W[c10][col] -= 1
    col += 1
    
    return 7

def _rank_68_construction():
    """
    Build a rank-68 decomposition using block decomposition.
    
    Split A (3x3) into A_top (2x3) and A_bot (1x3)
    Split B (3x8) into B1 (3x4) and B2 (3x4)
    
    C_top = A_top * B (2x8)
    C_bot = A_bot * B (1x8)
    
    For C_top:
      - A_top * B1: Split A_top into A_L (2x2) and A_R (2x1), B1 into B1_T (2x4) and B1_B (1x4)
        A_L * B1_T: 2x2 * 2x4 = two 2x2*2x2 blocks (rank 7+7=14)
        A_R * B1_B: 2x1 * 1x4 = rank 8
        Total: 22
      - A_top * B2: Similarly 22
    For C_bot:
      - A_bot * B: 1x3 * 3x8 = rank 24
      
    Total: 22 + 22 + 24 = 68
    """
    global col
    rank = 68
    U = [[0] * rank for _ in range(N * M)]
    V = [[0] * rank for _ in range(M * P)]
    W = [[0] * rank for _ in range(P * N)]
    col = 0
    
    # Index mappings:
    # U[i*M + j]: A[i,j], i=0..2, j=0..2
    # V[j*P + k]: B[j,k], j=0..2, k=0..7
    # W[k*N + i]: C[k,i], k=0..7, i=0..2
    
    # Block 1: A_L * B1_T1 (2x2 * 2x2)
    # A_L: A[0,0], A[0,1], A[1,0], A[1,1] -> U indices 0,1,2,3
    # B1_T1: B[0,0], B[0,1], B[1,0], B[1,1] -> V indices 0,1,4,5
    # C[0:2, 0:2]: W indices 0,1,3,4
    _strassen_2x2_2x2([0, 1, 2, 3], [0, 1, 4, 5], [0, 1, 3, 4])
    
    # Block 2: A_L * B1_T2 (2x2 * 2x2)
    # B1_T2: B[0,2], B[0,3], B[1,2], B[1,3] -> V indices 2,3,6,7
    # C[0:2, 2:4]: W indices 6,7,9,10
    _strassen_2x2_2x2([0, 1, 2, 3], [2, 3, 6, 7], [6, 7, 9, 10])
    
    # Block 3: A_R * B1_B (2x1 * 1x4) - rank 8
    # A_R: A[0,2], A[1,2] -> U indices 2, 5 (wait, A[0,2] is U[0*3+2]=2, A[1,2] is U[1*3+2]=5)
    # B1_B: B[2,0], B[2,1], B[2,2], B[2,3] -> V indices 16,17,18,19
    # C[0:2, 0:4]: W indices 0,1,3,4,6,7,9,10
    u_r = [2, 5]
    v_b = [16, 17, 18, 19]
    w_c = [0, 1, 3, 4, 6, 7, 9, 10]
    
    for i in range(2):
        for j in range(4):
            w_idx = w_c[i*4 + j]
            U[u_r[i]][col] = 1
            V[v_b[j]][col] = 1
            W[w_idx][col] = 1
            col += 1
    
    # Block 4: A_L * B2_T1 (2x2 * 2x2)
    # B2_T1: B[0,4], B[0,5], B[1,4], B[1,5] -> V indices 8,9,12,13
    # C[0:2, 4:6]: W indices 12,13,15,16
    _strassen_2x2_2x2([0, 1, 2, 3], [8, 9, 12, 13], [12, 13, 15, 16])
    
    # Block 5: A_L * B2_T2 (2x2 * 2x2)
    # B2_T2: B[0,6], B[0,7], B[1,6], B[1,7] -> V indices 10,11,14,15
    # C[0:2, 6:8]: W indices 18,19,21,22
    _strassen_2x2_2x2([0, 1, 2, 3], [10, 11, 14, 15], [18, 19, 21, 22])
    
    # Block 6: A_R * B2_B (2x1 * 1x4) - rank 8
    # B2_B: B[2,4], B[2,5], B[2,6], B[2,7] -> V indices 20,21,22,23
    # C[0:2, 4:8]: W indices 12,13,15,16,18,19,21,22
    u_r = [2, 5]
    v_b = [20, 21, 22, 23]
    w_c = [12, 13, 15, 16, 18, 19, 21, 22]
    
    for i in range(2):
        for j in range(4):
            w_idx = w_c[i*4 + j]
            U[u_r[i]][col] = 1
            V[v_b[j]][col] = 1
            W[w_idx][col] = 1
            col += 1
    
    # Block 7: A_bot * B (1x3 * 3x8) - rank 24
    # A_bot: A[2,0], A[2,1], A[2,2] -> U indices 6,7,8
    # B: all 24 entries -> V indices 0..23
    # C[2, 0:8]: W indices 2,5,8,11,14,17,20,23
    
    for k in range(8):
        w_idx = k * 3 + 2
        for i in range(3):
            u_idx = 6 + i
            v_idx = i * 8 + k
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
    
    if col != 68:
        return None
    
    if _verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    return None

def construct():
    start = time.time()
    deadline = start + 140.0
    
    # First try the rank-68 construction
    result = _rank_68_construction()
    if result is not None:
        return result
    
    # Fallback to schoolbook
    return _schoolbook()
