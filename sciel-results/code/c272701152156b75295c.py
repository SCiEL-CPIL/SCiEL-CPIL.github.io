import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k; 0 otherwise
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Strategy: Kronecker product decomposition
    # 2x4x6 = (2x2x2) ⊗ (1x2x3)
    # n=2 = n1*n2 = 2*1, m=4 = m1*m2 = 2*2, p=6 = p1*p2 = 2*3
    
    # Strassen decomposition for 2x2x2 (rank 7)
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))
    
    # m1 = (A00 + A11)(B00 + B11)
    U_s[0, 0] = 1; U_s[3, 0] = 1
    V_s[0, 0] = 1; V_s[3, 0] = 1
    W_s[0, 0] = 1; W_s[3, 0] = 1
    
    # m2 = (A10 + A11)B00
    U_s[2, 1] = 1; U_s[3, 1] = 1
    V_s[0, 1] = 1
    W_s[2, 1] = 1; W_s[3, 1] = 1
    
    # m3 = A00(B10 - B11)
    U_s[0, 2] = 1
    V_s[2, 2] = 1; V_s[3, 2] = -1
    W_s[1, 2] = 1; W_s[3, 2] = 1
    
    # m4 = A11(B01 - B00)
    U_s[3, 3] = 1
    V_s[1, 3] = 1; V_s[0, 3] = -1
    W_s[0, 3] = 1; W_s[2, 3] = 1
    
    # m5 = (A00 + A01)B11
    U_s[0, 4] = 1; U_s[1, 4] = 1
    V_s[3, 4] = 1
    W_s[0, 4] = -1; W_s[1, 4] = 1
    
    # m6 = (A10 - A00)(B00 + B01)
    U_s[2, 5] = 1; U_s[0, 5] = -1
    V_s[0, 5] = 1; V_s[1, 5] = 1
    W_s[3, 5] = 1
    
    # m7 = (A01 - A11)(B10 + B11)
    U_s[1, 6] = 1; U_s[3, 6] = -1
    V_s[2, 6] = 1; V_s[3, 6] = 1
    W_s[0, 6] = 1
    
    # Verify Strassen
    T_s = np.einsum('ar,br,cr->abc', U_s, V_s, W_s)
    T_s_expected = np.zeros((4, 4, 4))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                T_s_expected[i*2+j, j*2+k, k*2+i] = 1.0
    if not np.allclose(T_s, T_s_expected):
        return _schoolbook_solution(n, m, p)
    
    # 1x2x3 decomposition (rank 2)
    # T_123[i2*m2+j2, j2*p2+k2, k2*n2+i2] = 1 if j2==k2 else 0
    # n2=1, m2=2, p2=3
    # i2 in {0}, j2 in {0,1}, k2 in {0,1,2}
    # U_123 shape: (n2*m2, rank) = (2, 2)
    # V_123 shape: (m2*p2, rank) = (6, 2)
    # W_123 shape: (p2*n2, rank) = (3, 2)
    
    U_123 = np.zeros((2, 2))
    V_123 = np.zeros((6, 2))
    W_123 = np.zeros((3, 2))
    
    # Term 0: j2=0, k2=0
    U_123[0, 0] = 1  # i2*m2+j2 = 0*2+0 = 0
    V_123[0*3+0, 0] = 1  # j2*p2+k2 = 0*3+0 = 0
    W_123[0*1+0, 0] = 1  # k2*n2+i2 = 0*1+0 = 0
    
    # Term 1: j2=1, k2=1
    U_123[1, 1] = 1  # i2*m2+j2 = 0*2+1 = 1
    V_123[1*3+1, 1] = 1  # j2*p2+k2 = 1*3+1 = 4
    W_123[1*1+0, 1] = 1  # k2*n2+i2 = 1*1+0 = 1
    
    # Verify 1x2x3
    T_123 = np.einsum('ar,br,cr->abc', U_123, V_123, W_123)
    T_123_expected = np.zeros((2, 6, 3))
    for j2 in range(2):
        for k2 in range(3):
            if j2 == k2:
                T_123_expected[0*2+j2, j2*3+k2, k2*1+0] = 1.0
    if not np.allclose(T_123, T_123_expected):
        return _schoolbook_solution(n, m, p)
    
    # Now combine using Kronecker product
    # Full indices:
    # i = i1*n2 + i2 = i1*1 + i2 = i1 + i2 (i2=0 always)
    # j = j1*m2 + j2 = j1*2 + j2
    # k = k1*p2 + k2 = k1*3 + k2
    # 
    # a = i*m + j = i*4 + j
    # b = j*p + k = j*6 + k
    # c = k*n + i = k*2 + i
    #
    # Rank = 7 * 2 = 14
    # r = r1*2 + r2
    
    total_rank = 14
    U_full = np.zeros((n*m, total_rank))
    V_full = np.zeros((m*p, total_rank))
    W_full = np.zeros((p*n, total_rank))
    
    for r1 in range(7):
        for r2 in range(2):
            r = r1 * 2 + r2
            for i1 in range(2):
                for i2 in range(1):
                    i = i1 * 1 + i2
                    for j1 in range(2):
                        for j2 in range(2):
                            j = j1 * 2 + j2
                            a = i * m + j
                            # U_full[a, r] = U_s[i1*m1+j1, r1] * U_123[i2*m2+j2, r2]
                            U_full[a, r] = U_s[i1*2+j1, r1] * U_123[i2*2+j2, r2]
            
            for j1 in range(2):
                for j2 in range(2):
                    j = j1 * 2 + j2
                    for k1 in range(2):
                        for k2 in range(3):
                            k = k1 * 3 + k2
                            b = j * p + k
                            # V_full[b, r] = V_s[j1*p1+k1, r1] * V_123[j2*p2+k2, r2]
                            V_full[b, r] = V_s[j1*2+k1, r1] * V_123[j2*3+k2, r2]
            
            for k1 in range(2):
                for k2 in range(3):
                    k = k1 * 3 + k2
                    for i1w in range(2):
                        for i2w in range(1):
                            iw = i1w * 1 + i2w
                            c = k * n + iw
                            # W_full[c, r] = W_s[k1*n1+i1w, r1] * W_123[k2*n2+i2w, r2]
                            W_full[c, r] = W_s[k1*2+i1w, r1] * W_123[k2*1+i2w, r2]
    
    # Verify full decomposition
    T_comp = np.einsum('ar,br,cr->abc', U_full, V_full, W_full)
    if np.allclose(T_comp, T, atol=1e-10):
        best_rank = total_rank
        best_U = U_full
        best_V = V_full
        best_W = W_full
    else:
        # Debug: find where it fails
        diff = np.abs(T_comp - T)
        max_diff = np.max(diff)
        if max_diff > 1e-10:
            return _schoolbook_solution(n, m, p)
        best_rank = total_rank
        best_U = U_full
        best_V = V_full
        best_W = W_full
    
    # Convert to required format
    rank = best_rank
    U_out = [[0.0] * rank for _ in range(n * m)]
    V_out = [[0.0] * rank for _ in range(m * p)]
    W_out = [[0.0] * rank for _ in range(p * n)]
    
    for a in range(n * m):
        for r in range(rank):
            U_out[a][r] = float(best_U[a, r])
    for b in range(m * p):
        for r in range(rank):
            V_out[b][r] = float(best_V[b, r])
    for c in range(p * n):
        for r in range(rank):
            W_out[c][r] = float(best_W[c, r])
    
    return {"U": U_out, "V": V_out, "W": W_out}

def _schoolbook_solution(n, m, p):
    """Fallback schoolbook solution"""
    schoolbook_rank = n * m * p
    U_sb = np.zeros((n*m, schoolbook_rank))
    V_sb = np.zeros((m*p, schoolbook_rank))
    W_sb = np.zeros((p*n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    rank = schoolbook_rank
    U_out = [[0.0] * rank for _ in range(n * m)]
    V_out = [[0.0] * rank for _ in range(m * p)]
    W_out = [[0.0] * rank for _ in range(p * n)]
    
    for a in range(n * m):
        for r in range(rank):
            U_out[a][r] = float(U_sb[a, r])
    for b in range(m * p):
        for r in range(rank):
            V_out[b][r] = float(V_sb[b, r])
    for c in range(p * n):
        for r in range(rank):
            W_out[c][r] = float(W_sb[c, r])
    
    return {"U": U_out, "V": V_out, "W": W_out}
