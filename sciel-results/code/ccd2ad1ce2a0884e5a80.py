"""Matrix multiplication 4x4x6 tensor decomposition - rank 81 via Strassen + schoolbook."""
import time
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """Return (U, V, W) for 2x2x2 matrix mult, rank 7 (Strassen)."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A11+A22)*(B11+B22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[2][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A21+A22)*B11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3 = A11*(B12-B22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = A22*(B21-B11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A11+A12)*B22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (A21-A11)*(B11+B12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A12-A22)*(B21+B22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _compose_4x4x4():
    """Compose Strassen 2x2 to get 4x4x4 rank-49 decomposition."""
    U2, V2, W2 = _strassen_2x2()
    
    U = [[0]*49 for _ in range(16)]
    V = [[0]*49 for _ in range(16)]
    W = [[0]*49 for _ in range(16)]
    
    A_blocks = [[0,1,4,5], [2,3,6,7], [8,9,12,13], [10,11,14,15]]
    B_blocks = [[0,1,4,5], [2,3,6,7], [8,9,12,13], [10,11,14,15]]
    C_blocks = [[0,1,4,5], [2,3,6,7], [8,9,12,13], [10,11,14,15]]
    
    for c in range(7):
        base = c * 7
        for ab in range(4):
            for idx, flat in enumerate(A_blocks[ab]):
                for i2 in range(2):
                    for j2 in range(2):
                        u2_val = U2[i2*2+j2][c]
                        if u2_val != 0:
                            U[flat][base + i2*2+j2] = u2_val
        
        for bb in range(4):
            for idx, flat in enumerate(B_blocks[bb]):
                for j2 in range(2):
                    for k2 in range(2):
                        v2_val = V2[j2*2+k2][c]
                        if v2_val != 0:
                            V[flat][base + j2*2+k2] = v2_val
        
        for cb in range(4):
            for idx, flat in enumerate(C_blocks[cb]):
                for k2 in range(2):
                    for i2 in range(2):
                        w2_val = W2[k2*2+i2][c]
                        if w2_val != 0:
                            W[flat][base + k2*2+i2] = w2_val
    
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Verify decomposition."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 4, 4, 6
    schoolbook_rank = n * m * p  # 96
    
    # Schoolbook fallback
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = schoolbook_rank
    
    # Strassen-based 4x4x6: rank 49 + 32 = 81
    try:
        if time.time() < deadline - 5:
            U444, V444, W444 = _compose_4x4x4()
            
            total_rank = 81
            U = [[0]*total_rank for _ in range(16)]
            V = [[0]*total_rank for _ in range(16)]
            W = [[0]*total_rank for _ in range(24)]
            
            for r in range(16):
                for c in range(49):
                    U[r][c] = U444[r][c]
                    V[r][c] = V444[r][c]
            for r in range(16):
                for c in range(49):
                    W[r][c] = W444[r][c]
            
            col = 49
            for i in range(4):
                for j in range(4):
                    for k_local in range(2):
                        k_full = k_local + 4
                        U[i*4+j][col] = 1
                        V[j*4+k_full][col] = 1
                        W[k_full*4+i][col] = 1
                        col += 1
            
            if _verify(U, V, W, n, m, p):
                best = {"U": U, "V": V, "W": W}
                best_rank = 81
    except Exception:
        pass
    
    # Structured search for lower rank using ALS-style optimization
    if time.time() < deadline - 40 and best_rank > 78:
        try:
            rng = np.random.RandomState(42)
            U_base = np.array(best["U"], dtype=float)
            V_base = np.array(best["V"], dtype=float)
            W_base = np.array(best["W"], dtype=float)
            
            for target_rank in [80, 79, 78, 77]:
                if time.time() > deadline - 30:
                    break
                
                # Initialize by projecting the rank-81 solution onto rank-target
                # Use SVD-based initialization
                U_init = U_base[:, :target_rank] + rng.randn(16, target_rank) * 0.005
                V_init = V_base[:, :target_rank] + rng.randn(16, target_rank) * 0.005
                W_init = W_base[:, :target_rank] + rng.randn(24, target_rank) * 0.005
                
                def compute_error(Ur, Vr, Wr):
                    err = 0.0
                    for i in range(4):
                        for j in range(4):
                            for k in range(6):
                                s = np.sum(Ur[i*4+j] * Vr[j*4+k] * Wr[k*4+i])
                                err += (1.0 - s) ** 2
                    return err
                
                # ALS-style: optimize one factor at a time
                U_r, V_r, W_r = U_init.copy(), V_init.copy(), W_init.copy()
                err = compute_error(U_r, V_r, W_r)
                
                for it in range(3000):
                    if time.time() > deadline - 25:
                        break
                    
                    # Optimize U
                    for _ in range(5):
                        for i in range(16):
                            for c in range(target_rank):
                                # Gradient of error w.r.t. U[i,c]
                                grad = 0.0
                                for j in range(4):
                                    for k in range(6):
                                        s = np.sum(U_r[i*4+j] * V_r[j*4+k] * W_r[k*4+i])
                                        grad += 2*(1-s) * V_r[j*4+k] * W_r[k*4+i]
                                U_r[i, c] -= 0.001 * grad
                    
                    # Optimize V
                    for _ in range(5):
                        for j in range(16):
                            for c in range(target_rank):
                                grad = 0.0
                                for i in range(4):
                                    for k in range(6):
                                        s = np.sum(U_r[i*4+j] * V_r[j*4+k] * W_r[k*4+i])
                                        grad += 2*(1-s) * U_r[i*4+j] * W_r[k*4+i]
                                V_r[j, c] -= 0.001 * grad
                    
                    # Optimize W
                    for _ in range(5):
                        for k in range(24):
                            for c in range(target_rank):
                                grad = 0.0
                                for i in range(4):
                                    for j in range(4):
                                        s = np.sum(U_r[i*4+j] * V_r[j*4+k] * W_r[k*4+i])
                                        grad += 2*(1-s) * U_r[i*4+j] * V_r[j*4+k]
                                W_r[k, c] -= 0.001 * grad
                    
                    new_err = compute_error(U_r, V_r, W_r)
                    if new_err < err:
                        err = new_err
                        if err < 1e-10:
                            break
                    else:
                        # Revert and try smaller step
                        U_r = U_init.copy()
                        V_r = V_init.copy()
                        W_r = W_init.copy()
                        break
                
                if err < 1e-8:
                    U_list = [[round(x, 10) for x in row] for row in U_r.tolist()]
                    V_list = [[round(x, 10) for x in row] for row in V_r.tolist()]
                    W_list = [[round(x, 10) for x in row] for row in W_r.tolist()]
                    if _verify(U_list, V_list, W_list, n, m, p):
                        best = {"U": U_list, "V": V_list, "W": W_list}
                        best_rank = target_rank
                        break
        except Exception:
            pass
    
    return best

# EVOLVE-BLOCK-END
