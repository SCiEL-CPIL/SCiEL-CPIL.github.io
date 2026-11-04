"""Rewritten program for matrix multiplication tensor decomposition 4x4x7."""
import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145

    def make_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best_result = make_schoolbook()
    best_rank = n * m * p

    # Precompute index arrays
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    u_idx = i_idx * m + j_idx
    v_idx = j_idx * p + k_idx
    w_idx = k_idx * n + i_idx

    def verify(U, V, W, r):
        Ua = np.asarray(U, dtype=float)
        Va = np.asarray(V, dtype=float)
        Wa = np.asarray(W, dtype=float)
        u_vals = Ua[u_idx, :]
        v_vals = Va[v_idx, :]
        w_vals = Wa[w_idx, :]
        result = (u_vals * v_vals * w_vals).sum(axis=1)
        return float(np.max(np.abs(result - 1.0)))

    def strassen_2x2x2():
        U = np.zeros((4, 7)); V = np.zeros((4, 7)); W = np.zeros((4, 7))
        U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[1,0]=1; W[2,0]=1; W[3,0]=1
        U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1; W[3,1]=1
        U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[1,2]=1; W[3,2]=1
        U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[2,3]=1; W[0,3]=1
        U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=1; W[1,4]=1
        U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[2,5]=1; W[3,5]=1
        U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[1,6]=1; W[3,6]=1
        return U, V, W

    def kronecker_decomp(U1, V1, W1, U2, V2, W2, n1, m1, p1, n2, m2, p2):
        r1 = W1.shape[1]; r2 = W2.shape[1]; r = r1 * r2
        nn = n1*n2; mm = m1*m2; pp = p1*p2
        U = np.zeros((nn*mm, r)); V = np.zeros((mm*pp, r)); W = np.zeros((pp*nn, r))
        for c1 in range(r1):
            for c2 in range(r2):
                c = c1*r2 + c2
                for i1 in range(n1):
                    for j1 in range(m1):
                        v1 = U1[i1*m1+j1, c1]
                        if v1 == 0: continue
                        for i2 in range(n2):
                            for j2 in range(m2):
                                v2 = U2[i2*m2+j2, c2]
                                if v2 == 0: continue
                                row = (i1*n2+i2)*(mm) + (j1*m2+j2)
                                U[row, c] += v1*v2
                for j1 in range(m1):
                    for k1 in range(p1):
                        v1 = V1[j1*p1+k1, c1]
                        if v1 == 0: continue
                        for j2 in range(m2):
                            for k2 in range(p2):
                                v2 = V2[j2*p2+k2, c2]
                                if v2 == 0: continue
                                row = (j1*m2+j2)*(pp) + (k1*p2+k2)
                                V[row, c] += v1*v2
                for k1 in range(p1):
                    for i1 in range(n1):
                        v1 = W1[k1*n1+i1, c1]
                        if v1 == 0: continue
                        for k2 in range(p2):
                            for i2 in range(n2):
                                v2 = W2[k2*n2+i2, c2]
                                if v2 == 0: continue
                                row = (k1*p2+k2)*(nn) + (i1*n2+i2)
                                W[row, c] += v1*v2
        return U, V, W

    def schoolbook_2x2x1():
        U = np.zeros((4,2)); V = np.zeros((2,2)); W = np.zeros((2,2))
        U[0,0]=1; V[0,0]=1; W[0,0]=1
        U[2,1]=1; V[0,1]=1; W[1,1]=1
        return U, V, W

    def schoolbook_4x4x1():
        U = np.zeros((16,4)); V = np.zeros((4,4)); W = np.zeros((4,4))
        for j in range(4):
            for i in range(4):
                U[i*4+j, j] = 1
            V[j,j] = 1
            W[i,i] = 1
        return U, V, W

    def schoolbook_4x4x3():
        """4x4x3 schoolbook, rank 48."""
        U = np.zeros((16, 48)); V = np.zeros((12, 48)); W = np.zeros((12, 48))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(3):
                    U[i*4+j, col] = 1
                    V[j*3+k, col] = 1
                    W[k*4+i, col] = 1
                    col += 1
        return U, V, W

    # Strategy 1: Block decomposition [2,2,2,1] -> rank 46
    def construct_block_decomp():
        U_s, V_s, W_s = strassen_2x2x2()
        U_sb, V_sb, W_sb = schoolbook_2x2x1()
        U_442, V_442, W_442 = kronecker_decomp(U_s, V_s, W_s, U_sb, V_sb, W_sb, 2,2,2, 2,2,1)
        U_441, V_441, W_441 = schoolbook_4x4x1()
        total_rank = 3*14 + 4
        U = np.zeros((16, total_rank)); V = np.zeros((28, total_rank)); W = np.zeros((28, total_rank))
        co = 0
        U[:, co:co+14] = U_442; V[0:8, co:co+14] = V_442; W[0:8, co:co+14] = W_442; co += 14
        U[:, co:co+14] = U_442; V[8:16, co:co+14] = V_442; W[8:16, co:co+14] = W_442; co += 14
        U[:, co:co+14] = U_442; V[16:24, co:co+14] = V_442; W[16:24, co:co+14] = W_442; co += 14
        U[:, co:co+4] = U_441; V[24:28, co:co+4] = V_441; W[24:28, co:co+4] = W_441; co += 4
        return U, V, W, total_rank

    # Strategy 2: 4x4x4 (Strassen rank 49) + 4x4x3 (schoolbook rank 48) = 97
    def construct_strassen_plus_schoolbook():
        U_s, V_s, W_s = strassen_2x2x2()
        U_sb, V_sb, W_sb = schoolbook_2x2x1()
        # 4x4x2 = Strassen(2x2x2) ⊗ schoolbook(2x2x1), rank 14
        U_442, V_442, W_442 = kronecker_decomp(U_s, V_s, W_s, U_sb, V_sb, W_sb, 2,2,2, 2,2,1)
        # 4x4x4 = Strassen(2x2x2) ⊗ Strassen(2x2x2), rank 49
        U_444, V_444, W_444 = kronecker_decomp(U_s, V_s, W_s, U_s, V_s, W_s, 2,2,2, 2,2,2)
        # 4x4x3 schoolbook
        U_443, V_443, W_443 = schoolbook_4x4x3()
        total_rank = 49 + 48
        U = np.zeros((16, total_rank)); V = np.zeros((28, total_rank)); W = np.zeros((28, total_rank))
        U[:, 0:49] = U_444; V[0:16, 0:49] = V_444; W[0:16, 0:49] = W_444
        U[:, 49:97] = U_443; V[16:28, 49:97] = V_443; W[16:28, 49:97] = W_443
        return U, V, W, total_rank

    # Strategy 3: [2,2,2,1] but try [4,3] split with Strassen for 4x4x4
    # Actually try: 4x4x4 via Strassen (49) + 4x4x3 optimized
    # For now, just use the block decomp which should give 46

    # Try block decomposition first (best expected rank)
    try:
        U_b, V_b, W_b, r_b = construct_block_decomp()
        err = verify(U_b, V_b, W_b, r_b)
        if err < 1e-8:
            best_rank = r_b
            best_result = {"U": U_b.tolist(), "V": V_b.tolist(), "W": W_b.tolist()}
    except Exception:
        pass

    # If block decomp failed, try Strassen+schoolbook
    if best_rank == n*m*p:
        try:
            U_s2, V_s2, W_s2, r_s2 = construct_strassen_plus_schoolbook()
            err = verify(U_s2, V_s2, W_s2, r_s2)
            if err < 1e-8:
                best_rank = r_s2
                best_result = {"U": U_s2.tolist(), "V": V_s2.tolist(), "W": W_s2.tolist()}
        except Exception:
            pass

    # Now try to reduce rank below best_rank using structured optimization
    if best_rank < n*m*p:
        U_init = np.array(best_result["U"], dtype=float)
        V_init = np.array(best_result["V"], dtype=float)
        W_init = np.array(best_result["W"], dtype=float)

        for target_rank in range(max(20, best_rank - 15), best_rank):
            if time.time() > deadline - 20:
                break
            found = False
            for trial in range(8):
                if time.time() > deadline - 15:
                    break
                np.random.seed(trial * 7919 + target_rank * 131)
                
                if trial == 0 and target_rank <= U_init.shape[1]:
                    U = U_init[:, :target_rank].copy()
                    V = V_init[:, :target_rank].copy()
                    W = W_init[:, :target_rank].copy()
                elif trial == 1 and target_rank <= U_init.shape[1]:
                    # Random column selection
                    cols = np.random.choice(U_init.shape[1], size=target_rank, replace=False)
                    U = U_init[:, cols].copy()
                    V = V_init[:, cols].copy()
                    W = W_init[:, cols].copy()
                else:
                    # Structured random init: sparse with small integers
                    U = np.zeros((n*m, target_rank))
                    V = np.zeros((m*p, target_rank))
                    W = np.zeros((p*n, target_rank))
                    for c in range(target_rank):
                        # Pick random support
                        u_rows = np.random.choice(n*m, size=np.random.randint(2,5), replace=False)
                        v_rows = np.random.choice(m*p, size=np.random.randint(2,5), replace=False)
                        w_rows = np.random.choice(p*n, size=np.random.randint(2,5), replace=False)
                        U[u_rows, c] = np.random.choice([-1, 1], size=len(u_rows))
                        V[v_rows, c] = np.random.choice([-1, 1], size=len(v_rows))
                        W[w_rows, c] = np.random.choice([-1, 1], size=len(w_rows))
                
                # Iterative refinement
                for it in range(300):
                    if time.time() > deadline - 10:
                        break
                    u_vals = U[u_idx, :]
                    v_vals = V[v_idx, :]
                    w_vals = W[w_idx, :]
                    prod = u_vals * v_vals * w_vals
                    result = prod.sum(axis=1)
                    error = result - 1.0
                    err_norm = np.linalg.norm(error)
                    if err_norm < 1e-10:
                        break
                    
                    grad_U = np.zeros_like(U)
                    grad_V = np.zeros_like(V)
                    grad_W = np.zeros_like(W)
                    
                    for i in range(n):
                        for j in range(m):
                            u_row = U[i*m+j, :]
                            for k in range(p):
                                e = error[i*m*p + j*p + k]
                                if abs(e) > 1e-12:
                                    v_row = V[j*p+k, :]
                                    w_row = W[k*n+i, :]
                                    grad_U[i*m+j, :] += e * v_row * w_row
                                    grad_V[j*p+k, :] += e * u_row * w_row
                                    grad_W[k*n+i, :] += e * u_row * v_row
                    
                    grad_norm = np.sqrt(np.sum(grad_U**2) + np.sum(grad_V**2) + np.sum(grad_W**2))
                    if grad_norm < 1e-15:
                        break
                    lr = min(0.5, 1.0/(1.0 + grad_norm))
                    U -= lr * grad_U
                    V -= lr * grad_V
                    W -= lr * grad_W
                
                err = verify(U, V, W, target_rank)
                if err < 1e-8:
                    best_rank = target_rank
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    found = True
                    break
            if found:
                continue
            else:
                break

    return best_result
# EVOLVE-BLOCK-END
