import time
import numpy as np

def construct():
    start = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill
    n, m, p = 4, 4, 7

    def _strassen_2x2x2():
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        # p1 = (a11+a22)(b11+b22)
        U[0][0]=1; U[3][0]=1; V[0][0]=1; V[3][0]=1; W[0][0]=1; W[3][0]=1
        # p2 = (a21+a22)b11
        U[2][1]=1; U[3][1]=1; V[0][1]=1; W[1][1]=1; W[3][1]=-1
        # p3 = a11(b12-b22)
        U[0][2]=1; V[1][2]=1; V[3][2]=-1; W[2][2]=1; W[3][2]=1
        # p4 = a22(b21-b11)
        U[3][3]=1; V[2][3]=1; V[0][3]=-1; W[0][3]=1; W[1][3]=1
        # p5 = (a11+a12)b22
        U[0][4]=1; U[1][4]=1; V[3][4]=1; W[0][4]=-1; W[2][4]=1
        # p6 = (a21-a11)(b11+b12)
        U[2][5]=1; U[0][5]=-1; V[0][5]=1; V[1][5]=1; W[3][5]=1
        # p7 = (a12-a22)(b21+b22)
        U[1][6]=1; U[3][6]=-1; V[2][6]=1; V[3][6]=1; W[0][6]=1
        return U, V, W

    def _kron_4x4x4():
        U2, V2, W2 = _strassen_2x2x2()
        U = [[0]*49 for _ in range(16)]
        V = [[0]*49 for _ in range(16)]
        W = [[0]*49 for _ in range(16)]
        for r1 in range(7):
            for r2 in range(7):
                r = r1*7 + r2
                for i1 in range(2):
                    for j1 in range(2):
                        for i2 in range(2):
                            for j2 in range(2):
                                i = i1*2+i2; j = j1*2+j2
                                v = U2[i1*2+j1][r1] * U2[i2*2+j2][r2]
                                if v: U[i*4+j][r] += v
                for j1 in range(2):
                    for k1 in range(2):
                        for j2 in range(2):
                            for k2 in range(2):
                                j = j1*2+j2; k = k1*2+k2
                                v = V2[j1*2+k1][r1] * V2[j2*2+k2][r2]
                                if v: V[j*4+k][r] += v
                for k1 in range(2):
                    for i1 in range(2):
                        for k2 in range(2):
                            for i2 in range(2):
                                k = k1*2+k2; i = i1*2+i2
                                v = W2[k1*2+i1][r1] * W2[k2*2+i2][r2]
                                if v: W[k*4+i][r] += v
        return U, V, W

    def _build_base():
        U2, V2, W2 = _strassen_2x2x2()
        # Part 1: 4x4 * 4x4 (B cols 0-3, C cols 0-3), rank 49
        U1, V1, W1 = _kron_4x4x4()
        # Part 2: 4x4 * 4x2 (B cols 4-5, C cols 4-5), rank 28
        rank2 = 28
        U2p = [[0]*rank2 for _ in range(16)]
        V2p = [[0]*rank2 for _ in range(8)]
        W2p = [[0]*rank2 for _ in range(8)]
        for br in range(2):
            for bc in range(2):
                for t in range(7):
                    r = (br*2+bc)*7 + t
                    for lr in range(2):
                        for lc in range(2):
                            i = br*2+lr; j = bc*2+lc
                            v = U2[lr*2+lc][t]
                            if v: U2p[i*4+j][r] += v
                    for lr in range(2):
                        for kl in range(2):
                            j = bc*2+lr; k = 4+kl
                            v = V2[lr*2+kl][t]
                            if v: V2p[j*2+k-4][r] += v
                    for kl in range(2):
                        for li in range(2):
                            k = 4+kl; i = br*2+li
                            v = W2[kl*2+li][t]
                            if v: W2p[k-4+4*li][r] += v
        # Part 3: 4x4 * 4x1 (B col 6, C col 6), rank 16
        rank3 = 16
        U3 = [[0]*rank3 for _ in range(16)]
        V3 = [[0]*rank3 for _ in range(4)]
        W3 = [[0]*rank3 for _ in range(4)]
        for i in range(4):
            for j in range(4):
                r = i*4+j
                U3[i*4+j][r] = 1
                V3[j*1+0][r] = 1
                W3[0*4+i][r] = 1

        # Assemble: total rank = 49+28+16 = 93
        total_rank = 93
        U = [[0]*total_rank for _ in range(16)]
        V = [[0]*total_rank for _ in range(28)]
        W = [[0]*total_rank for _ in range(28)]
        # Copy part 1
        for i in range(16):
            for r in range(49):
                U[i][r] = U1[i][r]
        for j in range(4):
            for k in range(4):
                for r in range(49):
                    V[j*7+k][r] = V1[j*4+k][r]
        for k in range(4):
            for i in range(4):
                for r in range(49):
                    W[k*4+i][r] = W1[k*4+i][r]
        # Copy part 2
        off2 = 49
        for i in range(16):
            for r in range(28):
                U[i][off2+r] = U2p[i][r]
        for j in range(4):
            for k in range(2):
                for r in range(28):
                    V[j*7+4+k][off2+r] = V2p[j*2+k][r]
        for k in range(2):
            for i in range(4):
                for r in range(28):
                    W[(4+k)*4+i][off2+r] = W2p[k*4+i][r]
        # Copy part 3
        off3 = 77
        for i in range(4):
            for j in range(4):
                r = off3 + i*4+j
                U[i*4+j][r] = 1
                V[j*7+6][r] = 1
                W[6*4+i][r] = 1
        return U, V, W

    def _verify_np(U_np, V_np, W_np, tol=1e-9):
        r = U_np.shape[1]
        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # Vectorized: for each c, outer product structure
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_np[i*m+j], V_np[j*p+k] * W_np[k*n+i])
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def _verify_fast(U_np, V_np, W_np, tol=1e-9):
        """Faster verification using matrix operations."""
        r = U_np.shape[1]
        # For each (i,k), we need: sum_j sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all j... no
        # Actually T[i,j,k] must equal 1 for all i,j,k
        # Let's check all 112 entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_np[i*m+j], V_np[j*p+k] * W_np[k*n+i])
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def _try_merge(U_np, V_np, W_np, c1, c2, alpha, beta):
        """Try merging column c2 into c1 with coefficients alpha, beta. Returns (new_U, new_V, new_W) or None."""
        r = U_np.shape[1]
        if c1 == c2:
            return None
        U_new = U_np.copy()
        V_new = V_np.copy()
        W_new = W_np.copy()
        U_new[:, c1] = alpha * U_new[:, c1] + beta * U_new[:, c2]
        V_new[:, c1] = alpha * V_new[:, c1] + beta * V_new[:, c2]
        W_new[:, c1] = alpha * W_new[:, c1] + beta * W_new[:, c2]
        mask = [i for i in range(r) if i != c2]
        U_new = U_new[:, mask]
        V_new = V_new[:, mask]
        W_new = W_new[:, mask]
        return U_new, V_new, W_new

    # Build base decomposition
    U, V, W = _build_base()
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    # Verify base
    assert _verify_fast(U_np, V_np, W_np), "Base decomposition is invalid!"

    best_rank = 93
    best_U, best_V, best_W = U_np.copy(), V_np.copy(), W_np.copy()

    # Rank reduction search
    # Strategy: try to find pairs of columns where one factor is a scalar multiple
    # or where a linear combination eliminates one column while preserving validity

    import random
    random.seed(42)

    r = best_rank
    iterations = 0
    while time.time() - start < DEADLINE and r > 1:
        iterations += 1
        # Try random merges
        found = False
        for attempt in range(200):
            if time.time() - start > DEADLINE:
                break
            c1 = random.randint(0, r-1)
            c2 = random.randint(0, r-1)
            if c1 == c2:
                continue
            # Try several coefficient combinations
            for alpha, beta in [(1.0, 1.0), (1.0, -1.0), (1.0, 0.5), (1.0, -0.5),
                                (0.5, 1.0), (-0.5, 1.0), (1.0, 0.0), (0.0, 1.0)]:
                result = _try_merge(best_U, best_V, best_W, c1, c2, alpha, beta)
                if result is None:
                    continue
                U_new, V_new, W_new = result
                if _verify_fast(U_new, V_new, W_new):
                    best_U, best_V, best_W = U_new, V_new, W_new
                    r -= 1
                    best_rank = r
                    found = True
                    break
            if found:
                break
        if not found and iterations > 50:
            # No more simple merges found, break
            break

    # Convert back to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
