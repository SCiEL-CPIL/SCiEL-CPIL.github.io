import time
import numpy as np

def construct():
    n, m, p = 3, 4, 4
    dim_u = n * m  # 12
    dim_v = m * p  # 16
    dim_w = p * n  # 12

    start_time = time.time()
    deadline = start_time + 135  # safety margin vs 150s kill

    def build_rank44():
        rank = 44
        U = [[0] * rank for _ in range(dim_u)]
        V = [[0] * rank for _ in range(dim_v)]
        W = [[0] * rank for _ in range(dim_w)]

        col = 0
        def add_strassen(X_u, Y_v, Z_w, col):
            # M1 = (X00+X11)(Y00+Y11)
            U[X_u[0]][col] += 1; U[X_u[3]][col] += 1
            V[Y_v[0]][col] += 1; V[Y_v[3]][col] += 1
            W[Z_w[0]][col] += 1; W[Z_w[3]][col] += 1
            col += 1
            # M2 = (X10+X11)Y00
            U[X_u[2]][col] += 1; U[X_u[3]][col] += 1
            V[Y_v[0]][col] += 1
            W[Z_w[2]][col] += 1; W[Z_w[3]][col] -= 1
            col += 1
            # M3 = X00(Y01-Y11)
            U[X_u[0]][col] += 1
            V[Y_v[1]][col] += 1; V[Y_v[3]][col] -= 1
            W[Z_w[1]][col] += 1; W[Z_w[3]][col] += 1
            col += 1
            # M4 = X11(Y10-Y00)
            U[X_u[3]][col] += 1
            V[Y_v[2]][col] += 1; V[Y_v[0]][col] -= 1
            W[Z_w[0]][col] += 1; W[Z_w[2]][col] += 1
            col += 1
            # M5 = (X00+X01)Y11
            U[X_u[0]][col] += 1; U[X_u[1]][col] += 1
            V[Y_v[3]][col] += 1
            W[Z_w[0]][col] -= 1; W[Z_w[1]][col] += 1
            col += 1
            # M6 = (X10-X00)(Y00+Y01)
            U[X_u[2]][col] += 1; U[X_u[0]][col] -= 1
            V[Y_v[0]][col] += 1; V[Y_v[1]][col] += 1
            W[Z_w[3]][col] += 1
            col += 1
            # M7 = (X01-X11)(Y10+Y11)
            U[X_u[1]][col] += 1; U[X_u[3]][col] -= 1
            V[Y_v[2]][col] += 1; V[Y_v[3]][col] += 1
            W[Z_w[0]][col] += 1
            col += 1
            return col

        # 4 Strassen blocks (2x2x2) covering the top 2 rows of A
        col = add_strassen([0,1,4,5], [0,1,4,5], [0,3,1,4], col)
        col = add_strassen([2,3,6,7], [8,9,12,13], [0,3,1,4], col)
        col = add_strassen([0,1,4,5], [2,3,6,7], [6,9,7,10], col)
        col = add_strassen([2,3,6,7], [10,11,14,15], [6,9,7,10], col)

        # Bottom row of A (1x2x2 schoolbook): A_bot * B_left -> C[2,0], C[2,1]
        for j in range(4):
            U[8+j][col] += 1; V[j*4+0][col] += 1; W[2][col] += 1
            col += 1
            U[8+j][col] += 1; V[j*4+1][col] += 1; W[5][col] += 1
            col += 1
        # Bottom row of A: A_bot * B_right -> C[2,2], C[2,3]
        for j in range(4):
            U[8+j][col] += 1; V[j*4+2][col] += 1; W[8][col] += 1
            col += 1
            U[8+j][col] += 1; V[j*4+3][col] += 1; W[11][col] += 1
            col += 1

        return U, V, W

    def fast_verify(U, V, W):
        """Check all 48 'on' tensor positions sum to 1 (trilinear)."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], Va[j*p+k] * Wa[k*n+i])
                    if abs(val - 1.0) > 1e-8:
                        return False
        return True

    def verify_full(U, V, W):
        T_expected = np.zeros((dim_u, dim_v, dim_w))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_expected[i*m+j, j*p+k, k*n+i] = 1.0
        T_actual = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.allclose(T_actual, T_expected, atol=1e-8)

    def nullspace(M, tol=1e-9):
        """Return a basis (rows) for the nullspace of M (shape d x rank)."""
        try:
            _, s, vh = np.linalg.svd(M, full_matrices=True)
        except np.linalg.LinAlgError:
            return np.zeros((0, M.shape[1]))
        rank = int(np.sum(s > tol))
        if rank >= M.shape[1]:
            return np.zeros((0, M.shape[1]))
        return vh[rank:]  # (d-rank) x rank

    # Build rank-44
    U0, V0, W0 = build_rank44()
    U = np.array(U0, dtype=float)
    V = np.array(V0, dtype=float)
    W = np.array(W0, dtype=float)

    # Fast verify the base construction
    Va, Wa = V, W
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                if abs(val - 1.0) > 1e-8:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break
    if not ok:
        # Fallback: schoolbook rank 48
        rank = 48
        U = np.zeros((dim_u, rank)); V = np.zeros((dim_v, rank)); W = np.zeros((dim_w, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1; V[j*p+k, col] = 1; W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # --- Rank reduction ---
    # (a) Linear-dependence (nullspace) elimination: if sum_r lam_r (u_r,v_r,w_r)=0
    #     then the trilinear tensor is invariant to removing any column in the combo.
    current_U, current_V, current_W = U, V, W
    cur_rank = current_U.shape[1]
    for _ in range(12):
        if time.time() > deadline:
            break
        if cur_rank <= 1:
            break
        M = np.vstack([current_U, current_V, current_W])  # (40, rank)
        basis = nullspace(M)
        if basis.shape[0] == 0:
            break
        # pick a column with large coefficient in some null vector
        best = None
        for nv in basis:
            amax = int(np.argmax(np.abs(nv)))
            if best is None or abs(nv[amax]) > abs(best[1]):
                best = (amax, nv[amax])
        r_idx = best[0]
        # remove column r_idx
        mask = [c for c in range(cur_rank) if c != r_idx]
        current_U = current_U[:, mask]
        current_V = current_V[:, mask]
        current_W = current_W[:, mask]
        cur_rank -= 1

    # (b) Redundant-column (zero-sum) elimination
    for attempt in range(30):
        if time.time() > deadline:
            break
        if cur_rank <= 1:
            break
        found = False
        for r_idx in range(cur_rank):
            if time.time() > deadline:
                break
            mask = [c for c in range(cur_rank) if c != r_idx]
            Ut = current_U[:, mask]; Vt = current_V[:, mask]; Wt = current_W[:, mask]
            # quick on-position check
            good = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if abs(np.dot(Ut[i*m+j], Vt[j*p+k] * Wt[k*n+i]) - 1.0) > 1e-8:
                            good = False
                            break
                    if not good:
                        break
                if not good:
                    break
            if good:
                current_U, current_V, current_W = Ut, Vt, Wt
                cur_rank -= 1
                found = True
                break
        if not found:
            break

    # Final full verification before accepting a reduced rank
    if cur_rank < 44:
        if verify_full(current_U, current_V, current_W):
            return {"U": current_U.tolist(), "V": current_V.tolist(), "W": current_W.tolist()}

    return {"U": U0, "V": V0, "W": W0}
