import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 20, 30, 24

    # Build the target tensor T
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify_decomposition(U, V, W):
        r = U.shape[1]
        T_recon = np.zeros((nm, mp, pn))
        for t in range(r):
            uv = np.outer(U[:, t], V[:, t])
            T_recon += np.einsum('ab,c->abc', uv, W[:, t])
        return np.max(np.abs(T_recon - T))

    # Schoolbook decomposition (rank 120)
    def schoolbook():
        r = n * m * p
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        W = np.zeros((pn, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    U_sb, V_sb, W_sb = schoolbook()
    best_rank = n * m * p
    best_decomp = (U_sb, V_sb, W_sb)

    # Strassen-3: rank-7 decomposition for 3x3 * 3x3 (corrected m6)
    def strassen_3x3():
        r = 7
        U = np.zeros((9, r))
        V = np.zeros((9, r))
        W = np.zeros((9, r))
        # U idx: i*3+j ; V idx: j*3+k ; W idx: k*3+i
        # m1 = x00*(y11 - y12 - y21 + y22)
        t = 0
        U[0, t] = 1
        V[1*3+1, t] = 1; V[1*3+2, t] = -1; V[2*3+1, t] = -1; V[2*3+2, t] = 1
        W[0*3+0, t] = 1; W[0*3+1, t] = 1; W[0*3+2, t] = 1
        # m2 = (x00+x01)*(y12 + y22 - y11 - y21)
        t = 1
        U[0, t] = 1; U[1, t] = 1
        V[1*3+2, t] = 1; V[2*3+2, t] = 1; V[1*3+1, t] = -1; V[2*3+1, t] = -1
        W[1*3+0, t] = 1; W[1*3+1, t] = 1; W[1*3+2, t] = 1
        # m3 = (x01+x02)*(y10 + y20 - y11 - y21)
        t = 2
        U[1, t] = 1; U[2, t] = 1
        V[1*3+0, t] = 1; V[2*3+0, t] = 1; V[1*3+1, t] = -1; V[2*3+1, t] = -1
        W[2*3+0, t] = 1; W[2*3+1, t] = 1; W[2*3+2, t] = 1
        # m4 = x22*(y00 - y01 - y10 + y11)
        t = 3
        U[8, t] = 1
        V[0*3+0, t] = 1; V[0*3+1, t] = -1; V[1*3+0, t] = -1; V[1*3+1, t] = 1
        W[0*3+0, t] = 1; W[1*3+0, t] = 1; W[2*3+0, t] = 1
        # m5 = (x20+x22)*(y01 + y11 - y00 - y10)
        t = 4
        U[6, t] = 1; U[8, t] = 1
        V[0*3+1, t] = 1; V[1*3+1, t] = 1; V[0*3+0, t] = -1; V[1*3+0, t] = -1
        W[0*3+1, t] = 1; W[1*3+1, t] = 1; W[2*3+1, t] = 1
        # m6 = (x21+x22)*(y00 + y10 - y01 - y10)   [CORRECTED]
        t = 5
        U[7, t] = 1; U[8, t] = 1
        V[0*3+0, t] = 1; V[1*3+0, t] = 1; V[0*3+1, t] = -1; V[1*3+0, t] = -1
        W[0*3+2, t] = 1; W[1*3+2, t] = 1; W[2*3+2, t] = 1
        # m7 = (x00+x11+x22)*(y00+y11+y22)
        t = 6
        U[0, t] = 1; U[4, t] = 1; U[8, t] = 1
        V[0*3+0, t] = 1; V[1*3+1, t] = 1; V[2*3+2, t] = 1
        W[0*3+0, t] = 1; W[1*3+0, t] = 1; W[2*3+0, t] = 1
        W[0*3+1, t] = 1; W[1*3+1, t] = 1; W[2*3+1, t] = 1
        W[0*3+2, t] = 1; W[1*3+2, t] = 1; W[2*3+2, t] = 1
        return U, V, W

    # Verify Strassen-3 on its own 3x3x3 tensor
    U_s3, V_s3, W_s3 = strassen_3x3()
    T_33 = np.zeros((9, 9, 9))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                T_33[i*3+j, j*3+k, k*3+i] = 1.0
    T_recon_33 = np.zeros((9, 9, 9))
    for t in range(7):
        uv = np.outer(U_s3[:, t], V_s3[:, t])
        T_recon_33 += np.einsum('ab,c->abc', uv, W_s3[:, t])
    err_33 = np.max(np.abs(T_recon_33 - T_33))

    if err_33 > 1e-10:
        U, V, W = U_sb, V_sb, W_sb
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Full 4x5*5x6 decomposition: Strassen-3 on 3x3x3 cores + schoolbook strips.
    # Partition I={0,1,2},I'={3}; J={0,1,2},J'={3,4}; K={0,1,2},K'={3,4,5}
    r_total = 7 + 7 + 18 + 18 + 9 + 9 + 6 + 6
    U_full = np.zeros((nm, r_total))
    V_full = np.zeros((mp, r_total))
    W_full = np.zeros((pn, r_total))
    col = 0

    def place_strassen(k_off, i_off):
        nonlocal col
        for t in range(7):
            for a_idx in range(9):
                val = U_s3[a_idx, t]
                if val != 0:
                    i = a_idx // 3; j = a_idx % 3
                    U_full[i*5 + j, col] += val
            for b_idx in range(9):
                val = V_s3[b_idx, t]
                if val != 0:
                    j = b_idx // 3; k = b_idx % 3
                    V_full[j*6 + (k_off + k), col] += val
            for c_idx in range(9):
                val = W_s3[c_idx, t]
                if val != 0:
                    k = c_idx // 3; i = c_idx % 3
                    W_full[(k_off + k)*4 + (i_off + i), col] += val
            col += 1

    def place_schoolbook(i_vals, j_vals, k_vals, i_off=0, j_off=0, k_off=0):
        nonlocal col
        for i in i_vals:
            for j in j_vals:
                for k in k_vals:
                    U_full[(i_off + i)*5 + (j_off + j), col] = 1
                    V_full[(j_off + j)*6 + (k_off + k), col] = 1
                    W_full[(k_off + k)*4 + (i_off + i), col] = 1
                    col += 1

    # (I,J,K) and (I,J,K') cores via Strassen-3
    place_strassen(k_off=0, i_off=0)   # 7
    place_strassen(k_off=3, i_off=0)   # 7
    # Strips (schoolbook)
    place_schoolbook(range(3), range(3, 5), range(3))            # 18
    place_schoolbook(range(3), range(3, 5), range(3, 6))         # 18
    place_schoolbook(range(3), range(3), range(3), i_off=3)      # 9
    place_schoolbook(range(3), range(3), range(3, 6), i_off=3)   # 9
    place_schoolbook(range(3), range(3, 5), range(3), i_off=3)   # 6
    place_schoolbook(range(3), range(3, 5), range(3, 6), i_off=3) # 6

    err_full = verify_decomposition(U_full, V_full, W_full)
    if err_full < 1e-10:
        best_rank = r_total
        best_decomp = (U_full.copy(), V_full.copy(), W_full.copy())
    else:
        U, V, W = U_sb, V_sb, W_sb
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # --- ALS refinement (only accepts if it strictly lowers rank and stays exact) ---
    def try_als(U0, V0, W0, target_rank, max_iter):
        U = U0[:, :min(U0.shape[1], target_rank)].copy()
        V = V0[:, :min(V0.shape[1], target_rank)].copy()
        W = W0[:, :min(W0.shape[1], target_rank)].copy()
        r = target_rank
        # Pad if needed
        if U.shape[1] < r:
            pad = np.random.randn(U.shape[0], r - U.shape[1]) * 0.01
            U = np.hstack([U, pad]); V = np.hstack([V, np.random.randn(V.shape[0], r - V.shape[1]) * 0.01])
            W = np.hstack([W, np.random.randn(W.shape[0], r - W.shape[1]) * 0.01])
        X_uv = np.einsum('at,bt->abt', U, V).reshape(r, nm * mp).T   # (nm*mp, r)
        X_uw = np.einsum('at,ct->act', U, W).reshape(r, nm * pn).T   # (nm*pn, r)
        X_vw = np.einsum('bt,ct->bct', V, W).reshape(r, mp * pn).T   # (mp*pn, r)
        for _ in range(max_iter):
            if time.time() > deadline - 10:
                return None
            # Solve W
            W_new = np.zeros_like(W)
            for c in range(pn):
                W_new[c, :] = np.linalg.lstsq(X_uv, T[:, :, c].reshape(-1), rcond=None)[0]
            W = W_new
            # Solve U
            X_vw = np.einsum('bt,ct->bct', V, W).reshape(r, mp * pn).T
            U_new = np.zeros_like(U)
            for a in range(nm):
                U_new[a, :] = np.linalg.lstsq(X_vw, T[a, :, :].reshape(-1), rcond=None)[0]
            U = U_new
            # Solve V
            X_uw = np.einsum('at,ct->act', U, W).reshape(r, nm * pn).T
            V_new = np.zeros_like(V)
            for b in range(mp):
                V_new[b, :] = np.linalg.lstsq(X_uw, T[:, b, :].reshape(-1), rcond=None)[0]
            V = V_new
            err = verify_decomposition(U, V, W)
            if err < 1e-10:
                return (U.copy(), V.copy(), W.copy())
        return None

    np.random.seed(42)
    for target_rank in [78, 75, 72, 70, 68, 65]:
        if time.time() > deadline - 25:
            break
        if target_rank >= best_rank:
            continue
        found = None
        for attempt in range(3):
            if time.time() > deadline - 20:
                break
            found = try_als(*best_decomp, target_rank, max_iter=120)
            if found is not None:
                break
        if found is not None:
            U, V, W = found
            # Trim trailing zero columns to get true rank
            nz = np.where(np.any(np.abs(np.stack([U, V, W], axis=1)) > 1e-12, axis=0))[0]
            U, V, W = U[:, nz], V[:, nz], W[:, nz]
            r = U.shape[1]
            if r < best_rank and verify_decomposition(U, V, W) < 1e-10:
                best_rank = r
                best_decomp = (U.copy(), V.copy(), W.copy())

    U, V, W = best_decomp
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
