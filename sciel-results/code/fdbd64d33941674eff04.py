import numpy as np
import time

DIMENSIONS = [2, 3, 3]
_DEADLINE = 150  # seconds, with safety margin

def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    dim_U = n * m  # 6
    dim_V = m * p  # 9
    dim_W = p * n  # 6

    def build_rank17():
        rank = 17
        U = np.zeros((dim_U, rank), dtype=float)
        V = np.zeros((dim_V, rank), dtype=float)
        W = np.zeros((dim_W, rank), dtype=float)
        U[0,0]=1; U[4,0]=1; V[0,0]=1; V[4,0]=1; W[0,0]=1; W[3,0]=1
        U[3,1]=1; U[4,1]=1; V[0,1]=1; W[1,1]=1; W[3,1]=-1
        U[0,2]=1; V[1,2]=1; V[4,2]=-1; W[2,2]=1; W[3,2]=1
        U[4,3]=1; V[3,3]=1; V[0,3]=-1; W[0,3]=1; W[1,3]=1
        U[0,4]=1; U[1,4]=1; V[4,4]=1; W[2,4]=1; W[0,4]=-1
        U[3,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
        U[1,6]=1; U[4,6]=-1; V[3,6]=1; V[4,6]=1; W[0,6]=1
        U[2,7]=1; V[6,7]=1; W[0,7]=1
        U[2,8]=1; V[7,8]=1; W[2,8]=1
        U[5,9]=1; V[6,9]=1; W[1,9]=1
        U[5,10]=1; V[7,10]=1; W[3,10]=1
        U[0,11]=1; V[2,11]=1; W[4,11]=1
        U[1,12]=1; V[5,12]=1; W[4,12]=1
        U[3,13]=1; V[2,13]=1; W[5,13]=1
        U[4,14]=1; V[5,14]=1; W[5,14]=1
        U[2,15]=1; V[8,15]=1; W[4,15]=1
        U[5,16]=1; V[8,16]=1; W[5,16]=1
        return U, V, W, rank

    def verify(U, V, W, R, trials=50, seed=42):
        rng = np.random.RandomState(seed)
        for _ in range(trials):
            A = rng.randint(-6, 7, size=(n, m)).astype(float)
            B = rng.randint(-6, 7, size=(m, p)).astype(float)
            C_true = A @ B
            Af = A.flatten(); Bf = B.flatten()
            C_approx = np.zeros((n, p))
            for r in range(R):
                u = U[:, r] @ Af
                v = V[:, r] @ Bf
                if u == 0 and v == 0:
                    continue
                wvec = W[:, r] * u * v
                for i in range(n):
                    for k in range(p):
                        C_approx[i, k] += wvec[k * n + i]
            if not np.allclose(C_approx, C_true, atol=1e-6):
                return False
        return True

    constraints = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                constraints.append((a, b, c))
    cons_a = np.array([c[0] for c in constraints])
    cons_b = np.array([c[1] for c in constraints])
    cons_c = np.array([c[2] for c in constraints])
    cons_val = np.ones(len(constraints))

    def try_rank16():
        U17, V17, W17, _ = build_rank17()
        best_err = float('inf')
        best_sol = None
        rng = np.random.RandomState(123)
        for drop in range(17):
            if time.time() - t0 > _DEADLINE - 30:
                break
            keep = [i for i in range(17) if i != drop]
            U0 = U17[:, keep].copy()
            V0 = V17[:, keep].copy()
            W0 = W17[:, keep].copy()
            R = 16
            for restart in range(5):
                if time.time() - t0 > _DEADLINE - 12:
                    break
                U = U0.copy()
                V = V0.copy()
                W = W0.copy()
                U += rng.randn(*U.shape) * 0.02
                V += rng.randn(*V.shape) * 0.02
                W += rng.randn(*W.shape) * 0.02
                for it in range(500):
                    if time.time() - t0 > _DEADLINE - 8:
                        break
                    for a in range(dim_U):
                        mask = cons_a == a
                        if not mask.any():
                            continue
                        A_sys = V[cons_b[mask], :] * W[cons_c[mask], :]
                        sol, *_ = np.linalg.lstsq(A_sys, cons_val[mask], rcond=None)
                        U[a, :] = sol
                    for b in range(dim_V):
                        mask = cons_b == b
                        if not mask.any():
                            continue
                        A_sys = U[cons_a[mask], :] * W[cons_c[mask], :]
                        sol, *_ = np.linalg.lstsq(A_sys, cons_val[mask], rcond=None)
                        V[b, :] = sol
                    for c in range(dim_W):
                        mask = cons_c == c
                        if not mask.any():
                            continue
                        A_sys = U[cons_a[mask], :] * V[cons_b[mask], :]
                        sol, *_ = np.linalg.lstsq(A_sys, cons_val[mask], rcond=None)
                        W[c, :] = sol
                    pred = np.einsum('tr,tr,tr->t', U[cons_a, :], V[cons_b, :], W[cons_c, :])
                    err = float(np.sum((pred - cons_val) ** 2))
                    if err < best_err:
                        best_err = err
                        best_sol = (U.copy(), V.copy(), W.copy())
                    if err < 1e-13:
                        return U, V, W, R
            if best_err < 1e-10:
                return best_sol[0], best_sol[1], best_sol[2], R
        return None

    # Safe baseline
    U17, V17, W17, r17 = build_rank17()
    if not verify(U17, V17, W17, r17):
        rank = 18
        U = np.zeros((dim_U, rank), dtype=float)
        V = np.zeros((dim_V, rank), dtype=float)
        W = np.zeros((dim_W, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Attempt rank 16
    if time.time() - t0 < _DEADLINE - 45:
        result16 = try_rank16()
        if result16 is not None:
            U, V, W, R = result16
            if verify(U, V, W, R):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U17.tolist(), "V": V17.tolist(), "W": W17.tolist()}
