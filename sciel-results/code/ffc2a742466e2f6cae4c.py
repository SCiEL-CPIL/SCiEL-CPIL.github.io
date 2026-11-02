import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin before 150s kill

    n, m, p = 2, 2, 6
    nm, mp, pn = n * m, m * p, p * n  # 4, 12, 12

    # Construct the target tensor T
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Helper to check if a decomposition is valid and roundable
    def check_and_round(U, V, W):
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        T_calc = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
        if np.allclose(T_calc, T, atol=1e-9):
            return U_r, V_r, W_r
        return None

    # Helper to perform ALS for a given rank with Strassen-based initialization
    def try_rank(R, max_iters=80, n_restarts=8):
        for restart in range(n_restarts):
            if time.time() > deadline - 5:
                break

            np.random.seed(42 + restart * 1000 + R)

            # Initialize using Strassen structure as a warm start, then perturb
            U0, V0, W0 = _strassen_init(R)
            if U0 is None:
                U = np.random.randn(nm, R) * 0.5
                V = np.random.randn(mp, R) * 0.5
                W = np.random.randn(pn, R) * 0.5
            else:
                U = U0 + np.random.randn(nm, R) * 0.1
                V = V0 + np.random.randn(mp, R) * 0.1
                W = W0 + np.random.randn(pn, R) * 0.1

            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break

                # Update W: solve linear system
                A_mat = np.zeros((24, pn * R))
                b_vec = np.zeros(24)
                idx = 0
                for a in range(nm):
                    for b in range(mp):
                        for c in range(pn):
                            if T[a, b, c] != 0:
                                for r in range(R):
                                    A_mat[idx, c * R + r] = U[a, r] * V[b, r]
                                b_vec[idx] = T[a, b, c]
                                idx += 1
                try:
                    W_flat, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    W = W_flat.reshape(pn, R)
                except:
                    continue

                # Update U
                A_mat = np.zeros((24, nm * R))
                b_vec = np.zeros(24)
                idx = 0
                for a in range(nm):
                    for b in range(mp):
                        for c in range(pn):
                            if T[a, b, c] != 0:
                                for r in range(R):
                                    A_mat[idx, a * R + r] = V[b, r] * W[c, r]
                                b_vec[idx] = T[a, b, c]
                                idx += 1
                try:
                    U_flat, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    U = U_flat.reshape(nm, R)
                except:
                    continue

                # Update V
                A_mat = np.zeros((24, mp * R))
                b_vec = np.zeros(24)
                idx = 0
                for a in range(nm):
                    for b in range(mp):
                        for c in range(pn):
                            if T[a, b, c] != 0:
                                for r in range(R):
                                    A_mat[idx, b * R + r] = U[a, r] * W[c, r]
                                b_vec[idx] = T[a, b, c]
                                idx += 1
                try:
                    V_flat, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    V = V_flat.reshape(mp, R)
                except:
                    continue

                # Check convergence
                T_calc = np.einsum('ar,br,cr->abc', U, V, W)
                err = np.linalg.norm(T_calc - T)
                if err < 1e-8:
                    result = check_and_round(U, V, W)
                    if result is not None:
                        return result

        return None

    def _strassen_init(R):
        """Initialize with Strassen structure, padded/truncated to rank R."""
        if R < 7:
            return None, None, None
        strassen_steps = [
            {'u': [1, 0, 0, 1], 'v': [1, 0, 0, 1], 'w': [1, 0, 0, 1]},
            {'u': [0, 0, 1, 1], 'v': [1, 0, 0, 0], 'w': [0, 0, 1, -1]},
            {'u': [1, 0, 0, 0], 'v': [0, 1, 0, -1], 'w': [0, 1, 0, 1]},
            {'u': [0, 0, 0, 1], 'v': [-1, 0, 1, 0], 'w': [1, 0, 1, 0]},
            {'u': [1, 1, 0, 0], 'v': [0, 0, 0, 1], 'w': [-1, 1, 0, 0]},
            {'u': [-1, 0, 1, 0], 'v': [1, 1, 0, 0], 'w': [0, 0, 0, 1]},
            {'u': [0, 1, 0, -1], 'v': [0, 0, 1, 1], 'w': [1, 0, 0, 0]},
        ]
        U = np.zeros((nm, R))
        V = np.zeros((mp, R))
        W = np.zeros((pn, R))
        for g in range(3):
            for s in range(7):
                r = 7 * g + s
                if r >= R:
                    break
                step = strassen_steps[s]
                k0 = 2 * g
                k1 = 2 * g + 1
                b_map = [6*0+k0, 6*0+k1, 6*1+k0, 6*1+k1]
                c_map = [2*k0+0, 2*k1+0, 2*k0+1, 2*k1+1]
                for l in range(4):
                    U[l, r] = step['u'][l]
                    V[b_map[l], r] = step['v'][l]
                    W[c_map[l], r] = step['w'][l]
        return U, V, W

    # Try to find lower ranks, starting from 20 down to 15
    for R in range(20, 14, -1):
        if time.time() > deadline - 10:
            break
        result = try_rank(R, max_iters=60, n_restarts=6)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to Rank 21 (Strassen based)
    strassen_steps = [
        {'u': [1, 0, 0, 1], 'v': [1, 0, 0, 1], 'w': [1, 0, 0, 1]},
        {'u': [0, 0, 1, 1], 'v': [1, 0, 0, 0], 'w': [0, 0, 1, -1]},
        {'u': [1, 0, 0, 0], 'v': [0, 1, 0, -1], 'w': [0, 1, 0, 1]},
        {'u': [0, 0, 0, 1], 'v': [-1, 0, 1, 0], 'w': [1, 0, 1, 0]},
        {'u': [1, 1, 0, 0], 'v': [0, 0, 0, 1], 'w': [-1, 1, 0, 0]},
        {'u': [-1, 0, 1, 0], 'v': [1, 1, 0, 0], 'w': [0, 0, 0, 1]},
        {'u': [0, 1, 0, -1], 'v': [0, 0, 1, 1], 'w': [1, 0, 0, 0]},
    ]

    U = np.zeros((4, 21))
    V = np.zeros((12, 21))
    W = np.zeros((12, 21))

    for g in range(3):
        for s in range(7):
            r = 7 * g + s
            step = strassen_steps[s]
            k0 = 2 * g
            k1 = 2 * g + 1
            b_map = [6*0+k0, 6*0+k1, 6*1+k0, 6*1+k1]
            c_map = [2*k0+0, 2*k1+0, 2*k0+1, 2*k1+1]
            for l in range(4):
                U[l, r] = step['u'][l]
                V[b_map[l], r] = step['v'][l]
                W[c_map[l], r] = step['w'][l]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
