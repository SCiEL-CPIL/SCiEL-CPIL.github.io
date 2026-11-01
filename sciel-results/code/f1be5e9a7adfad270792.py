import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 2, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 150s limit, keep 20s safety margin

    n, m, p = DIMENSIONS  # 2, 2, 6

    # Build the target tensor T[a, b, c] where:
    # a = i*m + j  (i in [0,n), j in [0,m))  -> A index
    # b = j*p + k  (j in [0,m), k in [0,p))  -> B index
    # c = k*n + i  (k in [0,p), i in [0,n))  -> C index (trace convention W[k*n+i, r])
    # T[a, b, c] = 1 if the (i,j) entry of A and (j,k) entry of B contribute to (k,i) of C
    # Actually: C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[a,b,c] = 1 when a=(i,j), b=(j,k), c=(k,i) for the same i,j,k
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1.0

    def verify(U, V, W):
        """Check if U, V, W exactly decompose T."""
        S = np.einsum('as,bs,cs->abc', U, V, W)
        return np.allclose(S, T, atol=1e-9)

    def snap_half(U, V, W):
        """Snap to nearest half-integer and verify."""
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        if verify(U_s, V_s, W_s):
            return True, U_s, V_s, W_s
        return False, None, None, None

    # Fallback: Rank 24 schoolbook
    rank_school = n * m * p  # 24
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    for s, (i, j, k) in enumerate([(i, j, k) for i in range(n) for j in range(m) for k in range(p)]):
        U_school[i*m + j, s] = 1.0
        V_school[j*p + k, s] = 1.0
        W_school[k*n + i, s] = 1.0

    # Fallback: Rank 21 via 3 independent Strassen
    rank_strassen = 21
    U_str = np.zeros((n*m, rank_strassen))
    V_str = np.zeros((m*p, rank_strassen))
    W_str = np.zeros((p*n, rank_strassen))

    # Strassen 7 terms for 2x2 * 2x2
    # Each term: (u_vec, v_vec, w_vec) where each is length-4 vector
    # A flat index: i*2+j, B flat: j*2+k, C flat: k*2+i
    strassen_terms = [
        # M1 = (A00+A11)(B00+B11)
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        # M2 = (A10+A11)B00
        ([0,0,1,1], [1,0,0,0], [0,1,0,-1]),
        # M3 = A00(B10-B11)
        ([1,0,0,0], [0,1,0,-1], [0,0,1,1]),
        # M4 = A11(B00-B01)
        ([0,0,0,1], [-1,0,1,0], [1,1,0,0]),
        # M5 = (A00+A01)B11
        ([1,1,0,0], [0,0,0,1], [-1,0,1,0]),
        # M6 = (A10-A00)(B00+B01)
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),
        # M7 = (A01-A11)(B10+B11)
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),
    ]

    for b in range(3):  # 3 blocks of 2 columns
        for s_idx in range(7):
            col = b * 7 + s_idx
            u_vec, v_vec, w_vec = strassen_terms[s_idx]
            for idx in range(4):
                U_str[idx, col] = u_vec[idx]
            # V: B is 2x6, flat index j*6+k. Block b covers k=2b, 2b+1
            v_indices = [2*b, 2*b+1, 6+2*b, 6+2*b+1]
            for idx in range(4):
                V_str[v_indices[idx], col] = v_vec[idx]
            # W: C is 2x6, flat index k*2+i. Block b covers k=2b, 2b+1
            w_indices = [4*b, 4*b+1, 4*b+2, 4*b+3]
            for idx in range(4):
                W_str[w_indices[idx], col] = w_vec[idx]

    best_rank = 21
    best_U, best_V, best_W = U_str, V_str, W_str

    # Try to find rank 20 or lower
    for target_rank in [20, 19, 18]:
        if time.time() > deadline - 60:
            break

        def make_residuals(R):
            def residuals(params):
                U_s = params[0:R*4].reshape(R, 4).T  # 4 x R
                V_s = params[R*4:R*16].reshape(R, 12).T  # 12 x R
                W_s = params[R*16:].reshape(R, 12).T  # 12 x R
                S = np.einsum('as,bs,cs->abc', U_s, V_s, W_s)
                return (S - T).flatten()
            return residuals

        residuals = make_residuals(target_rank)

        # Generate initializations
        inits = []

        # Truncated rank-21 inits (drop one column)
        for drop_col in range(21):
            mask = [i for i in range(21) if i != drop_col]
            U_init = U_str[:, mask][:, :target_rank] if target_rank <= 20 else U_str[:, mask]
            V_init = V_str[:, mask][:, :target_rank] if target_rank <= 20 else V_str[:, mask]
            W_init = W_str[:, mask][:, :target_rank] if target_rank <= 20 else W_str[:, mask]

            if U_init.shape[1] < target_rank:
                # Pad with small random values
                pad_U = np.random.randn(4, target_rank - U_init.shape[1]) * 0.01
                pad_V = np.random.randn(12, target_rank - V_init.shape[1]) * 0.01
                pad_W = np.random.randn(12, target_rank - W_init.shape[1]) * 0.01
                U_init = np.hstack([U_init, pad_U])
                V_init = np.hstack([V_init, pad_V])
                W_init = np.hstack([W_init, pad_W])
            else:
                U_init = U_init[:, :target_rank]
                V_init = V_init[:, :target_rank]
                W_init = W_init[:, :target_rank]

            init_params = np.concatenate([
                U_init.T.flatten(),
                V_init.T.flatten(),
                W_init.T.flatten()
            ])
            inits.append(init_params)

        # Random inits
        np.random.seed(42)
        for _ in range(30):
            if time.time() > deadline - 30:
                break
            U_rand = np.random.randn(4, target_rank) * 0.5
            V_rand = np.random.randn(12, target_rank) * 0.5
            W_rand = np.random.randn(12, target_rank) * 0.5
            init_params = np.concatenate([
                U_rand.T.flatten(),
                V_rand.T.flatten(),
                W_rand.T.flatten()
            ])
            inits.append(init_params)

        found = False
        for init in inits:
            if time.time() > deadline - 10:
                break
            try:
                result = least_squares(residuals, init, method='lm', max_nfev=200)
                if result.cost < 1e-8:
                    R = target_rank
                    U_sol = result.x[0:R*4].reshape(R, 4).T
                    V_sol = result.x[R*4:R*16].reshape(R, 12).T
                    W_sol = result.x[R*16:].reshape(R, 12).T

                    valid, U_snap, V_snap, W_snap = snap_half(U_sol, V_sol, W_sol)
                    if valid:
                        best_rank = R
                        best_U, best_V, best_W = U_snap, V_snap, W_snap
                        found = True
                        break
            except Exception:
                continue

        if found:
            break

    # Convert to lists
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()

    return {"U": U_out, "V": V_out, "W": W_out}
