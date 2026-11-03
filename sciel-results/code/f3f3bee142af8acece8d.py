import numpy as np
import time

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    deadline = 145.0  # Leave 5s safety margin before 150s kill

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    # Build target tensor in flattened form for fast verification
    # T_flat[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0

    def verify(U, V, W, tol=1e-10):
        """Fast verification using einsum."""
        T_check = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.max(np.abs(T_check - T))

    def rank55_decomposition():
        rank = 55
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))

        col = 0

        # Part A: j=0 (first column of A, first row of B) - rank 20
        for i in range(n):
            for k in range(p):
                U[i*m + 0, col] = 1
                V[0*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1

        # Part B: j=1,2 with Strassen on 5 blocks of 2x2 - rank 35
        for b in range(5):
            k0 = 2 * b
            k1 = 2 * b + 1

            w_00 = k0 * n + 0
            w_01 = k1 * n + 0
            w_10 = k0 * n + 1
            w_11 = k1 * n + 1

            u_11 = 0 * m + 1
            u_12 = 0 * m + 2
            u_21 = 1 * m + 1
            u_22 = 1 * m + 2

            v_11 = 1 * p + k0
            v_12 = 1 * p + k1
            v_21 = 2 * p + k0
            v_22 = 2 * p + k1

            # M1 = (a11 + a22)(b11 + b22)
            U[u_11, col] += 1
            U[u_22, col] += 1
            V[v_11, col] += 1
            V[v_22, col] += 1
            W[w_00, col] += 1
            W[w_11, col] += 1
            col += 1

            # M2 = (a21 + a22)b11
            U[u_21, col] += 1
            U[u_22, col] += 1
            V[v_11, col] += 1
            W[w_10, col] += 1
            W[w_11, col] += -1
            col += 1

            # M3 = a11(b12 - b22)
            U[u_11, col] += 1
            V[v_12, col] += 1
            V[v_22, col] += -1
            W[w_01, col] += 1
            W[w_11, col] += 1
            col += 1

            # M4 = a22(b21 - b11)
            U[u_22, col] += 1
            V[v_21, col] += 1
            V[v_11, col] += -1
            W[w_00, col] += 1
            W[w_10, col] += 1
            col += 1

            # M5 = (a11 + a12)b22
            U[u_11, col] += 1
            U[u_12, col] += 1
            V[v_22, col] += 1
            W[w_00, col] += -1
            W[w_01, col] += 1
            col += 1

            # M6 = (a21 - a11)(b11 + b12)
            U[u_21, col] += 1
            U[u_11, col] += -1
            V[v_11, col] += 1
            V[v_12, col] += 1
            W[w_11, col] += 1
            col += 1

            # M7 = (a12 - a22)(b21 + b22)
            U[u_12, col] += 1
            U[u_22, col] += -1
            V[v_21, col] += 1
            V[v_22, col] += 1
            W[w_00, col] += 1
            col += 1

        return U, V, W, rank

    U_best, V_best, W_best, rank_best = rank55_decomposition()
    err = verify(U_best, V_best, W_best)

    # Vectorized ALS for the full problem
    def als_full(U, V, W, r, max_iters=30, tol=1e-10):
        """Alternating least squares for the full n x m x p problem."""
        T_flat = T.reshape(nm, mp * pn)
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break

            # Update U: for each (i,j), solve M @ U[i*m+j,:] = ones
            # M[j, c] = V[j*p+k, c] * W[k*n+i, c] for fixed i,k
            # Actually: T[i*m+j, j*p+k, k*n+i] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
            # For fixed i: sum_k T[i*m+j, j*p+k, k*n+i] * (something)
            # Let's do it properly:
            # For fixed i, the equation for U[i*m+j,:] is:
            #   sum_k V[j*p+k,:] * W[k*n+i,:] (elementwise) dot U[i*m+j,:] = T[i*m+j, j*p+k, k*n+i] for each k
            # This is a system: for each k, (V[j*p+k,:] * W[k*n+i,:]) . U[i*m+j,:] = T[i*m+j, j*p+k, k*n+i]
            # Stack over k: matrix A[i,j] has rows (V[j*p+k,:] * W[k*n+i,:]) for k=0..p-1
            # and RHS b[i,j] = T[i*m+j, j*p+0..p-1, k*n+i] for k=0..p-1

            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    # Build A: (p, r) matrix where row k = V[j*p+k,:] * W[k*n+i,:]
                    A = np.zeros((p, r))
                    b = np.zeros(p)
                    for k in range(p):
                        A[k, :] = V[j*p + k, :] * W[k*n + i, :]
                        b[k] = T[i*m + j, j*p + k, k*n + i]
                    try:
                        sol = np.linalg.lstsq(A, b, rcond=None)[0]
                        U_new[i*m + j, :] = sol
                    except:
                        U_new[i*m + j, :] = U[i*m + j, :]
            U = U_new

            # Update V
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, r))
                    b = np.zeros(n)
                    for i in range(n):
                        A[i, :] = U[i*m + j, :] * W[k*n + i, :]
                        b[i] = T[i*m + j, j*p + k, k*n + i]
                    try:
                        sol = np.linalg.lstsq(A, b, rcond=None)[0]
                        V_new[j*p + k, :] = sol
                    except:
                        V_new[j*p + k, :] = V[j*p + k, :]
            V = V_new

            # Update W
            W_new = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, r))
                    b = np.zeros(m)
                    for j in range(m):
                        A[j, :] = U[i*m + j, :] * V[j*p + k, :]
                        b[j] = T[i*m + j, j*p + k, k*n + i]
                    try:
                        sol = np.linalg.lstsq(A, b, rcond=None)[0]
                        W_new[k*n + i, :] = sol
                    except:
                        W_new[k*n + i, :] = W[k*n + i, :]
            W = W_new

            err = verify(U, V, W)
            if err < tol:
                break

        return U, V, W, err

    # Try to find rank 54 directly
    if time.time() < deadline - 20:
        for r in [54, 53]:
            if time.time() > deadline - 10:
                break

            max_restarts = 15 if r == 54 else 8

            for trial in range(max_restarts):
                if time.time() > deadline - 10:
                    break

                np.random.seed(trial * 7919 + r * 13)

                # Initialize near the rank-55 solution
                if r <= rank_best:
                    U_c = U_best[:, :r].copy() + 0.05 * np.random.randn(nm, r)
                    V_c = V_best[:, :r].copy() + 0.05 * np.random.randn(mp, r)
                    W_c = W_best[:, :r].copy() + 0.05 * np.random.randn(pn, r)
                else:
                    U_c = np.random.randn(nm, r) * 0.3
                    V_c = np.random.randn(mp, r) * 0.3
                    W_c = np.random.randn(pn, r) * 0.3

                # Run ALS
                U_c, V_c, W_c, err_c = als_full(U_c, V_c, W_c, r, max_iters=25)

                if err_c < 1e-8:
                    # Try to round to half-integers
                    U_r = np.round(U_c * 2) / 2
                    V_r = np.round(V_c * 2) / 2
                    W_r = np.round(W_c * 2) / 2
                    err_r = verify(U_r, V_r, W_r)
                    if err_r < 1e-10:
                        U_best, V_best, W_best, rank_best = U_r, V_r, W_r, r
                        break
                    elif err_c < 1e-10:
                        U_best, V_best, W_best, rank_best = U_c, V_c, W_c, r
                        break

    # Try sub-problem approach: fix j=0 part (rank 20), solve 2x2x10 at rank 34
    if time.time() < deadline - 15:
        n_sub, m_sub, p_sub = 2, 2, 10
        r_sub = 34

        T_sub = np.zeros((n_sub, m_sub, p_sub))
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    T_sub[i, j, k] = 1.0

        def als_sub(U_s, V_s, W_s, r_s, max_iters=40, tol=1e-10):
            """ALS for the 2x2x10 sub-problem."""
            for it in range(max_iters):
                if time.time() > deadline - 8:
                    break

                U_new = np.zeros_like(U_s)
                for i in range(n_sub):
                    for j in range(m_sub):
                        A = np.zeros((p_sub, r_s))
                        b = np.zeros(p_sub)
                        for k in range(p_sub):
                            A[k, :] = V_s[j*p_sub + k, :] * W_s[k*n_sub + i, :]
                            b[k] = T_sub[i, j, k]
                        try:
                            sol = np.linalg.lstsq(A, b, rcond=None)[0]
                            U_new[i*m_sub + j, :] = sol
                        except:
                            U_new[i*m_sub + j, :] = U_s[i*m_sub + j, :]
                U_s = U_new

                V_new = np.zeros_like(V_s)
                for j in range(m_sub):
                    for k in range(p_sub):
                        A = np.zeros((n_sub, r_s))
                        b = np.zeros(n_sub)
                        for i in range(n_sub):
                            A[i, :] = U_s[i*m_sub + j, :] * W_s[k*n_sub + i, :]
                            b[i] = T_sub[i, j, k]
                        try:
                            sol = np.linalg.lstsq(A, b, rcond=None)[0]
                            V_new[j*p_sub + k, :] = sol
                        except:
                            V_new[j*p_sub + k, :] = V_s[j*p_sub + k, :]
                V_s = V_new

                W_new = np.zeros_like(W_s)
                for k in range(p_sub):
                    for i in range(n_sub):
                        A = np.zeros((m_sub, r_s))
                        b = np.zeros(m_sub)
                        for j in range(m_sub):
                            A[j, :] = U_s[i*m_sub + j, :] * V_s[j*p_sub + k, :]
                            b[j] = T_sub[i, j, k]
                        try:
                            sol = np.linalg.lstsq(A, b, rcond=None)[0]
                            W_new[k*n_sub + i, :] = sol
                        except:
                            W_new[k*n_sub + i, :] = W_s[k*n_sub + i, :]
                W_s = W_new

                T_check = np.einsum('ir,jr,kr->ijk', U_s, V_s, W_s)
                err = np.max(np.abs(T_check - T_sub))
                if err < tol:
                    break

            return U_s, V_s, W_s, err

        found_sub = False
        for trial in range(20):
            if time.time() > deadline - 8:
                break
            if found_sub:
                break

            np.random.seed(trial * 31 + 100)
            U_s = np.random.randn(n_sub * m_sub, r_sub) * 0.3
            V_s = np.random.randn(m_sub * p_sub, r_sub) * 0.3
            W_s = np.random.randn(p_sub * n_sub, r_sub) * 0.3

            U_s, V_s, W_s, err_s = als_sub(U_s, V_s, W_s, r_sub, max_iters=50)

            if err_s < 1e-8:
                # Build full rank-54 solution
                U_54 = np.zeros((nm, 54))
                V_54 = np.zeros((mp, 54))
                W_54 = np.zeros((pn, 54))

                # j=0 part (first 20 columns)
                U_54[:, :20] = U_best[:, :20]
                V_54[:, :20] = V_best[:, :20]
                W_54[:, :20] = W_best[:, :20]

                # j=1,2 part (next 34 columns)
                for i in range(2):
                    for j_local in range(2):
                        j_full = j_local + 1
                        U_54[i*m + j_full, 20:54] = U_s[i*2 + j_local, :]
                for j_local in range(2):
                    j_full = j_local + 1
                    for k in range(10):
                        V_54[j_full*p + k, 20:54] = V_s[j_local*10 + k, :]
                for k in range(10):
                    for i in range(2):
                        W_54[k*n + i, 20:54] = W_s[k*2 + i, :]

                err_54 = verify(U_54, V_54, W_54)
                if err_54 < 1e-10:
                    # Try rounding
                    U_r = np.round(U_54 * 2) / 2
                    V_r = np.round(V_54 * 2) / 2
                    W_r = np.round(W_54 * 2) / 2
                    err_r = verify(U_r, V_r, W_r)
                    if err_r < 1e-10:
                        U_best, V_best, W_best, rank_best = U_r, V_r, W_r, 54
                    else:
                        U_best, V_best, W_best, rank_best = U_54, V_54, W_54, 54
                    found_sub = True
                    break

    # Final verification
    final_err = verify(U_best, V_best, W_best)
    if final_err > 1e-8:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        U_best, V_best, W_best = U, V, W
        rank_best = rank

    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
