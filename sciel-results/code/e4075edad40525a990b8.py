import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin

    n, m, p = DIMENSIONS

    # Build the tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1

    # Schoolbook decomposition
    rank = n*m*p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1

    best_rank = rank
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()

    def reconstruct(U, V, W):
        # T_rec[a,b,c] = sum_s U[a,s] V[b,s] W[c,s]
        # Vectorized: np.einsum('as,bs,cs->abc', U, V, W)
        return np.einsum('as,bs,cs->abc', U, V, W, optimize=True)

    def is_valid(U, V, W, tol=1e-8):
        T_rec = reconstruct(U, V, W)
        return np.allclose(T_rec, T, atol=tol)

    def try_round(U, V, W, tol=1e-6):
        """Try rounding to half-integers and check validity."""
        # Try rounding to nearest half-integer
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        if is_valid(U_r, V_r, W_r, tol):
            return U_r, V_r, W_r
        
        # Try rounding to nearest integer
        U_r = np.round(U)
        V_r = np.round(V)
        W_r = np.round(W)
        if is_valid(U_r, V_r, W_r, tol):
            return U_r, V_r, W_r
            
        return None

    def solve_U(V, W, T):
        """Solve for U given V, W, T. T[a,b,c] = sum_s U[a,s] V[b,s] W[c,s]"""
        m_p = m * p
        p_n = p * n
        rnk = V.shape[1]
        # For each a: T[a,:,:].flatten() = U[a,:] @ M_T where M_T[s, (b,c)] = V[b,s]*W[c,s]
        # M_T is (rnk, m_p * p_n)
        # We want U (n*m, rnk) such that U @ M_T.T = T_flat.T (n*m, m_p*p_n)
        # So M_T @ U.T = T_flat.T
        # U.T = M_T^-1 T_flat.T -> U = (M_T^-1 T_flat.T).T
        # Using lstsq: M_T.T @ U.T = T_flat.T is wrong dimension.
        # Equation: sum_s U[a,s] (V[b,s] W[c,s]) = T[a,b,c]
        # Let M[s, idx] = V[b,s] W[c,s] where idx = b*p_n + c
        # Then (U @ M)[a, idx] = T[a, b, c]
        # So U @ M = T_flat
        # U = T_flat @ M^+ (pseudo inverse)
        
        M = np.einsum('bs,cs->sc', V, W) # (rnk, m_p * p_n) ? No.
        # V is (m_p, rnk), W is (p_n, rnk)
        # We need M such that U @ M = T_flat.
        # U[a,s] * M[s, bc] = T[a, bc]
        # M[s, bc] should be V[b,s] * W[c,s].
        # So M is (rnk, m_p * p_n).
        M = (V[:, :, None] * W[:, None, :]).transpose(0, 2, 1).reshape(rnk, -1) 
        # Wait, V is (m_p, rnk). V[:, s] is (m_p,).
        # We want M[s, b, c] = V[b, s] * W[c, s].
        # M = np.einsum('bs,cs->scb', V, W).reshape(rnk, -1)
        
        M = np.einsum('bs,cs->scb', V, W).reshape(rnk, m_p * p_n)
        T_flat = T.reshape(n*m, m_p * p_n)
        
        # Solve U @ M = T_flat for U.
        # This is equivalent to M.T @ U.T = T_flat.T
        try:
            U_sol = np.linalg.lstsq(M.T, T_flat.T, rcond=None)[0].T
        except:
            U_sol = np.zeros((n*m, rnk))
        return U_sol

    def solve_V(U, W, T):
        """Solve for V given U, W, T."""
        m_p = m * p
        p_n = p * n
        rnk = U.shape[1]
        # Equation: sum_s U[a,s] V[b,s] W[c,s] = T[a,b,c]
        # For fixed b: sum_s V[b,s] (U[a,s] W[c,s]) = T[a,b,c]
        # Let M[s, ac] = U[a,s] W[c,s]
        # V[b,s] M[s, ac] = T[a,b,c]
        # V @ M = T_flat_b where T_flat_b[ac] = T[a,b,c]
        M = np.einsum('as,cs->sca', U, W).reshape(rnk, n*m * p_n)
        T_flat = T.transpose(1, 0, 2).reshape(m_p, n*m * p_n)
        try:
            V_sol = np.linalg.lstsq(M.T, T_flat.T, rcond=None)[0].T
        except:
            V_sol = np.zeros((m_p, rnk))
        return V_sol

    def solve_W(U, V, T):
        """Solve for W given U, V, T."""
        m_p = m * p
        p_n = p * n
        rnk = U.shape[1]
        # Equation: sum_s U[a,s] V[b,s] W[c,s] = T[a,b,c]
        # For fixed c: sum_s W[c,s] (U[a,s] V[b,s]) = T[a,b,c]
        # Let M[s, ab] = U[a,s] V[b,s]
        # W[c,s] M[s, ab] = T[a,b,c]
        M = np.einsum('as,bs->sab', U, V).reshape(rnk, n*m * m_p)
        T_flat = T.transpose(2, 0, 1).reshape(p_n, n*m * m_p)
        try:
            W_sol = np.linalg.lstsq(M.T, T_flat.T, rcond=None)[0].T
        except:
            W_sol = np.zeros((p_n, rnk))
        return W_sol

    # Try ALS for various target ranks
    # We prioritize ranks close to 75 first, as they are more likely to exist.
    target_ranks = [74, 73, 72, 71, 70, 69, 68, 67, 66, 65, 60, 55, 50, 45, 40, 35, 30, 25]

    for target_rank in target_ranks:
        if time.time() > deadline - 10:
            break

        max_iters = 50
        # Use more restarts for higher ranks
        max_restarts = 20 if target_rank > 70 else 10

        found = False
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break

            np.random.seed(42 + restart * 100 + target_rank)
            U = np.random.randn(n*m, target_rank) * 0.5
            V = np.random.randn(m*p, target_rank) * 0.5
            W = np.random.randn(p*n, target_rank) * 0.5

            prev_err = np.inf
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break

                U = solve_U(V, W, T)
                V = solve_V(U, W, T)
                W = solve_W(U, V, T)

                T_rec = reconstruct(U, V, W)
                err = np.linalg.norm(T_rec - T)

                if err < 1e-10:
                    break
                if abs(prev_err - err) < 1e-14:
                    break
                prev_err = err

            T_rec = reconstruct(U, V, W)
            err = np.linalg.norm(T_rec - T)

            if err < 1e-8:
                rounded = try_round(U, V, W, tol=1e-6)
                if rounded is not None:
                    U_r, V_r, W_r = rounded
                    if is_valid(U_r, V_r, W_r, 1e-8):
                        best_rank = target_rank
                        best_U = U_r
                        best_V = V_r
                        best_W = W_r
                        found = True
                        break
                elif is_valid(U, V, W, 1e-8):
                    # If exact float solution found, it's valid, but we prefer integers.
                    # The contract says "integer/half-integer entries".
                    # If we can't round, we might fail the "integer" check if the evaluator is strict.
                    # However, usually "exactly decompose" implies the values must be exact.
                    # If the float values are like 0.4999999, rounding works.
                    # If they are irrational, we can't use them.
                    pass

        if found:
            break

    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]

    return {"U": U_list, "V": V_list, "W": W_list}
