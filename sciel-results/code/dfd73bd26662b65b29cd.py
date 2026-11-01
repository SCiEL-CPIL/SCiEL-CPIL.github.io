import numpy as np
import time
from fractions import Fraction

def construct():
    n, m, p = 2, 2, 3
    start_time = time.time()
    deadline = start_time + 135.0

    nm, mp, pn = n * m, m * p, p * n  # 4, 6, 6

    # Build target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def verify(U, V, W):
        """Check if U, V, W exactly decompose T."""
        T_prime = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_prime, T, atol=1e-9)

    def try_rank(r):
        """Try to find a rank-r decomposition via randomized gradient descent."""
        if time.time() > deadline:
            return None

        best = None
        n_restarts = 30
        max_iter = 2000

        for restart in range(n_restarts):
            if time.time() > deadline:
                break

            # Random initialization
            U = np.random.randn(nm, r) * 0.5
            V = np.random.randn(mp, r) * 0.5
            W = np.random.randn(pn, r) * 0.5

            # Gradient descent
            lr = 0.01
            for it in range(max_iter):
                if time.time() > deadline:
                    break

                # Compute current tensor approximation
                T_approx = np.einsum('ar,br,cr->abc', U, V, W)
                error = T_approx - T
                err_norm = np.linalg.norm(error)

                if err_norm < 1e-10:
                    break

                # Gradients: d/dU = error contracted with V^T, W^T
                # dU[a,r] = sum_{b,c} error[a,b,c] * V[b,r] * W[c,r]
                dU = np.einsum('abc,br,cr->ar', error, V, W)
                dV = np.einsum('abc,ar,cr->br', error, U, W)
                dW = np.einsum('abc,ar,br->cr', error, U, V)

                # Adaptive learning rate
                grad_norm = np.sqrt(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2))
                if grad_norm > 0:
                    step = lr / (1.0 + 0.1 * err_norm)
                else:
                    break

                U -= step * dU
                V -= step * dV
                W -= step * dW

            # After optimization, try to recover rational values
            if err_norm < 1e-8:
                # Try rounding to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                if verify(U_r, V_r, W_r):
                    best = (U_r, V_r, W_r)
                    return best

                # Try continued fractions for more precise recovery
                def rationalize_matrix(M, max_den=100):
                    M_rat = np.zeros_like(M)
                    for idx in np.ndindex(M.shape):
                        val = M[idx]
                        # Try small denominators
                        found = False
                        for den in range(1, max_den + 1):
                            num = round(val * den)
                            if abs(val - num / den) < 1e-9:
                                M_rat[idx] = num / den
                                found = True
                                break
                        if not found:
                            M_rat[idx] = val
                    return M_rat

                U_r = rationalize_matrix(U)
                V_r = rationalize_matrix(V)
                W_r = rationalize_matrix(W)
                if verify(U_r, V_r, W_r):
                    best = (U_r, V_r, W_r)
                    return best

        return best

    # Try ranks from 10 upward (we know 11 works, schoolbook is 12)
    for r in [10, 11]:
        if time.time() > deadline:
            break
        result = try_rank(r)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: use the known rank-11 Strassen + schoolbook construction
    # Part 1: 2x2x2 Strassen for k=0,1
    U_strassen = np.zeros((nm, 7))
    V_strassen = np.zeros((mp, 7))
    W_strassen = np.zeros((pn, 7))

    def add_strassen_term(idx, u_dict, v_dict, w_dict):
        for (i, j), coeff in u_dict.items():
            a = i * 2 + j
            U_strassen[a, idx] += coeff
        for (j, k), coeff in v_dict.items():
            b = j * 3 + k
            V_strassen[b, idx] += coeff
        for (k, i), coeff in w_dict.items():
            c = k * 2 + i
            W_strassen[c, idx] += coeff

    add_strassen_term(0, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1})
    add_strassen_term(1, {(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1, (1,1):1})
    add_strassen_term(2, {(0,0):1}, {(1,0):1, (1,1):-1}, {(1,0):1, (0,1):1})
    add_strassen_term(3, {(1,1):1}, {(0,1):1, (0,0):-1}, {(0,0):1, (0,1):1})
    add_strassen_term(4, {(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (1,0):1})
    add_strassen_term(5, {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1})
    add_strassen_term(6, {(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1})

    # Part 2: k=2 Schoolbook
    U_school = np.zeros((nm, 4))
    V_school = np.zeros((mp, 4))
    W_school = np.zeros((pn, 4))

    U_school[0, 0] = 1; V_school[2, 0] = 1; W_school[4, 0] = 1
    U_school[1, 1] = 1; V_school[5, 1] = 1; W_school[4, 1] = 1
    U_school[2, 2] = 1; V_school[2, 2] = 1; W_school[5, 2] = 1
    U_school[3, 3] = 1; V_school[5, 3] = 1; W_school[5, 3] = 1

    U = np.hstack([U_strassen, U_school])
    V = np.hstack([V_strassen, V_school])
    W = np.hstack([W_strassen, W_school])

    # Verify fallback
    if not verify(U, V, W):
        # Absolute fallback: schoolbook rank 12
        U_sb = np.zeros((nm, 12))
        V_sb = np.zeros((mp, 12))
        W_sb = np.zeros((pn, 12))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1
                    V_sb[j * p + k, col] = 1
                    W_sb[k * n + i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


if __name__ == "__main__":
    result = construct()
    print("Rank:", len(result['U'][0]))
