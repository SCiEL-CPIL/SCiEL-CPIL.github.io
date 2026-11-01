import numpy as np
import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit

    # Build the matrix multiplication tensor T[a, b, c]
    # a = i*m + j, b = j*p + k, c = k*n + i
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Construct the rank-28 Strassen decomposition
    rank28 = 28
    U28 = np.zeros((n * m, rank28), dtype=int)
    V28 = np.zeros((m * p, rank28), dtype=int)
    W28 = np.zeros((p * n, rank28), dtype=int)

    strassen_terms = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]

    def map_u_a(side, local_idx):
        if side == 'L':
            return [0, 1, 4, 5][local_idx]
        else:
            return [2, 3, 6, 7][local_idx]

    def map_v_b(block, local_idx):
        r = local_idx // 2
        c = local_idx % 2
        if block == 'TL':
            return r * 4 + c
        elif block == 'TR':
            return r * 4 + (2 + c)
        elif block == 'BL':
            return (2 + r) * 4 + c
        elif block == 'BR':
            return (2 + r) * 4 + (2 + c)

    def map_w_c(side, local_idx):
        r = local_idx // 2
        c = local_idx % 2
        if side == 'L':
            return c * 2 + r
        else:
            return (2 + c) * 2 + r

    col = 0
    block_configs = [
        ('L', 'TL', 'L'),
        ('R', 'BL', 'L'),
        ('L', 'TR', 'R'),
        ('R', 'BR', 'R'),
    ]

    for a_side, b_block, c_side in block_configs:
        for u_l, v_l, w_l in strassen_terms:
            for idx, coeff in u_l:
                U28[map_u_a(a_side, idx), col] += coeff
            for idx, coeff in v_l:
                V28[map_v_b(b_block, idx), col] += coeff
            for idx, coeff in w_l:
                W28[map_w_c(c_side, idx), col] += coeff
            col += 1

    # Verify rank-28 solution
    def verify_decomp(U, V, W, tol=1e-10):
        rank = U.shape[1]
        T_decomp = np.zeros_like(T)
        for r in range(rank):
            T_decomp += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return np.allclose(T_decomp, T, atol=tol)

    if not verify_decomp(U28, V28, W28):
        rank = 32
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    U[i * m + j, idx] = 1
                    V[j * p + k, idx] = 1
                    W[k * n + i, idx] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to find a rank-27 decomposition
    if time.time() < deadline - 10:
        np.random.seed(42)
        best_rank = 28
        best_U, best_V, best_W = U28.copy().astype(float), V28.copy().astype(float), W28.copy().astype(float)
        found_rank27 = False

        # Try to find a rank-27 decomposition via optimization
        if time.time() < deadline - 20:
            # Initialize with the first 27 columns of the rank-28 solution
            U_init = U28[:, :27].astype(float)
            V_init = V28[:, :27].astype(float)
            W_init = W28[:, :27].astype(float)

            # Try a few random initializations and local searches
            for attempt in range(10):
                if time.time() > deadline - 15:
                    break

                if attempt == 0:
                    U_cur = U_init.copy()
                    V_cur = V_init.copy()
                    W_cur = W_init.copy()
                else:
                    # Random perturbation of the rank-27 initialization
                    U_cur = U28[:, :27].astype(float) + np.random.randn(n * m, 27) * 0.5
                    V_cur = V28[:, :27].astype(float) + np.random.randn(m * p, 27) * 0.5
                    W_cur = W28[:, :27].astype(float) + np.random.randn(p * n, 27) * 0.5

                # Use a more efficient optimization approach
                lr = 0.05
                for iter in range(500):
                    if time.time() > deadline - 15:
                        break

                    # Compute current decomposition
                    T_decomp = np.zeros_like(T)
                    for s in range(27):
                        T_decomp += np.einsum('a,b,c->abc', U_cur[:, s], V_cur[:, s], W_cur[:, s])
                    residual = T - T_decomp
                    err = np.sum(residual ** 2)

                    if err < 1e-10:
                        found_rank27 = True
                        best_U, best_V, best_W = U_cur, V_cur, W_cur
                        best_rank = 27
                        break

                    # Update U, V, W using gradient descent
                    for r in range(27):
                        # Gradient for U[:, r]
                        grad_U = -2 * np.einsum('abc,bc->a', residual, V_cur[:, r, None] * W_cur[:, r, None].T)
                        U_cur[:, r] -= lr * grad_U

                        # Gradient for V[:, r]
                        grad_V = -2 * np.einsum('abc,ac->b', residual, U_cur[:, r, None] * W_cur[:, r, None].T)
                        V_cur[:, r] -= lr * grad_V

                        # Gradient for W[:, r]
                        grad_W = -2 * np.einsum('abc,ab->c', residual, U_cur[:, r, None] * V_cur[:, r, None].T)
                        W_cur[:, r] -= lr * grad_W

                    if err < 1e-10:
                        found_rank27 = True
                        best_U, best_V, best_W = U_cur, V_cur, W_cur
                        best_rank = 27
                        break

                if found_rank27:
                    break

            # Check if we found a valid rank-27 solution
            if found_rank27 and verify_decomp(best_U, best_V, best_W):
                # Round to nearest integers or half-integers
                U_out = np.round(best_U * 2) / 2
                V_out = np.round(best_V * 2) / 2
                W_out = np.round(best_W * 2) / 2

                # Verify after rounding
                if verify_decomp(U_out, V_out, W_out):
                    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}

    # Return the best solution found
    if best_rank == 27:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    A = np.arange(n * m).reshape(n, m) + 1.0
    B = np.arange(m * p).reshape(m, p) + 2.0
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", U.shape[1])
