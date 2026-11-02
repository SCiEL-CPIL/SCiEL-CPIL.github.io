"""Find a rank-minimal exact tensor decomposition for 2x3 * 3x4 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145
    n, m, p = 2, 3, 4
    Na = n * m  # 6
    Nb = m * p  # 12
    Nc = p * n  # 8

    # Build target tensor
    T = np.zeros((Na, Nb, Nc))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(Na)]
        V = [[0] * rank for _ in range(Nb)]
        W = [[0] * rank for _ in range(Nc)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W, T):
        recon = np.zeros_like(T)
        r = U.shape[1]
        for c in range(r):
            recon += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
        return np.max(np.abs(recon - T)) < 1e-10

    def try_rank_r(r, n_restarts=20, max_iter=400):
        best_err = np.inf
        best_UVW = None
        rng = np.random.RandomState(123 + r)

        for restart in range(n_restarts):
            if time.time() > deadline - 15:
                break
            U = rng.randn(Na, r) * 0.5
            V = rng.randn(Nb, r) * 0.5
            W = rng.randn(Nc, r) * 0.5

            for it in range(max_iter):
                if time.time() > deadline - 15:
                    break

                # Update U
                VkrW = np.zeros((Nb * Nc, r))
                for c in range(r):
                    VkrW[:, c] = np.kron(V[:, c], W[:, c])
                T_mat = T.reshape(Na, Nb * Nc)
                try:
                    U_new, _, _, _ = np.linalg.lstsq(VkrW, T_mat.T, rcond=None)
                    U_new = U_new.T
                except:
                    U_new = U

                # Update V
                UkrW = np.zeros((Na * Nc, r))
                for c in range(r):
                    UkrW[:, c] = np.kron(U[:, c], W[:, c])
                T_mat2 = T.transpose(1, 0, 2).reshape(Nb, Na * Nc)
                try:
                    V_new, _, _, _ = np.linalg.lstsq(UkrW, T_mat2.T, rcond=None)
                    V_new = V_new.T
                except:
                    V_new = V

                # Update W
                UkV = np.zeros((Na * Nb, r))
                for c in range(r):
                    UkV[:, c] = np.kron(U[:, c], V[:, c])
                T_mat3 = T.transpose(2, 0, 1).reshape(Nc, Na * Nb)
                try:
                    W_new, _, _, _ = np.linalg.lstsq(UkV, T_mat3.T, rcond=None)
                    W_new = W_new.T
                except:
                    W_new = W

                U, V, W = U_new, V_new, W_new

                recon = np.zeros_like(T)
                for c in range(r):
                    recon += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
                err = np.sum((recon - T) ** 2)

                if err < best_err:
                    best_err = err
                    best_UVW = (U.copy(), V.copy(), W.copy())

                if err < 1e-12:
                    break

        if best_UVW is not None:
            U, V, W = best_UVW
            # Try snapping to half-integers
            all_arr = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            scaled = all_arr * 2
            rounded = np.round(scaled)
            if np.max(np.abs(scaled - rounded)) < 1e-6:
                U2 = (2 * U).round() / 2.0
                V2 = (2 * V).round() / 2.0
                W2 = (2 * W).round() / 2.0
                if verify(U2, V2, W2, T):
                    return r, U2, V2, W2
            # Try integers
            rounded_int = np.round(all_arr)
            if np.max(np.abs(all_arr - rounded_int)) < 1e-6:
                U2 = U.round()
                V2 = V.round()
                W2 = W.round()
                if verify(U2, V2, W2, T):
                    return r, U2, V2, W2
            # Accept if error is tiny
            recon = np.zeros_like(T)
            for c in range(r):
                recon += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
            if np.max(np.abs(recon - T)) < 1e-10:
                return r, U, V, W
        return None

    for r in range(1, 25):
        if time.time() > deadline - 20:
            break
        result = try_rank_r(r, n_restarts=15, max_iter=300)
        if result is not None:
            rank, U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return schoolbook()
# EVOLVE-BLOCK-END
