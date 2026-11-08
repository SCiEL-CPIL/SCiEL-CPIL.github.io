"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Leave 40s margin for safety

    def verify(U, V, W, n, m, p, tol=1e-5):
        """Verify decomposition correctness."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        np.random.seed(999)
        for _ in range(200):
            a = np.random.randint(n*m)
            b = np.random.randint(m*p)
            d = np.random.randint(p*n)
            i1, j1 = divmod(a, m)
            j2, k2 = divmod(b, p)
            k3, i3 = divmod(d, n)
            if not (j1 == j2 and k2 == k3 and i3 == i1):
                val = np.dot(U[a, :], V[b, :] * W[d, :])
                if abs(val) > tol:
                    return False
        return True

    def find_decomp(n, m, p, target_rank, max_iter=5000, seed=42, deadline=None):
        np.random.seed(seed)
        U = np.random.randn(n*m, target_rank) * 0.5
        V = np.random.randn(m*p, target_rank) * 0.5
        W = np.random.randn(p*n, target_rank) * 0.5

        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1.0

        def loss_and_grad(U, V, W):
            T_approx = np.zeros((n*m, m*p, p*n))
            for c in range(target_rank):
                T_approx += np.einsum('a,b,c->abc', U[:,c], V[:,c], W[:,c])

            diff = T_approx - T
            loss = np.sum(diff**2)

            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            for c in range(target_rank):
                grad_U[:,c] = np.sum(diff * V[:,c][None,:,None] * W[:,c][None,None,:], axis=(1,2))
                grad_V[:,c] = np.sum(diff * U[:,c][:,None,None] * W[:,c][None,None,:], axis=(0,2))
                grad_W[:,c] = np.sum(diff * U[:,c][:,None,None] * V[:,c][None,:,None], axis=(0,1))

            return loss, (grad_U, grad_V, grad_W)

        lr = 0.003
        best_loss = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

        for it in range(max_iter):
            if deadline is not None and time.time() > deadline:
                break
            loss, (gU, gV, gW) = loss_and_grad(U, V, W)

            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

            if best_loss < 1e-14:
                break

            U -= lr * gU
            V -= lr * gV
            W -= lr * gW

            if it % 200 == 0:
                lr *= 0.9

        return best_U, best_V, best_W, best_loss

    # Try decreasing ranks
    for target_rank in [44, 43, 42, 41]:
        if time.time() > deadline:
            break
        for seed in [42, 123, 456, 789, 1000]:
            if time.time() > deadline:
                break
            U, V, W, loss = find_decomp(n, m, p, target_rank, max_iter=4000, seed=seed, deadline=deadline)
            if loss < 1e-8:
                if verify(U, V, W, n, m, p):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fall back to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
