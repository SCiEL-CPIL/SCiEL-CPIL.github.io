"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 7 matrix.
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    The score is the rank (lower is better).
    """
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 42
    
    # Create the schoolbook decomposition
    U_school = np.zeros((n * m, rank_schoolbook), dtype=int)
    V_school = np.zeros((m * p, rank_schoolbook), dtype=int)
    W_school = np.zeros((p * n, rank_schoolbook), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    def check_decomposition(U, V, W, n, m, p):
        """Check if the decomposition exactly represents matrix multiplication."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        if V.shape[1] != rank or W.shape[1] != rank:
            return False
        # Check all nonzero positions
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a, b, c = i*m+j, j*p+k, k*n+i
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        # Check sampled zero positions
        for _ in range(500):
            a = np.random.randint(0, n*m)
            b = np.random.randint(0, m*p)
            c = np.random.randint(0, p*n)
            i, j = a // m, a % m
            j2, k = b // p, b % p
            k2, i2 = c // n, c % n
            if i == i2 and j == j2 and k == k2:
                continue
            val = np.dot(U[a, :], V[b, :] * W[c, :])
            if abs(val) > 1e-9:
                return False
        return True
    
    def gradient_descent(U, V, W, n, m, p, max_iter=200, lr=0.05):
        """Run gradient descent to minimize the decomposition error."""
        nonzero = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
        for it in range(max_iter):
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            # Nonzero positions: target = 1
            for i, j, k in nonzero:
                a, b, c = i*m+j, j*p+k, k*n+i
                ui, vj, wk = U[a, :], V[b, :], W[c, :]
                f = np.dot(ui, vj * wk)
                err = f - 1.0
                grad_U[a, :] += err * (vj * wk)
                grad_V[b, :] += err * (ui * wk)
                grad_W[c, :] += err * (ui * vj)
            # Zero positions: target = 0, sample 100
            for _ in range(100):
                a = np.random.randint(0, n*m)
                b = np.random.randint(0, m*p)
                c = np.random.randint(0, p*n)
                i, j = a // m, a % m
                j2, k = b // p, b % p
                k2, i2 = c // n, c % n
                if i == i2 and j == j2 and k == k2:
                    continue
                ui, vj, wk = U[a, :], V[b, :], W[c, :]
                f = np.dot(ui, vj * wk)
                grad_U[a, :] += f * (vj * wk)
                grad_V[b, :] += f * (ui * wk)
                grad_W[c, :] += f * (ui * vj)
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
        return U, V, W
    
    # Time budget: 100 seconds for search, 20 seconds safety margin
    deadline = time.time() + 100
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try ranks from 41 down to 21
    for rank in range(rank_schoolbook - 1, 20, -1):
        if time.time() > deadline - 10:
            break
        found = False
        for trial in range(15):
            if time.time() > deadline - 10:
                break
            np.random.seed(trial * 7919 + rank * 131)
            U = np.random.randn(n*m, rank) * 0.3
            V = np.random.randn(m*p, rank) * 0.3
            W = np.random.randn(p*n, rank) * 0.3
            U, V, W = gradient_descent(U, V, W, n, m, p, max_iter=150, lr=0.08)
            if check_decomposition(U, V, W, n, m, p):
                # Try rounding to integers
                Ur, Vr, Wr = np.round(U), np.round(V), np.round(W)
                if check_decomposition(Ur, Vr, Wr, n, m, p):
                    U, V, W = Ur, Vr, Wr
                best_rank = rank
                best_U, best_V, best_W = U, V, W
                found = True
                break
        if found:
            break
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
