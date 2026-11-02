"""Matrix multiplication 3x3 by 3x6: rank-minimal exact tensor decomposition."""
import numpy as np
import time
import itertools
import random

DIMENSIONS = [3, 3, 6]
DEADLINE = 130  # seconds, safety margin before 150s kill

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 9, 18, 18

    # Build target tensor T[a, b, c] where a in [0,9), b in [0,18), c in [0,18)
    # T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    # Otherwise 0
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Schoolbook rank = 54
    rank_schoolbook = n * m * p
    U_sb = np.zeros((nm, rank_schoolbook))
    V_sb = np.zeros((mp, rank_schoolbook))
    W_sb = np.zeros((pn, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    def verify(U, V, W):
        """Check if U,V,W exactly decompose T. Returns True if valid."""
        T_rec = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.allclose(T_rec, T, atol=1e-9)

    def rank_of_decomp(U, V, W):
        return U.shape[1]

    best_rank = rank_schoolbook
    best_result = {'U': U_sb.tolist(), 'V': V_sb.tolist(), 'W': W_sb.tolist()}

    start_time = time.time()

    def try_rank_structured(r, max_iters=200, seed=0):
        """Try to find a rank-r decomposition using structured initialization + refinement."""
        if time.time() - start_time > DEADLINE - 10:
            return None
        rng = np.random.RandomState(seed)
        best_err = np.inf
        best_U, best_V, best_W = None, None, None

        # Strategy 1: Start from schoolbook and randomly merge columns
        # Strategy 2: Random init with structure-aware perturbation
        # Strategy 3: Use known 3x3x3 rank-23 as building blocks

        for trial in range(5):
            if time.time() - start_time > DEADLINE - 10:
                return None

            if trial == 0:
                # Random initialization
                U = rng.randn(nm, r)
                V = rng.randn(mp, r)
                W = rng.randn(pn, r)
            elif trial == 1:
                # Initialize with some structure: first r columns of schoolbook
                U = U_sb[:, :r].copy()
                V = V_sb[:, :r].copy()
                W = W_sb[:, :r].copy()
                # Add small random perturbation
                U += 0.1 * rng.randn(nm, r)
                V += 0.1 * rng.randn(mp, r)
                W += 0.1 * rng.randn(pn, r)
            elif trial == 2:
                # Hybrid: use half schoolbook, half random
                half = r // 2
                U = np.hstack([U_sb[:, :half], rng.randn(nm, r - half)])
                V = np.hstack([V_sb[:, :half], rng.randn(mp, r - half)])
                W = np.hstack([W_sb[:, :half], rng.randn(pn, r - half)])
            elif trial == 3:
                # Sparse random init
                U = np.zeros((nm, r))
                V = np.zeros((mp, r))
                W = np.zeros((pn, r))
                for col in range(r):
                    # Each column gets a few non-zero entries
                    nz_u = rng.randint(1, 4)
                    nz_v = rng.randint(1, 6)
                    nz_w = rng.randint(1, 6)
                    idx_u = rng.choice(nm, size=nz_u, replace=False)
                    idx_v = rng.choice(mp, size=nz_v, replace=False)
                    idx_w = rng.choice(pn, size=nz_w, replace=False)
                    U[idx_u, col] = rng.choice([-1, 1])
                    V[idx_v, col] = rng.choice([-1, 1])
                    W[idx_w, col] = rng.choice([-1, 1])
            else:
                # Binary-like init
                U = rng.choice([-1, 0, 1], size=(nm, r))
                V = rng.choice([-1, 0, 1], size=(mp, r))
                W = rng.choice([-1, 0, 1], size=(pn, r))

            # Normalize columns
            for col in range(r):
                norm = np.sqrt(np.sum(U[:, col]**2) + np.sum(V[:, col]**2) + np.sum(W[:, col]**2))
                if norm > 1e-10:
                    U[:, col] /= norm
                    V[:, col] /= norm
                    W[:, col] /= norm

            for it in range(max_iters):
                if time.time() - start_time > DEADLINE - 10:
                    return None

                T_rec = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
                err = np.sum((T_rec - T)**2)

                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

                if err < 1e-12:
                    break

                # Compute gradients
                R = T - T_rec
                RW = np.einsum('abc,cr->abr', R, W, optimize=True)
                dU = -2.0 * np.einsum('br,abr->ar', V, RW, optimize=True)

                RU = np.einsum('abc,br->acr', R, V, optimize=True)
                dW = -2.0 * np.einsum('ar,acr->cr', U, RU, optimize=True)

                RV = np.einsum('abc,ar->bcr', R, U, optimize=True)
                dV = -2.0 * np.einsum('cr,bcr->br', W, RV, optimize=True)

                # Adaptive learning rate with momentum-like decay
                lr = 0.05 / (1.0 + it * 0.05)
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW

                # Re-normalize
                for col in range(r):
                    norm = np.sqrt(np.sum(U[:, col]**2) + np.sum(V[:, col]**2) + np.sum(W[:, col]**2))
                    if norm > 1e-10:
                        U[:, col] /= norm
                        V[:, col] /= norm
                        W[:, col] /= norm

            if best_err < 1e-12:
                break

        if best_U is not None and best_err < 1e-9:
            return best_U, best_V, best_W, best_err
        return None

    # Try ranks from lower bound upward
    # Lower bound for 3x3x3x6: at least max(9, 18, 18) = 18, realistically ~36-40
    for r in range(36, 55):
        if time.time() - start_time > DEADLINE - 15:
            break
        result = try_rank_structured(r, max_iters=150, seed=r)
        if result is not None:
            U, V, W, err = result
            if verify(U, V, W):
                best_rank = r
                best_result = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
                break

    # Check time
    if time.time() - start_time > DEADLINE - 5:
        return best_result

    return best_result
