"""Matrix multiplication 2x3x10 - rank-minimal exact tensor decomposition.

Combines the schoolbook rank-60 baseline with an alternating-minimization
refinement: fix two factor matrices, solve for the third via a linear least-
squares subproblem, iterate, and verify exactness on random matrices.
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def _normalize_factors(U, V, W):
    """Normalize columns of factor matrices to unit norm."""
    for M in [U, V, W]:
        norms = np.linalg.norm(M, axis=0)
        norms[norms < 1e-12] = 1.0
        M /= norms
    return U, V, W


def _snap_to_grid(U, V, W):
    """Snap entries to nearest integer or half-integer."""
    U = np.round(2 * U) / 2.0
    V = np.round(2 * V) / 2.0
    W = np.round(2 * W) / 2.0
    return U, V, W


def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _solve_W(U, V, n, m, p):
    """For fixed U, V, find W minimizing the tensor error (linear LS)."""
    W = np.zeros((p * n, U.shape[1]))
    for i in range(n):
        for k in range(p):
            A = np.zeros((m, U.shape[1]))
            for j in range(m):
                A[j, :] = U[i * m + j, :] * V[j * p + k, :]
            b = np.ones(m)
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
            W[k * n + i, :] = sol
    return W


def _solve_U(W, V, n, m, p):
    """For fixed W, V, find U minimizing the tensor error (linear LS)."""
    U = np.zeros((n * m, W.shape[1]))
    for i in range(n):
        for j in range(m):
            A = np.zeros((p, W.shape[1]))
            for k in range(p):
                A[k, :] = W[k * n + i, :] * V[j * p + k, :]
            b = np.ones(p)
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
            U[i * m + j, :] = sol
    return U


def _solve_V(U, W, n, m, p):
    """For fixed U, W, find V minimizing the tensor error (linear LS)."""
    V = np.zeros((m * p, U.shape[1]))
    for j in range(m):
        for k in range(p):
            A = np.zeros((n, U.shape[1]))
            for i in range(n):
                A[i, :] = U[i * m + j, :] * W[k * n + i, :]
            b = np.ones(n)
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
            V[j * p + k, :] = sol
    return V


def _verify(U, V, W, n, m, p, ntests=50, tol=1e-9):
    """Check exactness: for random A, B, does C = A@B equal the tensor contraction?"""
    rng = np.random.default_rng(12345)
    for _ in range(ntests):
        A = rng.standard_normal((n, m))
        B = rng.standard_normal((m, p))
        C_true = A @ B
        # C[i,k] = sum_j A[i,j] B[j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
        # Build via einsum over the flattened factors
        C_calc = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    s += np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                C_calc[i, k] = s
        if np.max(np.abs(C_calc - C_true)) > tol:
            return False
    return True


def _polish(U, V, W, n, m, p, deadline, max_iter=500):
    """High-precision polish: run alternating LS with tighter tolerance and more iterations."""
    U, V, W = U.copy(), V.copy(), W.copy()
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        U = _solve_U(W, V, n, m, p)
        V = _solve_V(U, W, n, m, p)
        W = _solve_W(U, V, n, m, p)
        U, V, W = _normalize_factors(U, V, W)
        
        # Compute error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    err += (val - 1.0) ** 2
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        if err < 1e-14:
            break
    
    return best_U, best_V, best_W


def _try_rank(r, n, m, p, deadline, max_iter=2000):
    """Attempt to find an exact rank-r decomposition via alternating LS."""
    best = None
    best_err = float('inf')
    # Increase restarts to improve chance of finding a good local minimum
    for trial in range(10):
        if time.time() > deadline:
            break
        rng = np.random.default_rng(1000 + trial)
        U = rng.standard_normal((n * m, r))
        V = rng.standard_normal((m * p, r))
        W = _solve_W(U, V, n, m, p)
        U, V, W = _normalize_factors(U, V, W)
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            # Alternating updates
            U = _solve_U(W, V, n, m, p)
            V = _solve_V(U, W, n, m, p)
            W = _solve_W(U, V, n, m, p)
            U, V, W = _normalize_factors(U, V, W)
            
            # Compute error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                        err += (val - 1.0) ** 2
            
            if err < best_err:
                best_err = err
                best = (U.copy(), V.copy(), W.copy())
            if err < 1e-12:
                break
        
        if best is not None:
            Uv, Vv, Wv = best
            # High-precision polish pass
            Uv, Vv, Wv = _polish(Uv, Vv, Wv, n, m, p, deadline, max_iter=200)
            # Try snapping to grid before verification
            Us, Vs, Ws = _snap_to_grid(Uv, Vv, Wv)
            if _verify(Us, Vs, Ws, n, m, p, ntests=10):
                # Final verification with more tests
                if _verify(Us, Vs, Ws, n, m, p, ntests=50):
                    return Us, Vs, Ws
            # If snapping failed, try original
            if _verify(Uv, Vv, Wv, n, m, p, ntests=10):
                if _verify(Uv, Vv, Wv, n, m, p, ntests=50):
                    return Uv, Vv, Wv
    return None


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # safety margin before 150s limit

    # Baseline: schoolbook rank 60
    U_sb, V_sb, W_sb = _build_schoolbook(n, m, p)
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p

    # Theoretical lower bound: n*m + m*p + n*p - m = 6 + 30 + 20 - 3 = 53
    lower_bound = n * m + m * p + n * p - m

    # Try to improve from rank 59 down toward the lower bound
    # Focus on ranks near the lower bound where improvements are most likely
    for r in range(best_rank - 1, lower_bound - 1, -1):
        if time.time() > deadline:
            break
        # Allocate time proportional to remaining budget, with minimum for lower ranks
        remaining_ranks = best_rank - r
        time_per_rank = max(5.0, (deadline - time.time()) / max(1, remaining_ranks))
        result = _try_rank(r, n, m, p, time.time() + time_per_rank)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = r
            # Stop at the lower bound
            if r <= lower_bound:
                break

    # Convert to lists of lists of numbers (integers or floats)
    def to_list(M):
        return [[float(x) for x in row] for row in M]

    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
