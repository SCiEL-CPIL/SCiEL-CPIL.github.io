import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

def _build_target_tensor(n, m, p):
    return np.ones((n, m, p), dtype=np.float64)

def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    rank = U.shape[1]
    T = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        T += np.einsum('ij,jk,ki->ijk', u, v, w)
    return np.allclose(T, 1.0, atol=tol, rtol=0)

def _construct_structured_decomposition():
    """Construct a rank-44 decomposition."""
    n, m, p = 3, 4, 4
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    
    def add_strassen(rA, cA, rB, cB):
        nonlocal col
        # 7 Strassen terms
        # M1 = (X11+X22)(Y11+Y22)
        U[(rA+0)*m + (cA+0), col] = 1
        U[(rA+1)*m + (cA+1), col] = 1
        V[(rB+0)*p + (cB+0), col] = 1
        V[(rB+1)*p + (cB+1), col] = 1
        W[(cB+0)*n + (rA+0), col] = 1
        W[(cB+1)*n + (rA+1), col] = 1
        col += 1
        # M2 = (X21+X22)Y11
        U[(rA+1)*m + (cA+0), col] = 1
        U[(rA+1)*m + (cA+1), col] = 1
        V[(rB+0)*p + (cB+0), col] = 1
        W[(cB+0)*n + (rA+1), col] = 1
        W[(cB+1)*n + (rA+1), col] = -1
        col += 1
        # M3 = X11(Y12-Y22)
        U[(rA+0)*m + (cA+0), col] = 1
        V[(rB+0)*p + (cB+1), col] = 1
        V[(rB+1)*p + (cB+1), col] = -1
        W[(cB+1)*n + (rA+0), col] = 1
        W[(cB+1)*n + (rA+1), col] = 1
        col += 1
        # M4 = X22(Y21-Y11)
        U[(rA+1)*m + (cA+1), col] = 1
        V[(rB+1)*p + (cB+0), col] = 1
        V[(rB+0)*p + (cB+0), col] = -1
        W[(cB+0)*n + (rA+0), col] = 1
        W[(cB+0)*n + (rA+1), col] = 1
        col += 1
        # M5 = (X11+X12)Y22
        U[(rA+0)*m + (cA+0), col] = 1
        U[(rA+0)*m + (cA+1), col] = 1
        V[(rB+1)*p + (cB+1), col] = 1
        W[(cB+1)*n + (rA+0), col] = 1
        W[(cB+0)*n + (rA+0), col] = -1
        col += 1
        # M6 = (X21-X11)(Y11+Y12)
        U[(rA+1)*m + (cA+0), col] = 1
        U[(rA+0)*m + (cA+0), col] = -1
        V[(rB+0)*p + (cB+0), col] = 1
        V[(rB+0)*p + (cB+1), col] = 1
        W[(cB+1)*n + (rA+1), col] = 1
        col += 1
        # M7 = (X12-X22)(Y21+Y22)
        U[(rA+0)*m + (cA+1), col] = 1
        U[(rA+1)*m + (cA+1), col] = -1
        V[(rB+1)*p + (cB+0), col] = 1
        V[(rB+1)*p + (cB+1), col] = 1
        W[(cB+0)*n + (rA+0), col] = 1
        col += 1

    def add_schoolbook_1x2x2(rA, cA, rB, cB):
        nonlocal col
        U[(rA+0)*m + (cA+0), col] = 1
        V[(rB+0)*p + (cB+0), col] = 1
        W[(cB+0)*n + (rA+0), col] = 1
        col += 1
        U[(rA+0)*m + (cA+1), col] = 1
        V[(rB+1)*p + (cB+0), col] = 1
        W[(cB+0)*n + (rA+0), col] = 1
        col += 1
        U[(rA+0)*m + (cA+0), col] = 1
        V[(rB+0)*p + (cB+1), col] = 1
        W[(cB+1)*n + (rA+0), col] = 1
        col += 1
        U[(rA+0)*m + (cA+1), col] = 1
        V[(rB+1)*p + (cB+1), col] = 1
        W[(cB+1)*n + (rA+0), col] = 1
        col += 1

    add_strassen(0, 0, 0, 0)
    add_strassen(0, 2, 2, 0)
    add_strassen(0, 0, 0, 2)
    add_strassen(0, 2, 2, 2)
    add_schoolbook_1x2x2(2, 0, 0, 0)
    add_schoolbook_1x2x2(2, 2, 2, 0)
    add_schoolbook_1x2x2(2, 0, 0, 2)
    add_schoolbook_1x2x2(2, 2, 2, 2)
    
    return U, V, W

def _als_decomposition(T, n, m, p, rank, max_iter=200, tol=1e-8):
    """ALS decomposition of tensor T into rank-r sum of outer products."""
    U = np.random.randn(n*m, rank) * 0.1
    V = np.random.randn(m*p, rank) * 0.1
    W = np.random.randn(p*n, rank) * 0.1
    
    for it in range(max_iter):
        # Update U
        for r in range(rank):
            # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # For fixed r, U[i*m+j, r] = sum_{j,k} T[i,j,k] * V[j*p+k, r] * W[k*n+i, r] / sum_{j,k} V[j*p+k,r]^2 * W[k*n+i,r]^2
            num = np.zeros(n*m)
            den = np.zeros(n*m)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        num[i*m+j] += T[i,j,k] * V[j*p+k, r] * W[k*n+i, r]
                        den[i*m+j] += V[j*p+k, r]**2 * W[k*n+i, r]**2
            mask = den > 1e-12
            U[mask, r] = num[mask] / den[mask]
        
        # Update V
        for r in range(rank):
            num = np.zeros(m*p)
            den = np.zeros(m*p)
            for j in range(m):
                for k in range(p):
                    for i in range(n):
                        num[j*p+k] += T[i,j,k] * U[i*m+j, r] * W[k*n+i, r]
                        den[j*p+k] += U[i*m+j, r]**2 * W[k*n+i, r]**2
            mask = den > 1e-12
            V[mask, r] = num[mask] / den[mask]
        
        # Update W
        for r in range(rank):
            num = np.zeros(p*n)
            den = np.zeros(p*n)
            for k in range(p):
                for i in range(n):
                    for j in range(m):
                        num[k*n+i] += T[i,j,k] * U[i*m+j, r] * V[j*p+k, r]
                        den[k*n+i] += U[i*m+j, r]**2 * V[j*p+k, r]**2
            mask = den > 1e-12
            W[mask, r] = num[mask] / den[mask]
        
        # Check convergence
        T_recon = np.zeros((n, m, p))
        for r in range(rank):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            T_recon += np.einsum('ij,jk,ki->ijk', u, v, w)
        if np.max(np.abs(T_recon - T)) < tol:
            break
    
    return U, V, W

def _round_to_halfint(X):
    """Round to nearest half-integer."""
    return np.round(X * 2) / 2.0

def _is_halfinteger(X, tol=1e-6):
    """Check if all entries are half-integers."""
    return np.allclose(X * 2, np.round(X * 2), atol=tol)

def _try_rank_r(T, n, m, p, r, n_trials=50, time_budget=30):
    """Try to find a rank-r decomposition with half-integer entries."""
    start_time = time.time()
    for trial in range(n_trials):
        if time.time() - start_time > time_budget:
            break
        U, V, W = _als_decomposition(T, n, m, p, r, max_iter=100, tol=1e-7)
        # Round to half-integers
        U_r = _round_to_halfint(U)
        V_r = _round_to_halfint(V)
        W_r = _round_to_halfint(W)
        # Verify
        if _verify_decomposition(U_r, V_r, W_r, n, m, p, tol=1e-4):
            return U_r, V_r, W_r
    return None

def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS
    T = _build_target_tensor(n, m, p)
    
    # Start with the structured rank-44 decomposition
    U44, V44, W44 = _construct_structured_decomposition()
    best_U, best_V, best_W = U44, V44, W44
    best_rank = 44
    
    # Try to find lower-rank decompositions
    for r in range(43, 30, -1):
        if time.time() > deadline - 20:
            break
        result = _try_rank_r(T, n, m, p, r, n_trials=20, time_budget=10)
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U, V, W
            best_rank = r
            break
    
    # Convert to lists
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
