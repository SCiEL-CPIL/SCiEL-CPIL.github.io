import time
import sys

import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 4, 4]


def _build_T(n, m, p):
    """Multiplication tensor T[a,b,c]=1 iff a=i*m+j, b=j*p+k, c=k*n+i."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _verify(U, V, W, T, tol=1e-7):
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    M = np.einsum('ar,br,cr->abc', U, V, W)
    return np.max(np.abs(M - T)) < tol


def _strassen_solution(n, m, p):
    """Known rank-28 exact decomposition via Strassen on four 2x2 blocks."""
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def add(A_row0, A_col0, B_row0, B_col0, C_row0, C_col0, start_r):
        a11 = A_row0 * m + A_col0
        a12 = A_row0 * m + A_col0 + 1
        a21 = (A_row0 + 1) * m + A_col0
        a22 = (A_row0 + 1) * m + A_col0 + 1
        b11 = B_row0 * p + B_col0
        b12 = B_row0 * p + B_col0 + 1
        b21 = (B_row0 + 1) * p + B_col0
        b22 = (B_row0 + 1) * p + B_col0 + 1
        c11 = C_col0 * n + C_row0
        c12 = (C_col0 + 1) * n + C_row0
        c21 = C_col0 * n + (C_row0 + 1)
        c22 = (C_col0 + 1) * n + (C_row0 + 1)
        r = start_r
        # M1 = (A11+A22)(B11+B22)
        U[a11][r] += 1; U[a22][r] += 1; V[b11][r] += 1; V[b22][r] += 1
        W[c11][r] += 1; W[c22][r] += 1; r += 1
        # M2 = (A21+A22)B11
        U[a21][r] += 1; U[a22][r] += 1; V[b11][r] += 1
        W[c21][r] += 1; W[c22][r] += -1; r += 1
        # M3 = A11(B12-B22)
        U[a11][r] += 1; V[b12][r] += 1; V[b22][r] += -1
        W[c12][r] += 1; W[c22][r] += 1; r += 1
        # M4 = A22(B21-B11)
        U[a22][r] += 1; V[b21][r] += 1; V[b11][r] += -1
        W[c11][r] += 1; W[c21][r] += 1; r += 1
        # M5 = (A11+A12)B22
        U[a11][r] += 1; U[a12][r] += 1; V[b22][r] += 1
        W[c11][r] += -1; W[c12][r] += 1; r += 1
        # M6 = (A21-A11)(B11+B12)
        U[a21][r] += 1; U[a11][r] += -1; V[b11][r] += 1; V[b12][r] += 1
        W[c22][r] += 1; r += 1
        # M7 = (A12-A22)(B21+B22)
        U[a12][r] += 1; U[a22][r] += -1; V[b21][r] += 1; V[b22][r] += 1
        W[c11][r] += 1; r += 1
        return r

    r = add(0, 0, 0, 0, 0, 0, 0)
    r = add(0, 2, 2, 0, 0, 0, r)
    r = add(0, 0, 0, 2, 0, 2, r)
    r = add(0, 2, 2, 2, 0, 2, r)
    assert r == 28
    return {"U": U, "V": V, "W": W}


def _round_to_half(U, V, W):
    """Round entries to nearest half-integer (integer/half-integer constraint)."""
    def rM(M):
        return np.round(2.0 * M) / 2.0
    return rM(U), rM(V), rM(W)


def _try_eliminate_column(U, V, W, T, col_idx, deadline):
    """Try to eliminate column col_idx by redistributing its contribution to other columns."""
    R = U.shape[1]
    if R <= 1:
        return None
    
    # Compute the contribution of column col_idx
    contrib = np.einsum('a,b,c->abc', U[:, col_idx], V[:, col_idx], W[:, col_idx])
    
    # Try to find adjustments to other columns that cancel this contribution
    # We'll use a small perturbation approach
    A, B, C = U.shape[0], V.shape[0], W.shape[0]
    
    # Create a mask for columns except col_idx
    other_cols = [i for i in range(R) if i != col_idx]
    
    # For each other column, try small perturbations
    best_delta = None
    best_norm = np.inf
    
    for target_col in other_cols:
        # Try to shift some contribution from col_idx to target_col
        # This is a linear system: find dU, dV, dW such that
        # dU*V[t]*W[t] + U*dV*W[t] + U*V[t]*dW ≈ -contrib
        # For simplicity, try scaling target_col by a factor
        
        # Compute current contribution of target_col
        contrib_t = np.einsum('a,b,c->abc', U[:, target_col], V[:, target_col], W[:, target_col])
        
        # We want: contrib + alpha * contrib_t = 0 for some alpha
        # This is generally not possible, but we can try to minimize the residual
        
        # Try a grid of alpha values
        for alpha in np.linspace(-2, 2, 81):
            residual = contrib + alpha * contrib_t
            norm = np.max(np.abs(residual))
            if norm < best_norm:
                best_norm = norm
                best_delta = (target_col, alpha)
    
    if best_delta is None or best_norm > 0.1:
        return None
    
    target_col, alpha = best_delta
    
    # Apply the perturbation
    U_new = U.copy()
    V_new = V.copy()
    W_new = W.copy()
    
    # Scale target_col by (1 + alpha)
    U_new[:, target_col] *= (1 + alpha)
    V_new[:, target_col] *= (1 + alpha)
    W_new[:, target_col] *= (1 + alpha)
    
    # Remove col_idx
    mask = [i for i in range(R) if i != col_idx]
    U_new = U_new[:, mask]
    V_new = V_new[:, mask]
    W_new = W_new[:, mask]
    
    # Verify
    M = np.einsum('ar,br,cr->abc', U_new, V_new, W_new)
    if np.max(np.abs(M - T)) < 1e-5:
        Uq, Vq, Wq = _round_to_half(U_new, V_new, W_new)
        if _verify(Uq, Vq, Wq, T, tol=1e-7):
            return {"U": Uq, "V": Vq, "W": Wq}
    
    return None


def _search_rank(T, n, m, p, R, seed, deadline, start_from):
    """Try to find a rank-R exact decomposition."""
    A = n * m
    B = m * p
    C = p * n

    def _resid(pv):
        Uf = pv[:A * R].reshape(A, R)
        Vf = pv[A * R:A * R + B * R].reshape(B, R)
        Wf = pv[A * R + B * R:].reshape(C, R)
        M = np.einsum('ar,br,cr->abc', Uf, Vf, Wf)
        return (M - T).reshape(-1)

    rng = np.random.default_rng(seed)

    # Start from Strassen solution (padded with zeros)
    U0 = np.asarray(start_from["U"], dtype=float)
    V0 = np.asarray(start_from["V"], dtype=float)
    W0 = np.asarray(start_from["W"], dtype=float)
    
    # If R == current rank, we have an exact solution
    # If R < current rank, we need to find a lower-rank decomposition
    
    if R == U0.shape[1]:
        return {"U": U0, "V": V0, "W": W0}
    
    # Strategy 1: Try to eliminate columns from the Strassen solution
    if R < U0.shape[1]:
        U_curr, V_curr, W_curr = U0.copy(), V0.copy(), W0.copy()
        current_rank = U_curr.shape[1]
        
        while current_rank > R:
            if time.time() >= deadline:
                break
            
            # Try to eliminate each column
            eliminated = False
            for col in range(current_rank):
                if time.time() >= deadline:
                    break
                
                result = _try_eliminate_column(U_curr, V_curr, W_curr, T, col, deadline)
                if result is not None:
                    U_curr, V_curr, W_curr = result["U"], result["V"], result["W"]
                    current_rank = U_curr.shape[1]
                    eliminated = True
                    break
            
            if not eliminated:
                break
        
        if current_rank == R:
            Uq, Vq, Wq = _round_to_half(U_curr, V_curr, W_curr)
            if _verify(Uq, Vq, Wq, T, tol=1e-7):
                return {"U": Uq, "V": Vq, "W": Wq}
    
    # Strategy 2: Random initialization with least_squares
    x0 = rng.standard_normal(A * R + B * R + C * R)
    Uf = x0[:A*R].reshape(A,R)
    Vf = x0[A*R:A*R+B*R].reshape(B,R)
    Wf = x0[A*R+B*R:].reshape(C,R)
    for r in range(R):
        for M in (Uf, Vf, Wf):
            nrm = np.linalg.norm(M[:, r])
            if nrm > 1e-12:
                M[:, r] /= nrm
    x0 = np.concatenate([Uf.reshape(-1), Vf.reshape(-1), Wf.reshape(-1)])

    try:
        sol = least_squares(_resid, x0, method='trf', max_nfev=300,
                            xtol=1e-12, ftol=1e-12, gtol=1e-12)
        if np.max(np.abs(sol.fun)) < 1e-6:
            Uf = sol.x[:A * R].reshape(A, R)
            Vf = sol.x[A * R:A * R + B * R].reshape(B, R)
            Wf = sol.x[A * R + B * R:].reshape(C, R)
            Uq, Vq, Wq = _round_to_half(Uf, Vf, Wf)
            if _verify(Uq, Vq, Wq, T, tol=1e-7):
                return {"U": Uq, "V": Vq, "W": Wq}
    except Exception:
        pass

    # Try a few random restarts
    for i in range(10):
        if time.time() >= deadline:
            break
        x0 = rng.standard_normal(A * R + B * R + C * R)
        Uf = x0[:A*R].reshape(A,R)
        Vf = x0[A*R:A*R+B*R].reshape(B,R)
        Wf = x0[A*R+B*R:].reshape(C,R)
        for r in range(R):
            for M in (Uf, Vf, Wf):
                nrm = np.linalg.norm(M[:, r])
                if nrm > 1e-12:
                    M[:, r] /= nrm
        x0 = np.concatenate([Uf.reshape(-1), Vf.reshape(-1), Wf.reshape(-1)])
        try:
            sol = least_squares(_resid, x0, method='trf', max_nfev=300,
                                xtol=1e-12, ftol=1e-12, gtol=1e-12)
            if np.max(np.abs(sol.fun)) < 1e-6:
                Uf = sol.x[:A * R].reshape(A, R)
                Vf = sol.x[A * R:A * R + B * R].reshape(B, R)
                Wf = sol.x[A * R + B * R:].reshape(C, R)
                Uq, Vq, Wq = _round_to_half(Uf, Vf, Wf)
                if _verify(Uq, Vq, Wq, T, tol=1e-7):
                    return {"U": Uq, "V": Wq, "W": Wq}
        except Exception:
            pass
            
    return None


def construct():
    n, m, p = DIMENSIONS
    t_start = time.time()
    deadline = t_start + 140.0   # safety margin before 150s limit

    T = _build_T(n, m, p)
    best = _strassen_solution(n, m, p)   # guaranteed rank-28 fallback

    # Try rank 24
    for seed in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15):
        if time.time() >= deadline - 10.0:
            break
        try:
            sol = _search_rank(T, n, m, p, 24, seed, deadline - 10.0, best)
        except Exception:
            sol = None
        if sol is not None and _verify(sol["U"], sol["V"], sol["W"], T, tol=1e-7):
            best = sol
            break

    # If still 28, try rank 21?
    if len(best["U"][0]) == 28:
        for seed in (0, 1, 2, 3):
            if time.time() >= deadline - 5.0:
                break
            try:
                sol = _search_rank(T, n, m, p, 21, seed, deadline - 5.0, best)
            except Exception:
                sol = None
            if sol is not None and _verify(sol["U"], sol["V"], sol["W"], T, tol=1e-7):
                best = sol
                break

    return best
