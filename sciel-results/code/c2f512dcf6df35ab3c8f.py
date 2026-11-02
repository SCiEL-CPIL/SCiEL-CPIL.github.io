"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _verify_exact(U, V, W, n, m, p):
    """
    Verify that the tensor decomposition is exact using exact arithmetic.
    Only checks the n*m*p non-zero entries.
    """
    rank = len(U[0]) if U else 0
    # Convert to Fraction for exact arithmetic
    U_frac = [[Fraction(U[i][c]) for c in range(rank)] for i in range(n * m)]
    V_frac = [[Fraction(V[j][c]) for c in range(rank)] for j in range(m * p)]
    W_frac = [[Fraction(W[k][c]) for c in range(rank)] for k in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ui = i * m + j
                vj = j * p + k
                wk = k * n + i
                s = Fraction(0)
                for c in range(rank):
                    s += U_frac[ui][c] * V_frac[vj][c] * W_frac[wk][c]
                if s != 1:
                    return False
    return True


def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast float verification checking only non-zero entries."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ui = i * m + j
                vj = j * p + k
                wk = k * n + i
                s = 0.0
                for c in range(rank):
                    s += U[ui][c] * V[vj][c] * W[wk][c]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _als_optimize(U, V, W, T, nm, mp, pn, n, m, p, target_rank, max_iters=200, deadline=None):
    """
    Alternating Least Squares optimization for tensor decomposition.
    T is the target tensor (nm, mp, pn) with 1s at non-zero positions.
    Returns (U, V, W) if converged, else None.
    """
    r = target_rank
    Ur = U.copy()
    Vr = V.copy()
    Wr = W.copy()
    
    # Precompute target matrix views
    T_mat_u = T.reshape(nm, mp * pn)
    T_mat_v = T.reshape(mp, nm * pn)
    T_mat_w = T.reshape(pn, nm * mp)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            return None
        
        # Update U: for each r_idx, solve least squares
        for r_idx in range(r):
            # Compute residual excluding r_idx
            S_other = np.zeros((nm, mp * pn))
            for s in range(r):
                if s != r_idx:
                    S_other += np.outer(Ur[:, s], np.outer(Vr[:, s], Wr[:, s]).flatten())
            
            residual = T_mat_u - S_other
            vw = np.outer(Vr[:, r_idx], Wr[:, r_idx]).flatten()
            vw_norm_sq = np.dot(vw, vw)
            if vw_norm_sq < 1e-15:
                continue
            Ur[:, r_idx] = residual @ vw / vw_norm_sq
        
        # Update V
        for r_idx in range(r):
            S_other = np.zeros((mp, nm * pn))
            for s in range(r):
                if s != r_idx:
                    S_other += np.outer(Vr[:, s], np.outer(Ur[:, s], Wr[:, s]).flatten())
            
            residual = T_mat_v - S_other
            uw = np.outer(Ur[:, r_idx], Wr[:, r_idx]).flatten()
            uw_norm_sq = np.dot(uw, uw)
            if uw_norm_sq < 1e-15:
                continue
            Vr[:, r_idx] = residual @ uw / uw_norm_sq
        
        # Update W
        for r_idx in range(r):
            S_other = np.zeros((pn, nm * mp))
            for s in range(r):
                if s != r_idx:
                    S_other += np.outer(Wr[:, s], np.outer(Ur[:, s], Vr[:, s]).flatten())
            
            residual = T_mat_w - S_other
            uv = np.outer(Ur[:, r_idx], Vr[:, r_idx]).flatten()
            uv_norm_sq = np.dot(uv, uv)
            if uv_norm_sq < 1e-15:
                continue
            Wr[:, r_idx] = residual @ uv / uv_norm_sq
        
        # Check convergence every 10 iterations
        if it % 10 == 0:
            # Quick check using non-zero entries only
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        ui = i * m + j
                        vj = j * p + k
                        wk = k * n + i
                        s = 0.0
                        for c in range(r):
                            s += Ur[ui, c] * Vr[vj, c] * Wr[wk, c]
                        err = abs(s - 1.0)
                        if err > max_err:
                            max_err = err
            if max_err < 1e-10:
                return Ur, Vr, Wr
    
    # Final check
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ui = i * m + j
                vj = j * p + k
                wk = k * n + i
                s = 0.0
                for c in range(r):
                    s += Ur[ui, c] * Vr[vj, c] * Wr[wk, c]
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    if max_err < 1e-10:
        return Ur, Vr, Wr
    
    return None


def _try_rank_search(n, m, p, deadline):
    """
    Try to find a low-rank decomposition using ALS with various initializations.
    """
    nm, mp, pn = n * m, m * p, p * n
    
    # Build target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook for reference
    school = _schoolbook(n, m, p)
    U_s = np.array(school["U"], dtype=float)
    V_s = np.array(school["V"], dtype=float)
    W_s = np.array(school["W"], dtype=float)
    
    best_result = None
    best_rank = n * m * p
    
    # Lower bound: max(n*m, m*p, p*n) = max(12, 28, 21) = 28
    # Try ranks from lower bound upward
    candidate_ranks = list(range(28, 85))
    
    for target_rank in candidate_ranks:
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            continue
        
        # Try multiple random initializations
        for attempt in range(30):
            if time.time() > deadline:
                break
            
            # Different initialization strategies
            if attempt == 0:
                # From schoolbook, pick target_rank random columns
                cols = np.random.choice(n*m*p, target_rank, replace=False)
                Ur = U_s[:, cols].copy()
                Vr = V_s[:, cols].copy()
                Wr = W_s[:, cols].copy()
            elif attempt == 1:
                # Sparse random initialization
                Ur = np.random.randn(nm, target_rank) * 0.5
                Vr = np.random.randn(mp, target_rank) * 0.5
                Wr = np.random.randn(pn, target_rank) * 0.5
            elif attempt == 2:
                # Binary random initialization
                Ur = (np.random.rand(nm, target_rank) < 0.3).astype(float)
                Vr = (np.random.rand(mp, target_rank) < 0.3).astype(float)
                Wr = (np.random.rand(pn, target_rank) < 0.3).astype(float)
            elif attempt == 3:
                # Structured initialization based on block structure
                Ur = np.zeros((nm, target_rank))
                Vr = np.zeros((mp, target_rank))
                Wr = np.zeros((pn, target_rank))
                for i in range(n):
                    for j in range(m):
                        for c in range(min(target_rank, n*m*p // (n*m))):
                            Ur[i*m+j, c] = 1.0
                for j in range(m):
                    for k in range(p):
                        for c in range(min(target_rank, n*m*p // (m*p))):
                            Vr[j*p+k, c] = 1.0
                for k in range(p):
                    for i in range(n):
                        for c in range(min(target_rank, n*m*p // (p*n))):
                            Wr[k*n+i, c] = 1.0
            else:
                # Random perturbation of schoolbook subset
                cols = np.random.choice(n*m*p, min(target_rank, n*m*p), replace=False)
                Ur = U_s[:, cols].copy() + np.random.randn(nm, len(cols)) * 0.1
                Vr = V_s[:, cols].copy() + np.random.randn(mp, len(cols)) * 0.1
                Wr = W_s[:, cols].copy() + np.random.randn(pn, len(cols)) * 0.1
                # Trim to target_rank
                if Ur.shape[1] > target_rank:
                    Ur = Ur[:, :target_rank]
                    Vr = Vr[:, :target_rank]
                    Wr = Wr[:, :target_rank]
            
            # Run ALS optimization
            result = _als_optimize(Ur, Vr, Wr, T, nm, mp, pn, n, m, p, target_rank, 
                                   max_iters=150, deadline=deadline)
            
            if result is not None:
                Ur, Vr, Wr = result
                # Convert to lists for exact verification
                U_list = [[float(Ur[i][c]) for c in range(target_rank)] for i in range(nm)]
                V_list = [[float(Vr[j][c]) for c in range(target_rank)] for j in range(mp)]
                W_list = [[float(Wr[k][c]) for c in range(target_rank)] for k in range(pn)]
                
                # Fast float check first
                if _verify_fast(U_list, V_list, W_list, n, m, p, tol=1e-8):
                    # Exact check
                    if _verify_exact(U_list, V_list, W_list, n, m, p):
                        best_rank = target_rank
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
                        break
        
        if best_result is not None:
            break
    
    return best_result


def _try_polish(n, m, p, U, V, W, deadline):
    """
    Try to polish a near-exact solution to get exact integer/half-integer entries.
    """
    nm, mp, pn = n * m, m * p, p * n
    rank = len(U[0])
    
    # Try rounding to nearest half-integer
    U_round = [[round(U[i][c] * 2) / 2 for c in range(rank)] for i in range(nm)]
    V_round = [[round(V[j][c] * 2) / 2 for c in range(rank)] for j in range(mp)]
    W_round = [[round(W[k][c] * 2) / 2 for c in range(rank)] for k in range(pn)]
    
    if _verify_fast(U_round, V_round, W_round, n, m, p, tol=1e-8):
        if _verify_exact(U_round, V_round, W_round, n, m, p):
            return {"U": U_round, "V": V_round, "W": W_round}
    
    # Try rounding to nearest integer
    U_int = [[round(U[i][c]) for c in range(rank)] for i in range(nm)]
    V_int = [[round(V[j][c]) for c in range(rank)] for j in range(mp)]
    W_int = [[round(W[k][c]) for c in range(rank)] for k in range(pn)]
    
    if _verify_fast(U_int, V_int, W_int, n, m, p, tol=1e-8):
        if _verify_exact(U_int, V_int, W_int, n, m, p):
            return {"U": U_int, "V": V_int, "W": W_int}
    
    return None


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s kill

    # Start with schoolbook
    result = _schoolbook(n, m, p)
    best_rank = n * m * p

    # Try rank search with ALS
    try:
        improved = _try_rank_search(n, m, p, deadline)
        if improved:
            r = len(improved["U"][0])
            if r < best_rank:
                best_rank = r
                result = improved
    except Exception:
        pass

    # Try to polish if we found something better than schoolbook
    if best_rank < n * m * p and time.time() < deadline - 5:
        try:
            polished = _try_polish(n, m, p, result["U"], result["V"], result["W"], deadline)
            if polished:
                result = polished
        except Exception:
            pass

    if time.time() > deadline:
        return result

    return result
# EVOLVE-BLOCK-END
