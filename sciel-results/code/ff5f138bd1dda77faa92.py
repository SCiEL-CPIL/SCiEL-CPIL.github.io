"""Matrix multiplication 3x3x5: rank-minimal exact tensor decomposition.

Uses smooth optimization followed by rational reconstruction to find
exact integer/half-integer decompositions.
"""

import time
import sys
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START

def _compute_residual(U, V, W, n, m, p, rank):
    """Compute residual tensor R where R[i,j,k] = 1 - sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]."""
    R = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        u_col = U[:, r]
        v_col = V[:, r]
        w_col = W[:, r]
        for i in range(n):
            for j in range(m):
                u_val = u_col[i * m + j]
                if u_val != 0:
                    v_slice = v_col[j * p:(j + 1) * p]
                    w_slice = w_col[i::n]
                    R[i, j, :] += u_val * v_slice * w_slice
    return 1.0 - R

def _verify_solution(U, V, W, n, m, p, rank, tol=1e-9):
    """Verify that the decomposition is exact."""
    R = _compute_residual(U, V, W, n, m, p, rank)
    return np.max(np.abs(R)) < tol

def _rational_reconstruct(x, max_denom=100):
    """
    Attempt to reconstruct a rational number from a float.
    Returns a Fraction or None if no good rational approximation found.
    """
    if abs(x) < 1e-12:
        return Fraction(0, 1)
    
    # Try to find a fraction p/q with |q| <= max_denom such that |x - p/q| is very small
    best_frac = None
    best_err = 1e-8
    
    for q in range(1, max_denom + 1):
        p = round(x * q)
        if p == 0:
            continue
        err = abs(x - p / q)
        if err < best_err:
            best_err = err
            best_frac = Fraction(p, q)
    
    if best_frac is not None and best_err < 1e-8:
        return best_frac
    return None

def _optimize_rank_r(n, m, p, rank, rng, deadline, max_iters=5000, lr_init=0.01):
    """
    Smooth optimization for rank-r decomposition.
    Minimizes sum of squared residuals.
    """
    N = n * m
    P = m * p
    Q = p * n
    
    # Initialize with small random values
    U = rng.normal(0, 0.5, size=(N, rank)).astype(np.float64)
    V = rng.normal(0, 0.5, size=(P, rank)).astype(np.float64)
    W = rng.normal(0, 0.5, size=(Q, rank)).astype(np.float64)
    
    best_loss = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    lr = lr_init
    for it in range(max_iters):
        if time.time() >= deadline:
            break
        
        # Compute residual
        R = _compute_residual(U, V, W, n, m, p, rank)
        loss = np.sum(R ** 2)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if loss < 1e-12:
            break
        
        # Compute gradients
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r_val = R[i, j, k]
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    for r in range(rank):
                        v_val = V[v_idx, r]
                        w_val = W[w_idx, r]
                        u_val = U[u_idx, r]
                        grad_U[u_idx, r] += 2 * r_val * v_val * w_val
                        grad_V[v_idx, r] += 2 * r_val * u_val * w_val
                        grad_W[w_idx, r] += 2 * r_val * u_val * v_val
        
        # Gradient descent with adaptive learning rate
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        # Adaptive learning rate
        if loss < best_loss * 0.95:
            lr = min(lr * 1.1, 0.1)
        else:
            lr = max(lr * 0.95, 0.001)
    
    return best_U, best_V, best_W, best_loss

def _try_rational_reconstruction(U, V, W, n, m, p, rank, max_denom=50):
    """
    Try to reconstruct exact rational values from the optimized factors.
    Returns (U_exact, V_exact, W_exact) as lists of Fractions, or None.
    """
    N = n * m
    P = m * p
    Q = p * n
    
    U_exact = []
    V_exact = []
    W_exact = []
    
    for i in range(N):
        row = []
        for r in range(rank):
            frac = _rational_reconstruct(U[i, r], max_denom)
            if frac is None:
                return None
            row.append(frac)
        U_exact.append(row)
    
    for i in range(P):
        row = []
        for r in range(rank):
            frac = _rational_reconstruct(V[i, r], max_denom)
            if frac is None:
                return None
            row.append(frac)
        V_exact.append(row)
    
    for i in range(Q):
        row = []
        for r in range(rank):
            frac = _rational_reconstruct(W[i, r], max_denom)
            if frac is None:
                return None
            row.append(frac)
        W_exact.append(row)
    
    return U_exact, V_exact, W_exact

def _verify_exact(U_exact, V_exact, W_exact, n, m, p, rank):
    """
    Verify exact decomposition using Fraction arithmetic.
    """
    N = n * m
    P = m * p
    Q = p * n
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0, 1)
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                for r in range(rank):
                    total += U_exact[u_idx][r] * V_exact[v_idx][r] * W_exact[w_idx][r]
                if total != Fraction(1, 1):
                    return False
    return True

def _convert_to_half_integers(U_exact, V_exact, W_exact):
    """
    Convert Fraction lists to lists of floats that are integers or half-integers.
    Returns (U, V, W) as lists of lists of floats, or None if not all are half-integers.
    """
    def convert_matrix(matrix):
        result = []
        for row in matrix:
            new_row = []
            for frac in row:
                num = frac.numerator
                den = frac.denominator
                # Check if it's an integer or half-integer
                if den == 1:
                    new_row.append(float(num))
                elif den == 2:
                    new_row.append(float(num) / 2.0)
                else:
                    return None
            result.append(new_row)
        return result
    
    U = convert_matrix(U_exact)
    if U is None:
        return None
    V = convert_matrix(V_exact)
    if V is None:
        return None
    W = convert_matrix(W_exact)
    if W is None:
        return None
    
    return U, V, W

def _local_search_refine(U, V, W, n, m, p, rank, rng, deadline, max_iters=2000, step_size=2):
    """
    Local search refinement for integer/half-integer values.
    """
    N = n * m
    P = m * p
    Q = p * n
    
    best_err = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for _iter in range(max_iters):
        if time.time() >= deadline:
            break
            
        R = _compute_residual(U, V, W, n, m, p, rank)
        err = np.abs(R)
        max_err = np.max(err)
        
        if max_err < 1e-9:
            return (U, V, W)
        
        if max_err < best_err:
            best_err = max_err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Pick a random non-zero residual element
        nonzero_idx = np.argwhere(err > 1e-9)
        if len(nonzero_idx) == 0:
            return (U, V, W)
        
        ri, rj, rk = nonzero_idx[rng.integers(0, len(nonzero_idx))]
        target = 1.0
        current = 0.0
        for r in range(rank):
            current += U[ri * m + rj, r] * V[rj * p + rk, r] * W[rk * n + ri, r]
            
        deficit = target - current
        
        # Try to adjust one entry
        r = rng.integers(0, rank)
        
        uv = U[ri * m + rj, r]
        vv = V[rj * p + rk, r]
        ww = W[rk * n + ri, r]
        
        choice = rng.integers(0, 3)
        
        improved = False
        
        if choice == 0:
            if vv * ww != 0:
                delta = deficit / (vv * ww)
                # Try nearby half-integer values
                base = uv + delta
                candidates = [base - 0.5, base, base + 0.5]
                for cand in candidates:
                    rounded = round(cand * 2) / 2.0
                    if abs(rounded - uv) <= step_size:
                        U[ri * m + rj, r] = rounded
                        new_current = 0.0
                        for r2 in range(rank):
                            new_current += U[ri * m + rj, r2] * V[rj * p + rk, r2] * W[rk * n + ri, r2]
                        if abs(new_current - target) < abs(current - target) + 1e-12:
                            improved = True
                            break
                        else:
                            U[ri * m + rj, r] = uv
        elif choice == 1:
            if uv * ww != 0:
                delta = deficit / (uv * ww)
                base = vv + delta
                candidates = [base - 0.5, base, base + 0.5]
                for cand in candidates:
                    rounded = round(cand * 2) / 2.0
                    if abs(rounded - vv) <= step_size:
                        V[rj * p + rk, r] = rounded
                        new_current = 0.0
                        for r2 in range(rank):
                            new_current += U[ri * m + rj, r2] * V[rj * p + rk, r2] * W[rk * n + ri, r2]
                        if abs(new_current - target) < abs(current - target) + 1e-12:
                            improved = True
                            break
                        else:
                            V[rj * p + rk, r] = vv
        else:
            if uv * vv != 0:
                delta = deficit / (uv * vv)
                base = ww + delta
                candidates = [base - 0.5, base, base + 0.5]
                for cand in candidates:
                    rounded = round(cand * 2) / 2.0
                    if abs(rounded - ww) <= step_size:
                        W[rk * n + ri, r] = rounded
                        new_current = 0.0
                        for r2 in range(rank):
                            new_current += U[ri * m + rj, r2] * V[rj * p + rk, r2] * W[rk * n + ri, r2]
                        if abs(new_current - target) < abs(current - target) + 1e-12:
                            improved = True
                            break
                        else:
                            W[rk * n + ri, r] = ww
        
        # Occasionally do a larger random perturbation
        if not improved and rng.random() < 0.1:
            r_pert = rng.integers(0, rank)
            choice_pert = rng.integers(0, 3)
            if choice_pert == 0:
                idx = rng.integers(0, N)
                U[idx, r_pert] += rng.choice([-1.0, -0.5, 0.5, 1.0])
            elif choice_pert == 1:
                idx = rng.integers(0, P)
                V[idx, r_pert] += rng.choice([-1.0, -0.5, 0.5, 1.0])
            else:
                idx = rng.integers(0, Q)
                W[idx, r_pert] += rng.choice([-1.0, -0.5, 0.5, 1.0])
    
    R = _compute_residual(U, V, W, n, m, p, rank)
    if np.max(np.abs(R)) < 1e-9:
        return (U, V, W)
    
    R_best = _compute_residual(best_U, best_V, best_W, n, m, p, rank)
    if np.max(np.abs(R_best)) < 1e-9:
        return (best_U, best_V, best_W)
    return None

def _schoolbook(n, m, p):
    """Return the schoolbook rank-nmp decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    """Return a rank-minimal exact tensor decomposition."""
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 140.0  # Safety margin before 150s limit

    rng = np.random.default_rng(42)

    # Try ranks from 41 upward
    ranks_to_try = [41, 42, 43, 44]
    
    for rank in ranks_to_try:
        if time.time() >= deadline:
            break
        
        # Allocate time based on rank
        time_per_rank = 35.0 if rank == 41 else 20.0
        rank_deadline = min(deadline, time.time() + time_per_rank)
        
        # Phase 1: Smooth optimization with multiple restarts
        max_restarts = 5
        for restart in range(max_restarts):
            if time.time() >= rank_deadline:
                break
            
            # Run smooth optimization
            U_opt, V_opt, W_opt, loss = _optimize_rank_r(
                n, m, p, rank, rng, rank_deadline, 
                max_iters=2000, lr_init=0.005
            )
            
            if loss > 1.0:
                continue
            
            # Phase 2: Try rational reconstruction
            exact_result = _try_rational_reconstruction(U_opt, V_opt, W_opt, n, m, p, rank, max_denom=20)
            
            if exact_result is not None:
                U_exact, V_exact, W_exact = exact_result
                
                # Verify exact decomposition
                if _verify_exact(U_exact, V_exact, W_exact, n, m, p, rank):
                    # Convert to half-integers
                    converted = _convert_to_half_integers(U_exact, V_exact, W_exact)
                    if converted is not None:
                        U_list, V_list, W_list = converted
                        return {
                            "U": U_list,
                            "V": V_list,
                            "W": W_list,
                        }
            
            # Phase 3: Local search refinement on the optimized values
            U_init = U_opt.copy()
            V_init = V_opt.copy()
            W_init = W_opt.copy()
            
            # Round to nearest half-integers as starting point for local search
            U_init = np.round(U_init * 2) / 2.0
            V_init = np.round(V_init * 2) / 2.0
            W_init = np.round(W_init * 2) / 2.0
            
            sol = _local_search_refine(
                U_init, V_init, W_init, n, m, p, rank, rng, rank_deadline,
                max_iters=1000, step_size=2
            )
            
            if sol is not None:
                U, V, W = sol
                # Final verification
                if _verify_solution(U, V, W, n, m, p, rank):
                    return {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist(),
                    }

    # Fall back to schoolbook
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
