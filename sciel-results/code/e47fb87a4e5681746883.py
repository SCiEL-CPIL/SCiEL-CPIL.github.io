"""Tensor decomposition search for 2x2 by 2x3 matrix multiplication."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 3]
n, m, p = DIMENSIONS


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _rank11_decomposition():
    """
    Rank-11 decomposition: Strassen (rank 7) for 2x2 block + schoolbook (rank 4) for col 2.
    """
    n, m, p = 2, 2, 3
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def set_u(i, j, r, val):
        U[i * m + j][r] = val

    def set_v(j, k, r, val):
        V[j * p + k][r] = val

    def set_w(k, i, r, val):
        W[k * n + i][r] = val

    # Term 0: (A00+A11)(B00+B11)
    set_u(0, 0, 0, 1); set_u(1, 1, 0, 1)
    set_v(0, 0, 0, 1); set_v(1, 1, 0, 1)
    set_w(0, 0, 0, 1)
    set_w(1, 1, 0, 1)

    # Term 1: (A10+A11)B00
    set_u(1, 0, 1, 1); set_u(1, 1, 1, 1)
    set_v(0, 0, 1, 1)
    set_w(0, 1, 1, 1)
    set_w(1, 1, 1, -1)

    # Term 2: A00(B01-B11)
    set_u(0, 0, 2, 1)
    set_v(0, 1, 2, 1); set_v(1, 1, 2, -1)
    set_w(1, 0, 2, 1)
    set_w(1, 1, 2, 1)

    # Term 3: A11(B10-B00)
    set_u(1, 1, 3, 1)
    set_v(1, 0, 3, 1); set_v(0, 0, 3, -1)
    set_w(0, 0, 3, 1)
    set_w(0, 1, 3, 1)

    # Term 4: (A00+A01)B11
    set_u(0, 0, 4, 1); set_u(0, 1, 4, 1)
    set_v(1, 1, 4, 1)
    set_w(0, 0, 4, -1)
    set_w(1, 0, 4, 1)

    # Term 5: (A10-A00)(B00+B01)
    set_u(1, 0, 5, 1); set_u(0, 0, 5, -1)
    set_v(0, 0, 5, 1); set_v(0, 1, 5, 1)
    set_w(1, 1, 5, 1)

    # Term 6: (A01-A11)(B10+B11)
    set_u(0, 1, 6, 1); set_u(1, 1, 6, -1)
    set_v(1, 0, 6, 1); set_v(1, 1, 6, 1)
    set_w(0, 0, 6, 1)

    # Term 7: A00*B02
    set_u(0, 0, 7, 1)
    set_v(0, 2, 7, 1)
    set_w(2, 0, 7, 1)

    # Term 8: A01*B12
    set_u(0, 1, 8, 1)
    set_v(1, 2, 8, 1)
    set_w(2, 0, 8, 1)

    # Term 9: A10*B02
    set_u(1, 0, 9, 1)
    set_v(0, 2, 9, 1)
    set_w(2, 1, 9, 1)

    # Term 10: A11*B12
    set_u(1, 1, 10, 1)
    set_v(1, 2, 10, 1)
    set_w(2, 1, 10, 1)

    return {"U": U, "V": V, "W": W}


def _schoolbook():
    """Schoolbook decomposition of rank n*m*p = 12."""
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


def _try_rank10_structured(deadline):
    """
    Structured rank-10 search: Fix the Strassen part for columns 0,1 (7 terms),
    and search for a 3-term decomposition of column 2 that combines with Strassen.
    
    Key insight: A rank-10 solution would need to merge one Strassen term with 
    column-2 terms. We parameterize the last 3 terms and solve.
    """
    if time.time() > deadline - 25:
        return None
    
    from scipy.optimize import least_squares
    
    rank = 10
    # We'll try: keep 7 Strassen terms fixed, optimize 3 remaining terms
    # for column 2 contributions
    
    def residuals(params):
        """params: 4*3 + 6*3 + 6*3 = 48 values for the 3 free terms"""
        U_free = params[:12].reshape(4, 3)
        V_free = params[12:30].reshape(6, 3)
        W_free = params[30:48].reshape(6, 3)
        
        res = np.empty(12)
        idx = 0
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    s = 0.0
                    # Contributions from fixed Strassen terms (0-6)
                    if k < 2:  # columns 0, 1 handled by Strassen
                        # These should be satisfied by Strassen alone
                        pass
                    # Add free term contributions
                    for r in range(3):
                        s += U_free[i*2+j, r] * V_free[j*3+k, r] * W_free[k*2+i, r]
                    res[idx] = s - (1.0 if k >= 2 else 0.0)
                    idx += 1
        return res
    
    # Actually, let's try a different approach: full rank-10 with warm start
    # from rank-11 by dropping each term and optimizing
    
    base = _rank11_decomposition()
    U11 = np.array(base["U"], dtype=float)
    V11 = np.array(base["V"], dtype=float)
    W11 = np.array(base["W"], dtype=float)
    
    rng = np.random.default_rng(12345)
    
    best_result = None
    best_cost = float('inf')
    
    def full_residuals(params):
        U_flat = params[:40]
        V_flat = params[40:100]
        W_flat = params[100:160]
        U_mat = U_flat.reshape(4, 10)
        V_mat = V_flat.reshape(6, 10)
        W_mat = W_flat.reshape(6, 10)
        
        res = np.empty(12)
        idx = 0
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    s = np.sum(U_mat[i*2+j, :] * V_mat[j*3+k, :] * W_mat[k*2+i, :])
                    res[idx] = s - 1.0
                    idx += 1
        return res
    
    # Try dropping each of the 11 terms
    for drop_term in range(11):
        if time.time() > deadline - 15:
            break
        
        keep = [r for r in range(11) if r != drop_term]
        U0 = U11[:, keep].copy()
        V0 = V11[:, keep].copy()
        W0 = W11[:, keep].copy()
        
        x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        # Multiple perturbations
        for pert_scale in [0.01, 0.1, 0.5]:
            if time.time() > deadline - 15:
                break
            x0_pert = x0 + rng.normal(0, pert_scale, 160)
            try:
                result = least_squares(full_residuals, x0_pert, method='lm', 
                                       max_nfev=3000, ftol=1e-14, xtol=1e-14)
                if result.cost < best_cost:
                    best_cost = result.cost
                    best_result = result.x.copy()
                    if result.cost < 1e-12:
                        return _extract_and_round(best_result)
            except Exception:
                pass
    
    # Random initializations
    for trial in range(10):
        if time.time() > deadline - 15:
            break
        x0 = rng.normal(0, 1.5, 160)
        try:
            result = least_squares(full_residuals, x0, method='lm',
                                   max_nfev=3000, ftol=1e-14, xtol=1e-14)
            if result.cost < best_cost:
                best_cost = result.cost
                best_result = result.x.copy()
                if result.cost < 1e-12:
                    return _extract_and_round(best_result)
        except Exception:
            pass
    
    if best_result is not None and best_cost < 1e-10:
        return _extract_and_round(best_result)
    
    return None


def _extract_and_round(params):
    """Extract U, V, W from params and try rounding to half-integers or integers."""
    U_flat = params[:40]
    V_flat = params[40:100]
    W_flat = params[100:160]
    
    U_mat = U_flat.reshape(4, 10)
    V_mat = V_flat.reshape(6, 10)
    W_mat = W_flat.reshape(6, 10)
    
    # Try half-integer rounding
    U = [[round(u * 2) / 2 for u in row] for row in U_mat.tolist()]
    V = [[round(v * 2) / 2 for v in row] for row in V_mat.tolist()]
    W = [[round(w * 2) / 2 for w in row] for row in W_mat.tolist()]
    
    if _verify(U, V, W, 2, 2, 3):
        return {"U": U, "V": V, "W": W}
    
    # Try integer rounding
    U = [[round(u) for u in row] for row in U_mat.tolist()]
    V = [[round(v) for v in row] for row in V_mat.tolist()]
    W = [[round(w) for w in row] for row in W_mat.tolist()]
    
    if _verify(U, V, W, 2, 2, 3):
        return {"U": U, "V": V, "W": W}
    
    return None


def _try_rank10_lattice(deadline):
    """
    Direct lattice search: Try small integer/half-integer values for a 
    structured rank-10 decomposition.
    
    Hypothesis: A rank-10 solution might have a form where column 2 is 
    handled by 3 terms that partially overlap with Strassen terms.
    """
    if time.time() > deadline - 20:
        return None
    
    # Search space: values in {-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2}
    vals = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    
    # We'll focus on the column-2 part. For rank 10, we need 3 terms for col 2
    # that might share structure with Strassen terms.
    
    # Strategy: Assume terms 7,8,9 handle column 2 with possible overlap
    # Term 7: U[*,7], V[*,7], W[*,7] - contributes to C[:,2]
    # Term 8: U[*,8], V[*,8], W[*,8] - contributes to C[:,2]  
    # Term 9: U[*,9], V[*,9], W[*,9] - contributes to C[:,2]
    
    # For column 2, we need:
    # C[0,2] = A00*B02 + A01*B12 = sum_r U[0,r]*V[0,2,r]*W[4,r] + U[1,r]*V[1,2,r]*W[4,r]
    # C[1,2] = A10*B02 + A11*B12 = sum_r U[2,r]*V[0,2,r]*W[5,r] + U[3,r]*V[1,2,r]*W[5,r]
    
    # If we use only terms 7,8,9 for column 2 (no overlap with Strassen):
    # We need 4 equations with 3*3*3=27 variables (but sparse)
    # This is the schoolbook approach with 4 terms, so we need overlap.
    
    # Let's try: merge one Strassen term with column 2
    # E.g., modify term 6 to also contribute to column 2
    
    # This is getting complex. Let's try a targeted search.
    
    # Fixed Strassen terms 0-5 (6 terms), then 4 terms for remaining
    # But that's still rank 10.
    
    # Actually, let's try: 6 Strassen-like terms + 4 col-2 terms = 10
    # where one Strassen term is modified to help with col 2
    
    rng = np.random.default_rng(777)
    
    best_cost = float('inf')
    best_params = None
    
    def cost_func(params):
        """Compute cost for a rank-10 decomposition."""
        U_flat = params[:40]
        V_flat = params[40:100]
        W_flat = params[100:160]
        U_mat = U_flat.reshape(4, 10)
        V_mat = V_flat.reshape(6, 10)
        W_mat = W_flat.reshape(6, 10)
        
        cost = 0.0
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    s = np.sum(U_mat[i*2+j, :] * V_mat[j*3+k, :] * W_mat[k*2+i, :])
                    cost += (s - 1.0) ** 2
        return cost
    
    # Try structured initializations based on Strassen + modified col-2
    base = _rank11_decomposition()
    U11 = np.array(base["U"], dtype=float)
    V11 = np.array(base["V"], dtype=float)
    W11 = np.array(base["W"], dtype=float)
    
    # Try dropping each col-2 term and redistributing
    for drop in [7, 8, 9, 10]:
        if time.time() > deadline - 15:
            break
        keep = [r for r in range(11) if r != drop]
        U0 = U11[:, keep].copy()
        V0 = V11[:, keep].copy()
        W0 = W11[:, keep].copy()
        
        x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        from scipy.optimize import least_squares
        
        def residuals(params):
            U_flat = params[:40]
            V_flat = params[40:100]
            W_flat = params[100:160]
            U_mat = U_flat.reshape(4, 10)
            V_mat = V_flat.reshape(6, 10)
            W_mat = W_flat.reshape(6, 10)
            
            res = np.empty(12)
            idx = 0
            for i in range(2):
                for j in range(2):
                    for k in range(3):
                        s = np.sum(U_mat[i*2+j, :] * V_mat[j*3+k, :] * W_mat[k*2+i, :])
                        res[idx] = s - 1.0
                        idx += 1
            return res
        
        for scale in [0.05, 0.2, 0.5]:
            if time.time() > deadline - 15:
                break
            x0_p = x0 + rng.normal(0, scale, 160)
            try:
                result = least_squares(residuals, x0_p, method='lm',
                                       max_nfev=2000, ftol=1e-13, xtol=1e-13)
                if result.cost < best_cost:
                    best_cost = result.cost
                    best_params = result.x.copy()
                    if result.cost < 1e-12:
                        return _extract_and_round(best_params)
            except Exception:
                pass
    
    if best_params is not None and best_cost < 1e-10:
        return _extract_and_round(best_params)
    
    return None


def construct():
    deadline = time.time() + 135  # Safety margin before 150s kill
    
    # First, get the known rank-11 construction
    result11 = _rank11_decomposition()
    if _verify(result11["U"], result11["V"], result11["W"], n, m, p):
        best = result11
        best_rank = 11
    else:
        best = _schoolbook()
        best_rank = 12
    
    # Try structured rank-10 search
    if best_rank > 10 and time.time() < deadline - 40:
        result10 = _try_rank10_structured(deadline)
        if result10 is not None:
            best = result10
            best_rank = 10
    
    # Try lattice-based rank-10 search if still at rank 11
    if best_rank > 10 and time.time() < deadline - 20:
        result10b = _try_rank10_lattice(deadline)
        if result10b is not None:
            best = result10b
            best_rank = 10
    
    return best
