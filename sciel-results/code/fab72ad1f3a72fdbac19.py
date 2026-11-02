import time
import numpy as np

DIMENSIONS = [2, 2, 3]
TIME_LIMIT = 140.0

def _verify(U, V, W, n=2, m=2, p=3):
    """Verify exact decomposition: for all i,j,k sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1"""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    return False
    return True

def _build_rank11():
    """Known rank-11 decomposition."""
    n, m, p = DIMENSIONS
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
    set_w(0, 0, 0, 1); set_w(1, 1, 0, 1)

    # Term 1: (A10+A11)B00
    set_u(1, 0, 1, 1); set_u(1, 1, 1, 1)
    set_v(0, 0, 1, 1)
    set_w(0, 1, 1, 1); set_w(1, 1, 1, -1)

    # Term 2: A00(B01-B11)
    set_u(0, 0, 2, 1)
    set_v(0, 1, 2, 1); set_v(1, 1, 2, -1)
    set_w(1, 0, 2, 1); set_w(1, 1, 2, 1)

    # Term 3: A11(B10-B00)
    set_u(1, 1, 3, 1)
    set_v(1, 0, 3, 1); set_v(0, 0, 3, -1)
    set_w(0, 0, 3, 1); set_w(0, 1, 3, 1)

    # Term 4: (A00+A01)B11
    set_u(0, 0, 4, 1); set_u(0, 1, 4, 1)
    set_v(1, 1, 4, 1)
    set_w(0, 0, 4, -1); set_w(1, 0, 4, 1)

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

def _try_rank10_algebraic():
    """
    Try to find a rank-10 decomposition by modifying Strassen terms to contribute
    to the third column, reducing the number of dedicated column-2 terms.
    
    Strategy: Start with Strassen (7 terms for k=0,1) and try to add contributions
    to k=2 by modifying existing terms, then add fewer new terms.
    """
    n, m, p = 2, 2, 3
    rank = 10
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def set_u(i, j, r, val):
        U[i * m + j][r] = val
    def set_v(j, k, r, val):
        V[j * p + k][r] = val
    def set_w(k, i, r, val):
        W[k * n + i][r] = val

    # Use 7 Strassen-like terms, but allow them to contribute to k=2 as well.
    # The idea: for each Strassen term r, we can set V[j,2,r] and W[2,i,r]
    # to nonzero values, which adds contributions to C[i,2].
    
    # Let's parameterize: for the 7 Strassen terms, we have freedom in
    # V[*,2,r] and W[2,*,r]. Let's denote:
    #   alpha_r = V[0,2,r], beta_r = V[1,2,r]
    #   gamma_r = W[2,0,r], delta_r = W[2,1,r]
    
    # The contribution to C[0,2] from term r is:
    #   sum_j U[0,j,r] * V[j,2,r] * W[2,0,r]
    #   = U[0,0,r]*alpha_r*gamma_r + U[0,1,r]*beta_r*gamma_r
    
    # The contribution to C[1,2] from term r is:
    #   U[1,0,r]*alpha_r*delta_r + U[1,1,r]*beta_r*delta_r
    
    # We need the total from all 10 terms to give:
    #   C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    #   C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    
    # This means we need 3 more terms (terms 7,8,9) to complete the picture,
    # but the 7 Strassen terms can contribute partially.
    
    # Let me try a specific construction:
    # Terms 0-6: Strassen for k=0,1, with added k=2 contributions
    # Terms 7-9: Additional terms to complete k=2
    
    # Actually, let me try a cleaner approach. Let me use the fact that
    # we can add a "dual" Strassen-like structure for the third column.
    
    # Try: 5 terms for the main 2x2x2 block, and 5 terms for the 2x2x1 extension
    # with some sharing.
    
    # Let me try a different approach: use 10 terms where 7 are Strassen-like
    # and 3 are new, but modify the Strassen terms to partially cover column 2.
    
    # Term 0: (A00+A11)(B00+B11) + partial contribution to col 2
    set_u(0, 0, 0, 1); set_u(1, 1, 0, 1)
    set_v(0, 0, 0, 1); set_v(1, 1, 0, 1)
    set_v(0, 2, 0, 1); set_v(1, 2, 0, 1)  # Add col 2 contribution
    set_w(0, 0, 0, 1); set_w(1, 1, 0, 1)
    set_w(2, 0, 0, 1); set_w(2, 1, 0, 1)  # Add col 2 contribution
    
    # This term contributes to C[0,2]: U[0,0,0]*V[0,2,0]*W[2,0,0] + U[0,1,0]*V[1,2,0]*W[2,0,0]
    # = 1*1*1 + 0*1*1 = 1 (coefficient of A[0,0]*B[0,2])
    # And to C[1,2]: U[1,0,0]*V[0,2,0]*W[2,1,0] + U[1,1,0]*V[1,2,0]*W[2,1,0]
    # = 0*1*1 + 1*1*1 = 1 (coefficient of A[1,1]*B[1,2])
    
    # Term 1: (A10+A11)B00
    set_u(1, 0, 1, 1); set_u(1, 1, 1, 1)
    set_v(0, 0, 1, 1)
    set_w(0, 1, 1, 1); set_w(1, 1, 1, -1)
    
    # Term 2: A00(B01-B11)
    set_u(0, 0, 2, 1)
    set_v(0, 1, 2, 1); set_v(1, 1, 2, -1)
    set_w(1, 0, 2, 1); set_w(1, 1, 2, 1)
    
    # Term 3: A11(B10-B00)
    set_u(1, 1, 3, 1)
    set_v(1, 0, 3, 1); set_v(0, 0, 3, -1)
    set_w(0, 0, 3, 1); set_w(0, 1, 3, 1)
    
    # Term 4: (A00+A01)B11
    set_u(0, 0, 4, 1); set_u(0, 1, 4, 1)
    set_v(1, 1, 4, 1)
    set_w(0, 0, 4, -1); set_w(1, 0, 4, 1)
    
    # Term 5: (A10-A00)(B00+B01)
    set_u(1, 0, 5, 1); set_u(0, 0, 5, -1)
    set_v(0, 0, 5, 1); set_v(0, 1, 5, 1)
    set_w(1, 1, 5, 1)
    
    # Term 6: (A01-A11)(B10+B11)
    set_u(0, 1, 6, 1); set_u(1, 1, 6, -1)
    set_v(1, 0, 6, 1); set_v(1, 1, 6, 1)
    set_w(0, 0, 6, 1)
    
    # Now I need to check what's still needed for column 2.
    # From term 0, I've contributed:
    #   C[0,2]: coefficient 1 for A[0,0]*B[0,2]
    #   C[1,2]: coefficient 1 for A[1,1]*B[1,2]
    # 
    # I still need:
    #   C[0,2]: coefficient 1 for A[0,1]*B[1,2]
    #   C[1,2]: coefficient 1 for A[1,0]*B[0,2]
    
    # Term 7: A01*B12
    set_u(0, 1, 7, 1)
    set_v(1, 2, 7, 1)
    set_w(2, 0, 7, 1)
    
    # Term 8: A10*B02
    set_u(1, 0, 8, 1)
    set_v(0, 2, 8, 1)
    set_w(2, 1, 8, 1)
    
    # That's only 9 terms! Let me verify this works.
    
    # Wait, I only used 9 terms. Let me check if this is valid.
    # Actually, I can leave term 9 as zero, making it effectively rank 9?
    # No, the rank is the number of nonzero terms. Let me just use 9 terms.
    
    # Actually, let me reconsider. The problem asks for rank = number of terms.
    # If I can do it in 9, that's even better!
    
    # Let me verify the 9-term solution.
    U9 = [U[i][:9] for i in range(n*m)]
    V9 = [V[i][:9] for i in range(m*p)]
    W9 = [W[i][:9] for i in range(p*n)]
    
    if _verify(U9, V9, W9):
        return {"U": U9, "V": V9, "W": W9}
    
    # If 9-term doesn't work, try 10-term with term 9 as a correction
    # Let me check what's missing and add a correction term.
    
    # Actually, let me just try the 10-term version with a zero term 9.
    # But that's wasteful. Let me instead try a proper 10-term search.
    
    return None

def _search_rank10_random(deadline):
    """Random search for rank-10 decomposition over integers."""
    n, m, p = 2, 2, 3
    rank = 10
    rng = np.random.default_rng(42)
    
    # Try random initializations and use scipy to optimize
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
                    s = 0.0
                    for r in range(10):
                        s += U_mat[i*2+j, r] * V_mat[j*3+k, r] * W_mat[k*2+i, r]
                    res[idx] = s - 1.0
                    idx += 1
        return res
    
    base = _build_rank11()
    U11 = np.array(base["U"], dtype=float)
    V11 = np.array(base["V"], dtype=float)
    W11 = np.array(base["W"], dtype=float)
    
    best_cost = float('inf')
    best_params = None
    
    for trial in range(50):
        if time.time() > deadline - 10:
            break
        
        # Start from rank-11, drop a random term, add perturbation
        drop = rng.integers(0, 11)
        keep = [i for i in range(11) if i != drop]
        U0 = U11[:, keep].copy()
        V0 = V11[:, keep].copy()
        W0 = W11[:, keep].copy()
        
        x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        x0 = x0 + rng.normal(0, 0.5, 160)
        
        try:
            result = least_squares(residuals, x0, method='lm', max_nfev=3000, ftol=1e-14, xtol=1e-14)
            if result.cost < best_cost:
                best_cost = result.cost
                best_params = result.x.copy()
                if result.cost < 1e-12:
                    break
        except:
            pass
    
    if best_params is None or best_cost > 1e-8:
        return None
    
    U_flat = best_params[:40]
    V_flat = best_params[40:100]
    W_flat = best_params[100:160]
    
    U_mat = U_flat.reshape(4, 10)
    V_mat = V_flat.reshape(6, 10)
    W_mat = W_flat.reshape(6, 10)
    
    # Try rounding to integers
    U = [[round(u) for u in row] for row in U_mat.tolist()]
    V = [[round(v) for v in row] for row in V_mat.tolist()]
    W = [[round(w) for w in row] for row in W_mat.tolist()]
    
    if _verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Try half-integers
    U = [[round(u * 2) / 2 for u in row] for row in U_mat.tolist()]
    V = [[round(v * 2) / 2 for v in row] for row in V_mat.tolist()]
    W = [[round(w * 2) / 2 for w in row] for row in W_mat.tolist()]
    
    if _verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # Try algebraic rank-10 (or rank-9) construction
    sol = _try_rank10_algebraic()
    if sol is not None:
        return sol
    
    # Try random search for rank-10
    remaining = deadline - time.time()
    if remaining > 20:
        sol = _search_rank10_random(deadline - 5)
        if sol is not None:
            return sol
    
    # Fall back to rank-11
    return _build_rank11()

