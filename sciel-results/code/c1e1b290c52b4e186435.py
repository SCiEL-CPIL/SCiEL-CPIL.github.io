import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 4, 7]


def construct():
    n, m, p = DIMENSIONS
    nm = n * m   # 8
    mp = m * p   # 28
    pn = p * n   # 14

    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    # ------------------------------------------------------------------
    # Precompute the multiplication tensor T of shape (nm, mp, pn)
    # where T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k.
    # All other entries are 0.
    # ------------------------------------------------------------------
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # ------------------------------------------------------------------
    # 1) Build the structural Strassen-based decomposition first (fast, exact)
    #    Expected rank: 46. This serves as our initial best and a strong
    #    initialization for the CP search.
    # ------------------------------------------------------------------
    U_str, V_str, W_str = _build_strassen_based(n, m, p)
    best_U, best_V, best_W = U_str.copy(), V_str.copy(), W_str.copy()
    best_rank = 46

    # ------------------------------------------------------------------
    # 2) Direct CP (Kruskal-Candecards) tensor rank minimization:
    #    We seek the smallest r such that
    #        T[i,j,k] = sum_{r=0}^{R-1} U[i,r] * V[j,r] * W[k,r]
    #    for all (i,j,k). This is a nonlinear least-squares problem over
    #    the entries of U, V, W. We use a smooth surrogate: minimize
    #        f(U,V,W) = 0.5 * || T - sum_r U[:,r] o V[:,r] o W[:,r] ||_F^2
    #    via scipy.optimize.minimize (L-BFGS-B) with a smooth objective.
    #
    #    We start from rank 46 (Strassen) and try to reduce rank by:
    #    a) Re-optimizing at rank 46 to see if we can find a cleaner solution
    #    b) Trying rank 45, 44, ... down to a reasonable lower bound
    #
    #    Lower bound: The matmul tensor for 2x4x7 has tensor rank at least
    #    max(nm, mp, pn) = 28? No, that's not a valid lower bound.
    #    A known lower bound for matmul(n,m,p) is at least max(nm, mp, pn)
    #    is false. The trivial lower bound is 1 (if all entries were 1,
    #    but they're not in a way that allows rank 1).
    #    Actually, for the matmul tensor, the tensor rank is known to be
    #    at least the matrix rank of any unfolding. The (nm x mp) unfolding
    #    has rank at most p=7, the (nm x pn) unfolding has rank at most m=4,
    #    the (mp x pn) unfolding has rank at most n=2. So these give weak
    #    lower bounds.
    #
    #    We'll try ranks from 46 down to 30, stopping when time runs out.
    # ------------------------------------------------------------------

    def _cp_objective(x, R, T_flat):
        """Smooth least-squares objective for CP decomposition.
        x is a flat vector of length R*(nm+mp+pn).
        Returns the sum of squared residuals.
        """
        U = x[:R * nm].reshape(nm, R)
        V = x[R * nm:R * nm + R * mp].reshape(mp, R)
        W = x[R * nm + R * mp:].reshape(pn, R)

        # Compute the CP reconstruction
        # T_rec[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
        # Efficiently: T_rec = np.einsum('ir,jr,kr->ijk', U, V, W)
        T_rec = np.einsum('ir,jr,kr->ijk', U, V, W, optimize=True)
        residual = T_rec - T_flat
        return np.sum(residual * residual)

    def _cp_jacobian(x, R, T_flat):
        """Jacobian of the CP objective (gradient of sum of squared residuals)."""
        U = x[:R * nm].reshape(nm, R)
        V = x[R * nm:R * nm + R * mp].reshape(mp, R)
        W = x[R * nm + R * mp:].reshape(pn, R)

        T_rec = np.einsum('ir,jr,kr->ijk', U, V, W, optimize=True)
        residual = T_rec - T_flat  # shape (nm, mp, pn)

        # Gradient w.r.t. U[i,r] = sum_{j,k} residual[i,j,k] * V[j,r] * W[k,r]
        # grad_U = np.einsum('ijk,jr,kr->ir', residual, V, W, optimize=True)
        grad_U = np.einsum('ijk,jr,kr->ir', residual, V, W, optimize=True)
        grad_V = np.einsum('ijk,ir,kr->jr', residual, U, W, optimize=True)
        grad_W = np.einsum('ijk,ir,jr->kr', residual, U, V, optimize=True)

        # The objective is 0.5 * ||residual||^2, so gradient is residual-based
        # df/dU = 2 * residual contracted... actually let's be careful.
        # f = 0.5 * sum_{i,j,k} (T_rec - T)^2
        # df/dU[i,r] = sum_{j,k} (T_rec[i,j,k] - T[i,j,k]) * V[j,r] * W[k,r]
        # So the gradient is just the einsum above (no factor of 2 since we have 0.5)
        grad = np.concatenate([grad_U.ravel(), grad_V.ravel(), grad_W.ravel()])
        return grad

    # Try CP optimization at various ranks
    # We'll go from 46 down, but only if we have time.
    # Each optimization attempt gets a time budget.
    
    # First, let's see if we can improve the Strassen solution at rank 46
    # by re-optimizing (this might find a cleaner representation but same rank)
    
    # Actually, the key insight: if we start CP at rank R with the Strassen
    # solution (padded with zeros for R < 46, or truncated for R > 46),
    # we might find that a lower rank suffices if the Strassen solution
    # has linearly dependent columns.
    
    # Let's try a different approach: start with random initializations
    # and use the smooth objective to find low-rank solutions.
    
    # Lower bound heuristic: the tensor rank of matmul(2,4,7) is unknown
    # but likely between 30 and 46. Let's try to find it.
    
    # Time budget per rank attempt
    time_per_rank = 8.0  # seconds
    
    for R in range(46, 25, -1):
        if time.time() > deadline - time_per_rank - 5:
            break
        
        # Try multiple random initializations for this rank
        n_restarts = 3
        found_exact = False
        
        for restart in range(n_restarts):
            if time.time() > deadline - 3:
                break
            
            # Initialize
            if R == 46:
                # Use Strassen solution as initialization
                U_init = U_str.astype(np.float64)
                V_init = V_str.astype(np.float64)
                W_init = W_str.astype(np.float64)
            elif R < 46:
                # Truncate the Strassen solution (take first R columns)
                # This is a heuristic - might not be optimal but gives a start
                U_init = U_str[:, :R].astype(np.float64)
                V_init = V_str[:, :R].astype(np.float64)
                W_init = W_str[:, :R].astype(np.float64)
            else:
                # Pad with small random values
                U_init = np.zeros((nm, R), dtype=np.float64)
                V_init = np.zeros((mp, R), dtype=np.float64)
                W_init = np.zeros((pn, R), dtype=np.float64)
                U_init[:, :46] = U_str
                V_init[:, :46] = V_str
                W_init[:, :46] = W_str
                U_init[:, 46:] = np.random.randn(nm, R - 46) * 0.1
                V_init[:, 46:] = np.random.randn(mp, R - 46) * 0.1
                W_init[:, 46:] = np.random.randn(pn, R - 46) * 0.1
            
            # For R < 46, also try random initialization
            if R < 46 and restart > 0:
                U_init = np.random.randn(nm, R) * 0.5
                V_init = np.random.randn(mp, R) * 0.5
                W_init = np.random.randn(pn, R) * 0.5
            
            x0 = np.concatenate([U_init.ravel(), V_init.ravel(), W_init.ravel()])
            
            try:
                res = minimize(
                    _cp_objective,
                    x0,
                    args=(R, T),
                    jac=_cp_jacobian,
                    method='L-BFGS-B',
                    options={
                        'maxiter': 200,
                        'ftol': 1e-12,
                        'gtol': 1e-10,
                        'maxls': 50
                    }
                )
                
                if res.success or res.fun < 1e-6:
                    # Check if the solution is exact
                    U_opt = res.x[:R * nm].reshape(nm, R)
                    V_opt = res.x[R * nm:R * nm + R * mp].reshape(mp, R)
                    W_opt = res.x[R * nm + R * mp:].reshape(pn, R)
                    
                    # Verify exactness
                    T_rec = np.einsum('ir,jr,kr->ijk', U_opt, V_opt, W_opt, optimize=True)
                    max_err = np.max(np.abs(T_rec - T))
                    
                    if max_err < 1e-8:
                        # Found an exact decomposition at rank R!
                        # Round to nearest integers/half-integers for cleanliness
                        U_round = np.round(U_opt * 2) / 2
                        V_round = np.round(V_opt * 2) / 2
                        W_round = np.round(W_opt * 2) / 2
                        
                        # Re-verify after rounding
                        T_rec_r = np.einsum('ir,jr,kr->ijk', U_round, V_round, W_round, optimize=True)
                        max_err_r = np.max(np.abs(T_rec_r - T))
                        
                        if max_err_r < 1e-8:
                            best_U, best_V, best_W = U_round, V_round, W_round
                            best_rank = R
                            found_exact = True
                            break
                        else:
                            # Rounding didn't preserve exactness, use the unrounded solution
                            # but we need integer/half-integer entries. Let's keep the
                            # original Strassen if rounding fails.
                            pass
                    elif res.fun < 1e-10:
                        # Very close to exact, try rounding
                        U_opt = res.x[:R * nm].reshape(nm, R)
                        V_opt = res.x[R * nm:R * nm + R * mp].reshape(mp, R)
                        W_opt = res.x[R * nm + R * mp:].reshape(pn, R)
                        
                        U_round = np.round(U_opt * 2) / 2
                        V_round = np.round(V_opt * 2) / 2
                        W_round = np.round(W_opt * 2) / 2
                        
                        T_rec_r = np.einsum('ir,jr,kr->ijk', U_round, V_round, W_round, optimize=True)
                        max_err_r = np.max(np.abs(T_rec_r - T))
                        
                        if max_err_r < 1e-8:
                            best_U, best_V, best_W = U_round, V_round, W_round
                            best_rank = R
                            found_exact = True
                            break
            except Exception:
                pass
        
        if found_exact:
            # We found a lower rank, continue trying to go even lower
            continue

    # ------------------------------------------------------------------
    # 3) Final verification and return
    # ------------------------------------------------------------------
    if _verify(best_U, best_V, best_W, n, m, p):
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Fallback: schoolbook rank 56
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _build_strassen_based(n, m, p):
    """Build a Strassen-based decomposition for 2x4x7 matmul.
    
    Strategy:
    - A (2x4) = [A1 | A2] where A1, A2 are 2x2
    - B (4x7) = [B1; B2] where B1, B2 are 2x7
    - C = A1*B1 + A2*B2
    - For each 2x2 * 2x7 product, split 7 cols into 2+2+2+1
    - 2x2*2x2 uses Strassen (rank 7), 2x2*2x1 uses schoolbook (rank 2)
    - Total: 2 * (3*7 + 1*2) = 46
    """
    # Strassen 2x2x2
    U_s = np.zeros((4, 7), dtype=int)
    V_s = np.zeros((4, 7), dtype=int)
    W_s = np.zeros((4, 7), dtype=int)

    # M1 = (A00+A11)(B01+B10)
    U_s[0, 0] = 1; U_s[3, 0] = 1
    V_s[1, 0] = 1; V_s[2, 0] = 1
    W_s[0, 0] = 1; W_s[3, 0] = 1
    # M2 = (A10+A11)B00
    U_s[2, 1] = 1; U_s[3, 1] = 1
    V_s[0, 1] = 1
    W_s[2, 1] = 1; W_s[3, 1] = -1
    # M3 = A00(B00-B10)
    U_s[0, 2] = 1
    V_s[0, 2] = 1; V_s[2, 2] = -1
    W_s[1, 2] = 1; W_s[3, 2] = 1
    # M4 = A11(B11-B01)
    U_s[3, 3] = 1
    V_s[3, 3] = 1; V_s[1, 3] = -1
    W_s[0, 3] = 1; W_s[2, 3] = 1
    # M5 = (A00+A01)B11
    U_s[0, 4] = 1; U_s[1, 4] = 1
    V_s[3, 4] = 1
    W_s[0, 4] = -1; W_s[1, 4] = 1
    # M6 = (A10-A00)(B00+B01)
    U_s[2, 5] = 1; U_s[0, 5] = -1
    V_s[0, 5] = 1; V_s[1, 5] = 1
    W_s[3, 5] = 1
    # M7 = (A01-A11)(B10+B11)
    U_s[1, 6] = 1; U_s[3, 6] = -1
    V_s[2, 6] = 1; V_s[3, 6] = 1
    W_s[0, 6] = 1

    # Schoolbook 2x2x1 (rank 2)
    U_1 = np.zeros((4, 2), dtype=int)
    V_1 = np.zeros((2, 2), dtype=int)
    W_1 = np.zeros((2, 2), dtype=int)
    for j in range(2):
        for i in range(2):
            U_1[i * 2 + j, j] = 1
        V_1[j, j] = 1
        for i in range(2):
            W_1[i, j] = 1

    total_rank = 46
    U = np.zeros((8, total_rank), dtype=int)
    V = np.zeros((28, total_rank), dtype=int)
    W = np.zeros((14, total_rank), dtype=int)

    col = 0
    for block in range(2):
        # A cols: 2*block, 2*block+1
        # B rows: 2*block, 2*block+1
        # B cols: 0..6, split as 2,2,2,1

        # Three 2x2x2 Strassen blocks
        for sub in range(3):
            for r in range(7):
                # U
                for i in range(2):
                    for j in range(2):
                        U[i * 4 + (2 * block + j), col] += U_s[i * 2 + j, r]
                # V
                for j in range(2):
                    for k in range(2):
                        V[(2 * block + j) * 7 + (sub * 2 + k), col] += V_s[j * 2 + k, r]
                # W
                for k in range(2):
                    for i in range(2):
                        W[(sub * 2 + k) * 2 + i, col] += W_s[k * 2 + i, r]
                col += 1

        # One 2x2x1 schoolbook block (col 6)
        for r in range(2):
            for i in range(2):
                for j in range(2):
                    U[i * 4 + (2 * block + j), col] += U_1[i * 2 + j, r]
            for j in range(2):
                V[(2 * block + j) * 7 + 6, col] += V_1[j, r]
            for i in range(2):
                W[6 * 2 + i, col] += W_1[i, r]
            col += 1

    return U, V, W


def _build_schoolbook(n, m, p):
    """Standard schoolbook decomposition, rank = n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = 1 for all i,j,k."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True
