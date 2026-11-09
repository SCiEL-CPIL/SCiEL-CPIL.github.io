# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def arithmetic_kakeya_distribution() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin before 600s kill

    n = 10
    ys = np.arange(n)
    xs = np.arange(n)
    X, Y = np.meshgrid(xs, ys)  # X[y,x] = x, Y[y,x] = y

    # Precompute the index arrays for each projection
    XY_diff = (X - Y + (n - 1)).flatten()  # 0 to 2*(n-1)
    X_flat = X.flatten()  # 0 to n-1
    XY_sum = (X + Y).flatten()  # 0 to 2*(n-1)
    X2Y = (X + 2 * Y).flatten()  # 0 to 3*(n-1)

    def compute_ratio(P):
        """Compute D/M ratio for a given distribution P (vectorized)."""
        P = np.maximum(P, 0)
        total = P.sum()
        if total <= 1e-9:
            return 0.0
        P = P / total
        P_flat = P.flatten()

        # H(X-Y)
        XY_dist = np.bincount(XY_diff, weights=P_flat, minlength=2 * n - 1)
        XY_nonzero = XY_dist[XY_dist > 1e-15]
        H_XY = -np.sum(XY_nonzero * np.log2(XY_nonzero)) if len(XY_nonzero) > 0 else 0.0

        # H(X)
        X_dist = np.bincount(X_flat, weights=P_flat, minlength=n)
        X_nonzero = X_dist[X_dist > 1e-15]
        H_X = -np.sum(X_nonzero * np.log2(X_nonzero)) if len(X_nonzero) > 0 else 0.0

        # H(X+Y)
        XY_sum_dist = np.bincount(XY_sum, weights=P_flat, minlength=2 * n - 1)
        XY_sum_nonzero = XY_sum_dist[XY_sum_dist > 1e-15]
        H_XY_sum = -np.sum(XY_sum_nonzero * np.log2(XY_sum_nonzero)) if len(XY_sum_nonzero) > 0 else 0.0

        # H(X+2Y)
        X2Y_dist = np.bincount(X2Y, weights=P_flat, minlength=3 * n - 1)
        X2Y_nonzero = X2Y_dist[X2Y_dist > 1e-15]
        H_X2Y = -np.sum(X2Y_nonzero * np.log2(X2Y_nonzero)) if len(X2Y_nonzero) > 0 else 0.0

        M = max(H_X, H_XY_sum, H_X2Y)
        if M <= 1e-9:
            return 0.0
        return H_XY / M

    def soft_max(vals, temperature=0.1):
        """Smooth approximation of max using log-sum-exp."""
        vals = np.array(vals)
        v_max = np.max(vals)
        if v_max <= 0:
            return 0.0
        exp_vals = np.exp((vals - v_max) / temperature)
        return v_max + temperature * np.log(np.sum(exp_vals))

    def objective_smooth(params, temperature=0.1):
        """Negative of the ratio with smooth max for better gradient info."""
        P = np.exp(params).reshape(n, n)
        total = P.sum()
        if total <= 1e-9:
            return 1e10
        P = P / total
        P_flat = P.flatten()

        # H(X-Y)
        XY_dist = np.bincount(XY_diff, weights=P_flat, minlength=2 * n - 1)
        XY_nonzero = XY_dist[XY_dist > 1e-15]
        H_XY = -np.sum(XY_nonzero * np.log2(XY_nonzero)) if len(XY_nonzero) > 0 else 0.0

        # H(X)
        X_dist = np.bincount(X_flat, weights=P_flat, minlength=n)
        X_nonzero = X_dist[X_dist > 1e-15]
        H_X = -np.sum(X_nonzero * np.log2(X_nonzero)) if len(X_nonzero) > 0 else 0.0

        # H(X+Y)
        XY_sum_dist = np.bincount(XY_sum, weights=P_flat, minlength=2 * n - 1)
        XY_sum_nonzero = XY_sum_dist[XY_sum_dist > 1e-15]
        H_XY_sum = -np.sum(XY_sum_nonzero * np.log2(XY_sum_nonzero)) if len(XY_sum_nonzero) > 0 else 0.0

        # H(X+2Y)
        X2Y_dist = np.bincount(X2Y, weights=P_flat, minlength=3 * n - 1)
        X2Y_nonzero = X2Y_dist[X2Y_dist > 1e-15]
        H_X2Y = -np.sum(X2Y_nonzero * np.log2(X2Y_nonzero)) if len(X2Y_nonzero) > 0 else 0.0

        M_smooth = soft_max([H_X, H_XY_sum, H_X2Y], temperature)
        if M_smooth <= 1e-9:
            return 1e10
        return -H_XY / M_smooth

    def objective(params):
        """Negative of the ratio for minimization (exact max)."""
        P = np.exp(params).reshape(n, n)
        total = P.sum()
        if total <= 1e-9:
            return 1e10
        P = P / total
        return -compute_ratio(P)

    # Start with structured initializations
    np.random.seed(42)
    
    # Initial point 1: uniform on anti-diagonals (maximizes H(X-Y))
    P_best = np.zeros((n, n), dtype=np.float64)
    for y in range(n):
        for x in range(n):
            k = abs(x - y)
            P_best[y, x] = 1.0 / (19.0 * (n - k))
    P_best /= P_best.sum()
    best_ratio = compute_ratio(P_best)

    # Try L-BFGS-B from multiple diverse starting points
    starting_points = []
    
    # 1. Anti-diagonal uniform (current best)
    starting_points.append(P_best.copy())
    
    # 2. Concentrate on cells where x-y has high entropy but x, x+y, x+2y are concentrated
    for subset_size in [3, 4, 5, 6, 7]:
        for offset in range(n):
            P_candidate = np.zeros((n, n), dtype=np.float64)
            for y in range(n):
                for x in range(n):
                    d = (x - y + n) % (2*n - 1)
                    if d in range(offset, offset + subset_size) or d in range(2*n-1 - subset_size, 2*n-1):
                        P_candidate[y, x] = 1.0
            if P_candidate.sum() > 0:
                P_candidate /= P_candidate.sum()
                starting_points.append(P_candidate.copy())
    
    # 3. Random Dirichlet samples
    for _ in range(15):
        P_rand = np.random.dirichlet(np.ones(n*n))
        P_rand = P_rand.reshape(n, n)
        starting_points.append(P_rand)
    
    # 4. Structured: emphasize certain rows/columns
    for alpha in [0.3, 0.5, 0.7, 0.9]:
        P_struct = np.zeros((n, n), dtype=np.float64)
        for y in range(n):
            for x in range(n):
                P_struct[y, x] = np.exp(-alpha * (x - y)**2 / (2*n))
        P_struct /= P_struct.sum()
        starting_points.append(P_struct)
    
    # 5. Sparse patterns
    for trial in range(10):
        P_sparse = np.zeros((n, n), dtype=np.float64)
        for _ in range(20):
            ry = np.random.randint(0, n)
            rx = np.random.randint(0, n)
            P_sparse[ry, rx] = 1.0
        P_sparse /= P_sparse.sum()
        starting_points.append(P_sparse)
    
    # 6. More structured starts from the current program
    # Weight by exp(-alpha * |x-y|)
    for alpha in [0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
        P_a = np.zeros((n, n))
        for y in range(n):
            for x in range(n):
                P_a[y, x] = np.exp(-alpha * abs(x - y))
        P_a /= P_a.sum()
        starting_points.append(P_a.copy())

    # Weight by exp(-beta * (x+y)^2) - concentrate near corners
    for beta in [0.1, 0.3, 0.5, 1.0]:
        P_b = np.zeros((n, n))
        for y in range(n):
            for x in range(n):
                P_b[y, x] = np.exp(-beta * (x + y) ** 2)
        P_b /= P_b.sum()
        starting_points.append(P_b.copy())

    # Weight by exp(-gamma * (x-y)^2) - concentrate near main diagonal
    for gamma in [0.1, 0.3, 0.5, 1.0, 2.0]:
        P_c = np.zeros((n, n))
        for y in range(n):
            for x in range(n):
                P_c[y, x] = np.exp(-gamma * (x - y) ** 2)
        P_c /= P_c.sum()
        starting_points.append(P_c.copy())

    # Weight by exp(-delta * (x+y-9)^2) - concentrate near anti-diagonal
    for delta in [0.1, 0.3, 0.5, 1.0, 2.0]:
        P_d = np.zeros((n, n))
        for y in range(n):
            for x in range(n):
                P_d[y, x] = np.exp(-delta * (x + y - (n - 1)) ** 2)
        P_d /= P_d.sum()
        starting_points.append(P_d.copy())
    
    # Concentrated line patterns: high weight on cells where x-y is in a small set
    for d_set in [[0], [0, 1], [-1, 0, 1], [0, 2], [-2, 0, 2], [0, 3, -3]]:
        P_line = np.zeros((n, n))
        for y in range(n):
            for x in range(n):
                if (x - y) in d_set:
                    P_line[y, x] = 1.0
        if P_line.sum() > 0:
            P_line /= P_line.sum()
            starting_points.append(P_line.copy())
    
    # Hybrid: concentrate on lines but with smooth decay
    for d_set, width in [([0], 1.0), ([0], 2.0), ([0, 1], 1.5), ([-1, 0, 1], 2.0)]:
        P_hybrid = np.zeros((n, n))
        for y in range(n):
            for x in range(n):
                d = x - y
                min_dist = min(abs(d - dd) for dd in d_set)
                P_hybrid[y, x] = np.exp(-min_dist**2 / (2 * width**2))
        if P_hybrid.sum() > 0:
            P_hybrid /= P_hybrid.sum()
            starting_points.append(P_hybrid.copy())
    
    # Checkerboard-like patterns that spread across projections
    for phase in [0, 1]:
        P_check = np.zeros((n, n))
        for y in range(n):
            for x in range(n):
                if (x + y + phase) % 2 == 0:
                    P_check[y, x] = 1.0
        if P_check.sum() > 0:
            P_check /= P_check.sum()
            starting_points.append(P_check.copy())
    
    # Weighted by product of projection spreads
    for alpha in [0.5, 1.0, 2.0]:
        P_prod = np.zeros((n, n))
        for y in range(n):
            for x in range(n):
                # Encourage spread in x, x+y, x+2y while concentrating x-y
                P_prod[y, x] = np.exp(-alpha * abs(x - y) / n) * (1 + 0.1 * np.sin(x * np.pi / n) * np.sin(y * np.pi / n))
        if P_prod.sum() > 0:
            P_prod /= P_prod.sum()
            starting_points.append(P_prod.copy())
    
    # Optimize from each starting point using L-BFGS-B with smooth objective first, then exact
    for i, P_init in enumerate(starting_points):
        if time.time() > deadline - 60:
            break
        
        P_init_safe = np.maximum(P_init, 1e-12)
        params_init = np.log(P_init_safe.flatten())
        
        try:
            # Phase 1: Use smooth objective for better convergence
            result_smooth = minimize(objective_smooth, params_init, method='L-BFGS-B',
                                     args=(0.1,),
                                     options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 3000})
            
            # Phase 2: Polish with exact objective
            result = minimize(objective, result_smooth.x, method='L-BFGS-B',
                             options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 3000})
            
            P_opt = np.exp(result.x).reshape(n, n)
            P_opt = np.maximum(P_opt, 0)
            total = P_opt.sum()
            if total > 1e-9:
                P_opt /= total
                ratio = compute_ratio(P_opt)
                if ratio > best_ratio:
                    best_ratio = ratio
                    P_best = P_opt.copy()
        except Exception:
            pass
        
        # Phase 3: Try SLSQP on the best result from L-BFGS-B for this starting point
        if time.time() > deadline - 60:
            break
        try:
            P_slsqp_init = P_opt.copy()
            P_slsqp_init = np.maximum(P_slsqp_init, 1e-12)
            params_slsqp_init = np.log(P_slsqp_init.flatten())
            
            def slsqp_obj(params):
                P = np.exp(params).reshape(n, n)
                P = np.maximum(P, 0)
                total_p = P.sum()
                if total_p <= 1e-9:
                    return 1e10
                P = P / total_p
                return -compute_ratio(P)
            
            res_slsqp = minimize(
                slsqp_obj,
                params_slsqp_init,
                method='SLSQP',
                bounds=[(-50, 50)] * (n*n),
                options={'maxiter': 300, 'ftol': 1e-12, 'disp': False}
            )
            
            P_slsqp = np.exp(res_slsqp.x).reshape(n, n)
            P_slsqp = np.maximum(P_slsqp, 0)
            total_slsqp = P_slsqp.sum()
            if total_slsqp > 1e-9:
                P_slsqp /= total_slsqp
                ratio_slsqp = compute_ratio(P_slsqp)
                if ratio_slsqp > best_ratio:
                    best_ratio = ratio_slsqp
                    P_best = P_slsqp.copy()
        except Exception:
            pass
    
    # Final polish: SLSQP-based refinement using epigraph formulation
    # This handles the non-smooth max more efficiently than coordinate descent
    def epigraph_objective(params, t):
        """Minimize -D/M subject to M <= t, P >= 0, sum(P) = 1.
        params: log-space weights (n*n), t: upper bound on M.
        Returns: -D/M (to minimize)"""
        P = np.exp(params).reshape(n, n)
        P = np.maximum(P, 0)
        total = P.sum()
        if total <= 1e-9:
            return 1e10
        P = P / total
        P_flat = P.flatten()
        
        # H(X-Y) = D
        XY_dist = np.bincount(XY_diff, weights=P_flat, minlength=2 * n - 1)
        XY_nonzero = XY_dist[XY_dist > 1e-15]
        D = -np.sum(XY_nonzero * np.log2(XY_nonzero)) if len(XY_nonzero) > 0 else 0.0
        
        # H(X), H(X+Y), H(X+2Y)
        X_dist = np.bincount(X_flat, weights=P_flat, minlength=n)
        X_nonzero = X_dist[X_dist > 1e-15]
        H_X = -np.sum(X_nonzero * np.log2(X_nonzero)) if len(X_nonzero) > 0 else 0.0
        
        XY_sum_dist = np.bincount(XY_sum, weights=P_flat, minlength=2 * n - 1)
        XY_sum_nonzero = XY_sum_dist[XY_sum_dist > 1e-15]
        H_XY_sum = -np.sum(XY_sum_nonzero * np.log2(XY_sum_nonzero)) if len(XY_sum_nonzero) > 0 else 0.0
        
        X2Y_dist = np.bincount(X2Y, weights=P_flat, minlength=3 * n - 1)
        X2Y_nonzero = X2Y_dist[X2Y_dist > 1e-15]
        H_X2Y = -np.sum(X2Y_nonzero * np.log2(X2Y_nonzero)) if len(X2Y_nonzero) > 0 else 0.0
        
        M_actual = max(H_X, H_XY_sum, H_X2Y)
        
        # Constraint: M_actual - t <= 0
        # Objective: -D / max(M_actual, t)
        M_eff = max(M_actual, t, 1e-9)
        return -D / M_eff
    
    # Run SLSQP refinement on the best candidate
    if time.time() < deadline - 60:
        P_init = np.maximum(P_best, 1e-12)
        params_init = np.log(P_init.flatten())
        # Initial t: slightly above the current M
        P_init_norm = P_init / P_init.sum()
        _, _, _, _, M_init = (lambda p: (
            np.bincount(XY_diff, weights=p.flatten(), minlength=2*n-1),
            np.bincount(X_flat, weights=p.flatten(), minlength=n),
            np.bincount(XY_sum, weights=p.flatten(), minlength=2*n-1),
            np.bincount(X2Y, weights=p.flatten(), minlength=3*n-1),
            max(
                -np.sum([v for v in np.bincount(X_flat, weights=p.flatten(), minlength=n) if v > 1e-15] * np.log2([v for v in np.bincount(X_flat, weights=p.flatten(), minlength=n) if v > 1e-15])),
                -np.sum([v for v in np.bincount(XY_sum, weights=p.flatten(), minlength=2*n-1) if v > 1e-15] * np.log2([v for v in np.bincount(XY_sum, weights=p.flatten(), minlength=2*n-1) if v > 1e-15])),
                -np.sum([v for v in np.bincount(X2Y, weights=p.flatten(), minlength=3*n-1) if v > 1e-15] * np.log2([v for v in np.bincount(X2Y, weights=p.flatten(), minlength=3*n-1) if v > 1e-15]))
            )
        ))(P_init_norm)
        t_init = M_init + 0.1
        
        try:
            # SLSQP with epigraph constraint
            res_slsqp = minimize(
                lambda x: epigraph_objective(x[:-1], x[-1]),
                np.concatenate([params_init, [t_init]]),
                method='SLSQP',
                bounds=[(-50, 50)] * (n*n) + [(1e-6, 5.0)],
                constraints=[
                    {'type': 'ineq', 'fun': lambda x: x[-1] - 1e-6},  # t > 0
                    {'type': 'eq', 'fun': lambda x: np.exp(x[:-1]).sum() - 1.0},  # sum(exp(params)) = 1
                ],
                options={'maxiter': 500, 'ftol': 1e-12, 'disp': False}
            )
            
            if res_slsqp.success or res_slsqp.fun < -best_ratio + 1e-6:
                P_slsqp = np.exp(res_slsqp.x[:-1]).reshape(n, n)
                P_slsqp = np.maximum(P_slsqp, 0)
                total_slsqp = P_slsqp.sum()
                if total_slsqp > 1e-9:
                    P_slsqp /= total_slsqp
                    ratio_slsqp = compute_ratio(P_slsqp)
                    if ratio_slsqp > best_ratio:
                        best_ratio = ratio_slsqp
                        P_best = P_slsqp.copy()
        except Exception:
            pass
    
    # Additional high-precision polish with Nelder-Mead on the best candidate
    if time.time() < deadline - 30:
        try:
            def nm_objective(params):
                P = np.exp(params).reshape(n, n)
                P = np.maximum(P, 0)
                total = P.sum()
                if total <= 1e-9:
                    return 1e10
                P = P / total
                return -compute_ratio(P)
            
            P_init_nm = np.maximum(P_best, 1e-12)
            params_init_nm = np.log(P_init_nm.flatten())
            
            res_nm = minimize(
                nm_objective,
                params_init_nm,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            
            P_nm = np.exp(res_nm.x).reshape(n, n)
            P_nm = np.maximum(P_nm, 0)
            total_nm = P_nm.sum()
            if total_nm > 1e-9:
                P_nm /= total_nm
                ratio_nm = compute_ratio(P_nm)
                if ratio_nm > best_ratio:
                    best_ratio = ratio_nm
                    P_best = P_nm.copy()
        except Exception:
            pass
    
    # Final normalization and clipping
    P_best = np.maximum(P_best, 0)
    total = P_best.sum()
    if total <= 1e-9:
        P_best = np.ones((n, n)) / (n * n)
    else:
        P_best /= total

    return P_best
# EVOLVE-BLOCK-END
