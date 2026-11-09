<<<<<<< SEARCH
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution


def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    t0 = time.time()
    deadline = t0 + 600.0 - 30.0  # 30s safety margin

    # Precompute projection value arrays (shape 10x10, indexed [y, x])
    xs = np.arange(n)
    ys = np.arange(n)
    X, Y = np.meshgrid(xs, ys, indexing='xy')  # X[y,x]=x, Y[y,x]=y
    # Actually with indexing='xy': X[i,j] = xs[j], Y[i,j] = ys[i]
    # So X[y,x] = x, Y[y,x] = y. Good.

    d_vals = X - Y          # x - y, range [-9, 9]
    m0_vals = X             # x, range [0, 9]
    m1_vals = X + Y         # x + y, range [0, 18]
    m2_vals = X + 2 * Y     # x + 2y, range [0, 27]

    # Unique values and inverse indices for pooling
    d_uniq, d_inv = np.unique(d_vals, return_inverse=True)
    m0_uniq, m0_inv = np.unique(m0_vals, return_inverse=True)
    m1_uniq, m1_inv = np.unique(m1_vals, return_inverse=True)
    m2_uniq, m2_inv = np.unique(m2_vals, return_inverse=True)

    nd = len(d_uniq)
    nm0 = len(m0_uniq)
    nm1 = len(m1_uniq)
    nm2 = len(m2_uniq)

    def compute_ratio_fast(P_flat):
        """Compute D/M ratio from a flat array of 100 weights (P[y,x] order)."""
        P = P_flat.reshape(n, n)
        total = P.sum()
        if total <= 1e-15:
            return -1e18
        probs = P / total

        # Pool probabilities by projection value
        d_probs = np.zeros(nd)
        np.add.at(d_probs, d_inv.ravel(), probs.ravel())
        m0_probs = np.zeros(nm0)
        np.add.at(m0_probs, m0_inv.ravel(), probs.ravel())
        m1_probs = np.zeros(nm1)
        np.add.at(m1_probs, m1_inv.ravel(), probs.ravel())
        m2_probs = np.zeros(nm2)
        np.add.at(m2_probs, m2_inv.ravel(), probs.ravel())

        def entropy(p):
            p = p[p > 1e-15]
            if len(p) == 0:
                return 0.0
            return -np.sum(p * np.log2(p))

        D = entropy(d_probs)
        M0 = entropy(m0_probs)
        M1 = entropy(m1_probs)
        M2 = entropy(m2_probs)
        M = max(M0, M1, M2)

        if M <= 1e-9:
            return -1e18
        return D / M

    # Parametric family: P[y,x] ∝ exp(a*(x-y)^2 + b*x + c*y + e*(x-y) + f*x*y)
    # This gives a rich 5-parameter family
    def params_to_P(params):
        a, b, c, e, f = params
        # Compute log-weights
        log_w = a * (X - Y)**2 + b * X + c * Y + e * (X - Y) + f * X * Y
        # Shift for numerical stability
        log_w = log_w - log_w.max()
        w = np.exp(log_w)
        s = w.sum()
        if s <= 1e-15:
            return None
        return w / s

    def negative_ratio(params):
        P = params_to_P(params)
        if P is None:
            return 1e18
        return -compute_ratio_fast(P.ravel())

    # Initial best: uniform
    best_P = np.ones((n, n)) / (n * n)
    best_ratio = compute_ratio_fast(best_P.ravel())

    # Multi-start local optimization
    np.random.seed(42)
    n_starts = 200

    for i in range(n_starts):
        if time.time() > deadline:
            break
        # Random initial parameters
        if i == 0:
            params0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            params0 = np.array([
                np.random.uniform(-1, 1),
                np.random.uniform(-1, 1),
                np.random.uniform(-1, 1),
                np.random.uniform(-2, 2),
                np.random.uniform(-0.5, 0.5)
            ])

        try:
            result = minimize(
                negative_ratio,
                params0,
                method='Nelder-Mead',
                options={
                    'maxiter': 500,
                    'xatol': 1e-6,
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
            P_opt = params_to_P(result.x)
            if P_opt is not None:
                ratio = compute_ratio_fast(P_opt.ravel())
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_P = P_opt.copy()
        except Exception:
            pass

    # Try differential evolution for global search if time permits
    if time.time() < deadline - 60:
        bounds = [(-2, 2), (-2, 2), (-2, 2), (-3, 3), (-1, 1)]
        try:
            de_result = differential_evolution(
                negative_ratio,
                bounds=bounds,
                maxiter=50,
                popsize=15,
                tol=1e-6,
                seed=42,
                polish=True
            )
            P_de = params_to_P(de_result.x)
            if P_de is not None:
                ratio_de = compute_ratio_fast(P_de.ravel())
                if ratio_de > best_ratio:
                    best_ratio = ratio_de
                    best_P = P_de.copy()
        except Exception:
            pass

    # Local refinement with finer Nelder-Mead from best
    if time.time() < deadline - 30:
        # Find params that gave best_P - we need to store them
        # Actually let's just do random perturbations on P directly
        for _ in range(2000):
            if time.time() > deadline:
                break
            perturbed = best_P.copy()
            # Random multiplicative perturbation on a few cells
            n_perturb = np.random.randint(2, 6)
            for _ in range(n_perturb):
                yi = np.random.randint(0, n)
                xi = np.random.randint(0, n)
                factor = np.random.uniform(0.5, 1.5)
                perturbed[yi, xi] *= factor
            ratio = compute_ratio_fast(perturbed.ravel())
            if ratio > best_ratio:
                best_ratio = ratio
                best_P = perturbed.copy()

    # Final: try a structured search - concentrate on anti-diagonals
    # P[y,x] = w[y] * v[x] (rank-1) won't help much
    # Try: P[y,x] ∝ exp(alpha * |x - y - c|) for various c, alpha
    if time.time() < deadline - 10:
        for c in np.arange(-9, 10, 0.5):
            if time.time() > deadline:
                break
            for alpha in np.arange(0.1, 3.0, 0.1):
                if time.time() > deadline:
                    break
                w = np.exp(-alpha * np.abs(X - Y - c))
                s = w.sum()
                if s <= 1e-15:
                    continue
                P_try = w / s
                ratio = compute_ratio_fast(P_try.ravel())
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_P = P_try.copy()

    # Normalize and clip
    P = best_P.copy()
    P = np.clip(P, 0, None)
    total = P.sum()
    if total <= 1e-15:
        P = np.ones((n, n)) / (n * n)
    else:
        P = P / total

    return P


# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    t0 = time.time()
    deadline = t0 + 600.0 - 30.0  # 30s safety margin

    # Precompute projection value arrays (shape 10x10, indexed [y, x])
    xs = np.arange(n)
    ys = np.arange(n)
    X, Y = np.meshgrid(xs, ys, indexing='xy')  # X[y,x]=x, Y[y,x]=y

    d_vals = X - Y          # x - y, range [-9, 9]
    m0_vals = X             # x, range [0, 9]
    m1_vals = X + Y         # x + y, range [0, 18]
    m2_vals = X + 2 * Y     # x + 2y, range [0, 27]

    # Unique values and inverse indices for pooling
    d_uniq, d_inv = np.unique(d_vals, return_inverse=True)
    m0_uniq, m0_inv = np.unique(m0_vals, return_inverse=True)
    m1_uniq, m1_inv = np.unique(m1_vals, return_inverse=True)
    m2_uniq, m2_inv = np.unique(m2_vals, return_inverse=True)

    nd = len(d_uniq)
    nm0 = len(m0_uniq)
    nm1 = len(m1_uniq)
    nm2 = len(m2_uniq)

    def compute_ratio_fast(P_flat):
        """Compute D/M ratio from a flat array of 100 weights (P[y,x] order)."""
        P = P_flat.reshape(n, n)
        total = P.sum()
        if total <= 1e-15:
            return -1e18
        probs = P / total

        # Pool probabilities by projection value
        d_probs = np.zeros(nd)
        np.add.at(d_probs, d_inv.ravel(), probs.ravel())
        m0_probs = np.zeros(nm0)
        np.add.at(m0_probs, m0_inv.ravel(), probs.ravel())
        m1_probs = np.zeros(nm1)
        np.add.at(m1_probs, m1_inv.ravel(), probs.ravel())
        m2_probs = np.zeros(nm2)
        np.add.at(m2_probs, m2_inv.ravel(), probs.ravel())

        def entropy(p):
            p = p[p > 1e-15]
            if len(p) == 0:
                return 0.0
            return -np.sum(p * np.log2(p))

        D = entropy(d_probs)
        M0 = entropy(m0_probs)
        M1 = entropy(m1_probs)
        M2 = entropy(m2_probs)
        M = max(M0, M1, M2)

        if M <= 1e-9:
            return -1e18
        return D / M

    # Richer parametric family: 
    # P[y,x] ∝ exp(a*(x-y)^2 + b*x^2 + c*y^2 + d*x + e*y + f*(x-y) + g*x*y + h*x^2*y + i*y^2*x)
    # 9 parameters for a very flexible family
    def params_to_P(params):
        a, b, c, d, e, f, g, h, i = params
        log_w = (a * (X - Y)**2 + b * X**2 + c * Y**2 + 
                 d * X + e * Y + f * (X - Y) + g * X * Y + 
                 h * X**2 * Y + i * Y**2 * X)
        # Shift for numerical stability
        log_w = log_w - log_w.max()
        w = np.exp(log_w)
        s = w.sum()
        if s <= 1e-15:
            return None
        return w / s

    def negative_ratio(params):
        P = params_to_P(params)
        if P is None:
            return 1e18
        return -compute_ratio_fast(P.ravel())

    # Initial best: uniform
    best_P = np.ones((n, n)) / (n * n)
    best_ratio = compute_ratio_fast(best_P.ravel())

    # Define diverse structural priors for multi-start
    np.random.seed(42)
    
    # Generate diverse starting points
    start_params_list = []
    
    # Uniform start
    start_params_list.append(np.zeros(9))
    
    # Diagonal concentration (x-y near 0)
    for a in [0.1, 0.5, 1.0, 2.0]:
        start_params_list.append(np.array([a, 0, 0, 0, 0, 0, 0, 0, 0]))
    
    # Anti-diagonal concentration (x+y near 9)
    for g in [0.1, 0.5, 1.0]:
        start_params_list.append(np.array([0, 0, 0, 0, 0, 0, g, 0, 0]))
    
    # Corner concentrations
    for (dx, dy) in [(0,0), (9,9), (0,9), (9,0)]:
        for scale in [0.5, 1.0, 2.0]:
            d = -scale * dx
            e = -scale * dy
            start_params_list.append(np.array([0, 0, 0, d, e, 0, 0, 0, 0]))
    
    # Random starts
    for _ in range(300):
        params = np.array([
            np.random.uniform(-1, 1),
            np.random.uniform(-1, 1),
            np.random.uniform(-1, 1),
            np.random.uniform(-2, 2),
            np.random.uniform(-2, 2),
            np.random.uniform(-2, 2),
            np.random.uniform(-1, 1),
            np.random.uniform(-0.5, 0.5),
            np.random.uniform(-0.5, 0.5)
        ])
        start_params_list.append(params)
    
    # Multi-start local optimization with Nelder-Mead
    for params0 in start_params_list:
        if time.time() > deadline:
            break
        try:
            result = minimize(
                negative_ratio,
                params0,
                method='Nelder-Mead',
                options={
                    'maxiter': 300,
                    'xatol': 1e-7,
                    'fatol': 1e-9,
                    'adaptive': True
                }
            )
            P_opt = params_to_P(result.x)
            if P_opt is not None:
                ratio = compute_ratio_fast(P_opt.ravel())
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_P = P_opt.copy()
        except Exception:
            pass

    # Direct coordinate-descent refinement on P
    if time.time() < deadline - 30:
        improved = True
        while improved and time.time() < deadline - 15:
            improved = False
            P_work = best_P.copy()
            for yi in range(n):
                for xi in range(n):
                    if time.time() > deadline - 15:
                        break
                    # Try scaling this cell
                    for factor in [0.5, 0.7, 1.3, 1.5, 0.3, 2.0]:
                        P_try = P_work.copy()
                        P_try[yi, xi] *= factor
                        if P_try[yi, xi] < 0:
                            continue
                        ratio = compute_ratio_fast(P_try.ravel())
                        if ratio > best_ratio + 1e-10:
                            best_ratio = ratio
                            best_P = P_try.copy()
                            P_work = P_try.copy()
                            improved = True
                    if time.time() > deadline - 15:
                        break
                if time.time() > deadline - 15:
                    break

    # Final: try structured searches with various parametric forms
    if time.time() < deadline - 10:
        # Concentrate on anti-diagonals: P[y,x] ∝ exp(-alpha * |x - y - c|)
        for c in np.arange(-9, 10, 0.5):
            if time.time() > deadline:
                break
            for alpha in np.arange(0.1, 3.0, 0.1):
                if time.time() > deadline:
                    break
                w = np.exp(-alpha * np.abs(X - Y - c))
                s = w.sum()
                if s <= 1e-15:
                    continue
                P_try = w / s
                ratio = compute_ratio_fast(P_try.ravel())
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_P = P_try.copy()
        
        # Concentrate on main diagonals: P[y,x] ∝ exp(-alpha * |x + y - c|)
        for c in np.arange(0, 19, 0.5):
            if time.time() > deadline:
                break
            for alpha in np.arange(0.1, 3.0, 0.1):
                if time.time() > deadline:
                    break
                w = np.exp(-alpha * np.abs(X + Y - c))
                s = w.sum()
                if s <= 1e-15:
                    continue
                P_try = w / s
                ratio = compute_ratio_fast(P_try.ravel())
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_P = P_try.copy()
        
        # Row/column concentration: P[y,x] ∝ row_w[y] * col_w[x]
        for rw_alpha in np.arange(0.1, 3.0, 0.3):
            if time.time() > deadline:
                break
            for cw_alpha in np.arange(0.1, 3.0, 0.3):
                if time.time() > deadline:
                    break
                row_w = np.exp(-rw_alpha * np.abs(Y - 4.5))
                col_w = np.exp(-cw_alpha * np.abs(X - 4.5))
                w = row_w[:, np.newaxis] * col_w[np.newaxis, :]
                s = w.sum()
                if s <= 1e-15:
                    continue
                P_try = w / s
                ratio = compute_ratio_fast(P_try.ravel())
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_P = P_try.copy()

    # Normalize and clip
    P = best_P.copy()
    P = np.clip(P, 0, None)
    total = P.sum()
    if total <= 1e-15:
        P = np.ones((n, n)) / (n * n)
    else:
        P = P / total

    return P


# EVOLVE-BLOCK-END
>>>>>>> REPLACE
