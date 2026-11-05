"""Compact dark-energy growth-index fit: Dual-space Chebyshev projection strategy."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

# Fixed internal deadline with a safety margin (evaluator kills at 90 s).
_DEADLINE = 75.0

# Cosmology grid: a in (0, 1] mapped to x = ln a in [-3, 0].
_A_GRID = np.exp(np.linspace(-3.0, 0.0, 240))
_X_GRID = np.log(_A_GRID)


def _omega_m(a, om0, w0, wa):
    a2 = a * a
    a3 = a2 * a
    # Handle wa = -1 case (though task implies smooth CPL, usually wa != -1)
    if abs(wa + 1.0) < 1e-12:
        E = a ** (3.0 * (w0 + 1.0))
    else:
        E = np.exp(3.0 * (w0 + 1.0 + wa) * (1.0 - a) / (1.0 + wa))
    return om0 / (om0 + (1.0 - om0) * a3 * E)


def _gamma_of(c, a, om0, w0, wa):
    om = _omega_m(a, om0, w0, wa)
    q = 1.0 - om
    u = 1.0 + w0 + wa * (1.0 - a)
    return (c[0] + c[1] * q + c[2] * q * q + c[3] * u + c[4] * wa * a + c[5] * u * q)


def _f_of(c, a, om0, w0, wa):
    om = _omega_m(a, om0, w0, wa)
    g = _gamma_of(c, a, om0, w0, wa)
    return om ** g


def _norm_d_of(c, om0, w0, wa):
    f = _f_of(c, _A_GRID, om0, w0, wa)
    integ = np.zeros_like(_X_GRID)
    integ[1:] = np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(_X_GRID))
    total = integ[-1]
    val = total - integ
    return np.exp(-val)


def _build_reference(om0, w0, wa):
    """Solve the second-order GR growth ODE for D(a) and f(a) = D'/(a D)."""
    N = 240
    x = np.linspace(-3.0, 0.0, N)
    a = np.exp(x)
    om = _omega_m(a, om0, w0, wa)
    
    # Numerical derivative of om
    da = np.diff(a)
    om_prime = np.zeros_like(a)
    om_prime[1:] = (om[1:] - om[:-1]) / da
    om_prime[0] = om_prime[1]
    
    # Equation: a f' + f^2 = (3/2) a (om + om')
    # Let g = f. a g' = (3/2) a (om + om') - g^2
    # This is a Riccati equation. We integrate from a=1 backwards? 
    # Actually, standard practice is to integrate from small a to large a or vice versa.
    # Let's integrate from a=1 (x=0) to a=exp(-3) (x=-3) is unstable for D.
    # Better: Integrate from a=exp(-3) to a=1.
    # f(a) = D'/D.
    # f' = (3/2)om + (3/2)om' - f^2 (d/da)
    # d/dx = a d/da.
    # a f' = (3/2) a om + (3/2) a om' - a f^2
    # Let's just use the quadratic solution derived in the original code which is robust for the range.
    # (a-1) g^2 + g - (3/2) a (om+om') = 0
    rhs = 1.5 * a * (om + om_prime)
    Acoef = (a - 1.0)
    disc = 1.0 + 4.0 * Acoef * rhs
    disc = np.maximum(disc, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        g = (-1.0 + np.sqrt(disc)) / (2.0 * Acoef)
    
    # Fix a=1 limit: g(1) = 0.5 om(1) = 0.5 om0
    g[0] = 0.5 * om[0]
    
    # D(a)/D(1)
    lnD = np.zeros_like(x)
    lnD[1:] = np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(x))
    D = np.exp(lnD)
    D = D / D[-1]
    return a, g, D


def _objective(c, models):
    worst = 0.0
    for (om0, w0, wa) in models:
        g = _gamma_of(c, _A_GRID, om0, w0, wa)
        if not (np.all(g >= 0.1) and np.all(g <= 1.2)):
            viol = np.maximum(0.1 - g, 0.0) + np.maximum(g - 1.2, 0.0)
            return 1.0 + float(np.max(viol))
        f = _f_of(c, _A_GRID, om0, w0, wa)
        D = _norm_d_of(c, om0, w0, wa)
        _, ref_f, ref_D = _build_reference(om0, w0, wa)
        er_f = np.max(np.abs((f - ref_f) / np.maximum(ref_f, 1e-12)))
        er_D = np.max(np.abs((D - ref_D) / np.maximum(np.abs(ref_D), 1e-12)))
        worst = max(worst, er_f, er_D)
    return float(worst)


def _nelder_mead(c0, obj, max_iter=400, tol=1e-6):
    n = len(c0)
    simplex = [np.array(c0, dtype=float)]
    base = 0.05
    for i in range(n):
        v = np.array(c0, dtype=float)
        v[i] += base
        simplex.append(v)
    fvals = [obj(v) for v in simplex]
    best_i = int(np.argmin(fvals))
    for _ in range(max_iter):
        order = sorted(range(n + 1), key=lambda i: fvals[i])
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        if abs(fvals[-1] - fvals[0]) < tol:
            break
        centroid = np.mean(simplex[:-1], axis=0)
        xr = centroid + (centroid - simplex[-1])
        fr = obj(xr)
        if fr < fvals[0]:
            xe = centroid + 2.0 * (centroid - simplex[-1])
            fe = obj(xe)
            if fe < fr:
                simplex[-1], fvals[-1] = xe, fe
            else:
                simplex[-1], fvals[-1] = xr, fr
        elif fr < fvals[-2]:
            simplex[-1], fvals[-1] = xr, fr
        else:
            xc = centroid + 0.5 * (simplex[-1] - centroid)
            fc = obj(xc)
            if fc < fvals[-1]:
                simplex[-1], fvals[-1] = xc, fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = 0.5 * (simplex[i] + simplex[0])
                    fvals[i] = obj(simplex[i])
    best_i = int(np.argmin(fvals))
    return simplex[best_i], fvals[best_i]


def _chebyshev_projection_fit(models, seed=12345):
    """
    Structural change: Use Chebyshev nodes in the parameter space to construct
    a weighted least-squares problem that approximates the minimax solution.
    """
    rng = np.random.default_rng(seed)
    
    # 1. Generate a dense set of "training" cosmologies to cover the parameter space.
    # We use the 6 provided models plus a grid of neighbors to ensure the fit is robust.
    base_models = [
        (0.30, -0.9, 0.0),
        (0.28, -1.0, 0.0),
        (0.31, -1.1, 0.3),
        (0.27, -0.8, -0.3),
        (0.30, -1.0, 0.5),
        (0.33, -0.95, 0.2),
    ]
    
    # Add a grid of models to span the space more uniformly
    fit_models = list(base_models)
    for om0 in [0.25, 0.28, 0.30, 0.32, 0.35]:
        for w0 in [-1.2, -1.0, -0.8, -0.6]:
            for wa in [-0.5, 0.0, 0.5]:
                fit_models.append((om0, w0, wa))
    
    # 2. Pre-compute references for the fit models
    ref_cache = {}
    for m in fit_models:
        ref_cache[m] = _build_reference(*m)
        
    # 3. Construct the basis matrix on a Chebyshev-like grid of (q, u)
    # We sample (q, u) from the fit_models on the a-grid.
    # To make it a "projection", we pick a representative set of (q, u) points.
    # Let's use the points from the first few models to define the geometry.
    
    # Collect all (q, u, wa, a) points from the fit models
    all_q = []
    all_u = []
    all_wa = []
    all_a = []
    all_ref_f = []
    all_ref_D = []
    
    for m in fit_models:
        om0, w0, wa = m
        a, ref_f, ref_D = ref_cache[m]
        om = _omega_m(a, om0, w0, wa)
        q = 1.0 - om
        u = 1.0 + w0 + wa * (1.0 - a)
        
        all_q.append(q)
        all_u.append(u)
        all_wa.append(np.full_like(a, wa))
        all_a.append(a)
        all_ref_f.append(ref_f)
        all_ref_D.append(ref_D)
        
    Q = np.concatenate(all_q)
    U = np.concatenate(all_u)
    WA = np.concatenate(all_wa)
    A = np.concatenate(all_a)
    REF_F = np.concatenate(all_ref_f)
    REF_D = np.concatenate(all_ref_D)
    
    # The model is gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
    # We want to find c such that f_model = Om^gamma approximates REF_F
    # and D_model approximates REF_D.
    # Since D depends on the integral of f, fitting f well is the primary driver.
    # We will fit gamma directly to minimize the error in f.
    
    # Linearize? No, gamma is in the exponent.
    # However, we can use an iterative re-weighting scheme (Remez/Minimax approx).
    # Start with an initial guess for c.
    
    c_curr = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Basis matrix B where B[i, j] is the j-th basis function value at point i
    # Basis: 1, q, q^2, u, wa*a, u*q
    B = np.column_stack([
        np.ones_like(Q),
        Q,
        Q**2,
        U,
        WA * A,
        U * Q
    ])
    
    # Iterative Minimax Approximation (Remez-like for non-linear)
    # We want to minimize max |error|.
    # We approximate this by minimizing sum w_i * error_i^2 where w_i are updated.
    
    max_iter_cheb = 10
    for _ in range(max_iter_cheb):
        # Compute current gamma, f, D
        gamma_curr = B @ c_curr
        # Ensure gamma is in valid range for exp/log stability in error calc
        gamma_safe = np.clip(gamma_curr, 0.1, 1.2)
        
        # Compute model f
        # Om(a) = 1 - q
        Om = 1.0 - Q
        f_curr = Om ** gamma_safe
        
        # Compute model D
        # This is expensive to do for all points at once because D depends on integration.
        # However, for the projection step, we can approximate the error in f as the primary metric.
        # The error in D is correlated with the error in f.
        # For speed in the Chebyshev projection, we will focus on f-error and use D-error
        # only in the final refinement.
        
        err_f = f_curr - REF_F
        # Relative error
        rel_err_f = np.abs(err_f) / np.maximum(REF_F, 1e-12)
        
        # Update weights: w_i = 1 / (rel_err_i + eps)
        # This pushes the fit towards equalizing the relative errors (equioscillation)
        eps = 1e-6
        weights = 1.0 / (rel_err_f + eps)
        
        # Normalize weights to prevent overflow
        weights = weights / np.sum(weights)
        
        # Solve weighted least squares: min sum w_i * (B_i @ c - target_i)^2
        # Target for gamma? We don't have a direct linear target for gamma.
        # We have a target for f.
        # This is non-linear.
        # Let's use a Gauss-Newton step on gamma, then map back to c?
        # Or simpler: Use the linear structure of B to update c such that the change in gamma
        # reduces the weighted error in f.
        
        # Let's approximate the Jacobian of f w.r.t c.
        # f = Om^gamma. df/dgamma = f * ln(Om).
        # dgamma/dc = B.
        # J = df/dc = (f * ln(Om)) * B
        
        log_Om = np.log(Om)
        J_factor = f_curr * log_Om
        J = B * J_factor[:, np.newaxis]
        
        # Weighted Gauss-Newton:
        # (J^T W J) dc = - J^T W err
        W = np.diag(weights) # This is huge, avoid explicit diag
        # J^T W J = (J * weights[:, np.newaxis])^T J
        JW = J * weights[:, np.newaxis]
        H = JW.T @ J
        g_vec = JW.T @ err_f
        
        # Add small regularization for stability
        H += 1e-8 * np.eye(6)
        
        try:
            dc = np.linalg.solve(H, -g_vec)
        except np.linalg.LinAlgError:
            dc = np.linalg.lstsq(H, -g_vec, rcond=None)[0]
            
        c_curr = c_curr + dc
        c_curr = np.clip(c_curr, -10.0, 10.0)
        
        # Check convergence
        if np.max(np.abs(dc)) < 1e-8:
            break
            
    # 4. Refine the Chebyshev-proposed solution with Nelder-Mead on the actual objective
    # The Chebyshev step gives a good global structure. NM will polish it.
    
    def obj(c):
        worst = 0.0
        for m in base_models:
            om0, w0, wa = m
            g = _gamma_of(c, _A_GRID, om0, w0, wa)
            if not (np.all(g >= 0.1) and np.all(g <= 1.2)):
                viol = np.maximum(0.1 - g, 0.0) + np.maximum(g - 1.2, 0.0)
                return 1.0 + float(np.max(viol))
            f = _f_of(c, _A_GRID, om0, w0, wa)
            D = _norm_d_of(c, om0, w0, wa)
            _, ref_f, ref_D = ref_cache[m]
            er_f = np.max(np.abs((f - ref_f) / np.maximum(ref_f, 1e-12)))
            er_D = np.max(np.abs((D - ref_D) / np.maximum(np.abs(ref_D), 1e-12)))
            worst = max(worst, er_f, er_D)
        return float(worst)
        
    best_c = c_curr.copy()
    best_val = obj(best_c)
    
    # Multi-start refinement from Chebyshev solution
    starts = 0
    max_starts = 10
    while time.time() < _DEADLINE - 10.0 and starts < max_starts:
        starts += 1
        # Perturb from Chebyshev solution
        c0 = best_c + rng.normal(0.0, 0.05, size=6)
        c0 = np.clip(c0, -10.0, 10.0)
        c, val = _nelder_mead(c0, obj, max_iter=200, tol=1e-7)
        c = np.clip(c, -10.0, 10.0)
        val = obj(c)
        if val < best_val:
            best_val = val
            best_c = c
            
    return best_c, best_val


def construct():
    t0 = time.time()
    # Public validation-style cosmologies (disjoint, smooth CPL):
    models = [
        (0.30, -0.9, 0.0),
        (0.28, -1.0, 0.0),
        (0.31, -1.1, 0.3),
        (0.27, -0.8, -0.3),
        (0.30, -1.0, 0.5),
        (0.33, -0.95, 0.2),
    ]
    
    # Use Chebyshev projection fit
    best_c, best_val = _chebyshev_projection_fit(models, seed=12345)
    
    # Final safety: ensure finite and in range
    if not np.all(np.isfinite(best_c)):
        best_c = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_c = np.clip(best_c, -10.0, 10.0)
    return [float(x) for x in best_c]
# EVOLVE-BLOCK-END
