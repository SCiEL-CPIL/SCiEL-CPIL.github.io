import numpy as np
from scipy.special import lpmv, factorial
from scipy.optimize import minimize
import time

def _build_harmonics():
    """Precompute real orthonormal degree-10 spherical harmonics on the quadrature grid."""
    k = 10
    x, w = np.polynomial.legendre.leggauss(70)
    phi = np.linspace(0, 2*np.pi, 130, endpoint=False)
    
    cos_theta, phi_grid = np.meshgrid(x, phi, indexing='ij')
    dphi = 2*np.pi / 130
    weights_2d = np.outer(w, np.ones(130)) * dphi
    
    P = {}
    for m in range(0, 11):
        P[m] = lpmv(m, 10, cos_theta)
    
    N = {}
    for m in range(0, 11):
        N[m] = np.sqrt(21.0/(4*np.pi) * factorial(10-m)/factorial(10+m))
    
    Y = np.zeros((21, 70, 130))
    for idx, m in enumerate(range(-10, 11)):
        if m == 0:
            Y[idx] = N[0] * P[0]
        elif m > 0:
            Y[idx] = np.sqrt(2) * N[m] * P[m] * np.cos(m * phi_grid)
        else:
            am = abs(m)
            Y[idx] = np.sqrt(2) * N[am] * P[am] * np.sin(am * phi_grid)
    
    return Y, weights_2d

def _compute_f(coeffs, Y):
    """Compute f = sum_m c_m Y_m on the grid."""
    return np.tensordot(coeffs, Y, axes=(0, 0))

def _objective_value(coeffs, Y, weights_2d):
    """Compute ||f||_4 / ||f||_2."""
    f = _compute_f(coeffs, Y)
    w = weights_2d
    norm2_sq = np.sum(f**2 * w)
    if norm2_sq < 1e-30:
        return 1e10
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sum(np.abs(f)**4 * w)**0.25
    return norm4 / norm2

def _minimize_f4_given_norm2(coeffs, Y, weights_2d):
    """
    Minimize ∫f⁴ subject to ∫f² = 1, using the Euler-Lagrange equation.
    
    The EL equation projected onto the harmonic subspace:
    ∫f³ Y_m w = (λ/2) c_m + (μ/2) ∫Y_m w
    
    Since we're in a finite-dimensional subspace, we can write:
    Let G_m = ∫Y_m w (these are constants, mostly zero for m≠0 due to symmetry)
    Let H_m = ∫f³ Y_m w
    
    The condition is: H = (λ/2) c + (μ/2) g  where g_m = ∫Y_m w
    
    But actually, since the constraint is ∫f² = 1 and we minimize ∫f⁴,
    the Lagrangian is L = ∫f⁴ - λ(∫f² - 1) - μ ∫f (no, no linear constraint)
    
    Actually: minimize ∫f⁴ s.t. ∫f² = 1
    L = ∫f⁴ - λ(∫f² - 1)
    dL/dc_m = 4∫f³ Y_m w - 2λ ∫f Y_m w = 0
    
    Since ∫f Y_m w = c_m (orthonormality), we get:
    4∫f³ Y_m w = 2λ c_m
    => 2∫f³ Y_m w = λ c_m
    
    So the system is: 2 * (f³ projected onto each Y_m) = λ * c
    
    This means c is proportional to the projection of f³.
    Since f = sum c_m Y_m, and f³ is a function, its projection gives us a nonlinear map.
    
    We can solve this by fixed-point iteration:
    1. Start with c
    2. Compute f = sum c_m Y_m
    3. Compute p_m = ∫f³ Y_m w
    4. Set c_new = p (this is the direction that satisfies the EL equation with λ=1)
    5. Normalize c_new to unit norm
    6. Iterate until convergence
    
    This is like a power iteration for the "cubic" operator.
    """
    c = coeffs.copy()
    norm_c = np.linalg.norm(c)
    if norm_c < 1e-15:
        return coeffs
    
    # Normalize to unit L2 norm (since ∫f² = ||c||² for orthonormal basis)
    c = c / norm_c
    
    w = weights_2d
    prev_c = None
    for it in range(500):
        f = _compute_f(c, Y)
        f3 = f**3
        # Project f³ onto each harmonic
        p = np.array([np.sum(f3 * Y[m] * w) for m in range(21)])
        
        # The EL condition says c should be proportional to p
        # But we need to be careful: the condition is 2*p = λ*c, so c ∝ p
        new_c = p.copy()
        new_norm = np.linalg.norm(new_c)
        if new_norm < 1e-15:
            break
        new_c = new_c / new_norm
        
        # Check convergence
        if prev_c is not None:
            diff = np.linalg.norm(new_c - prev_c)
            if diff < 1e-14:
                break
        prev_c = new_c
        c = new_c
    
    return c

def _newton_solve(coeffs, Y, weights_2d):
    """
    Solve the nonlinear system using Newton's method.
    
    System: F(c) = 2 * proj(f³) - λ * c = 0, with ||c|| = 1
    But λ is unknown. We can eliminate λ by noting that at the solution,
    c is parallel to proj(f³).
    
    Alternative: minimize ∫f⁴ subject to ||c||² = 1 using constrained optimization.
    Use the fact that the gradient of ∫f⁴ w.r.t. c is 4*proj(f³), and at the
    constrained minimum, grad(∫f⁴) = 2λ c.
    
    So we minimize g(c) = ∫f⁴ subject to ||c||² = 1.
    Use projected gradient or Newton on the sphere.
    
    For Newton: the Hessian of ∫f⁴ is 12 * ∫f² Y_m Y_n w (a matrix).
    The KKT system is:
    H * Δc = -grad + ν c
    c·Δc = 0
    where H_mn = 12 ∫f² Y_m Y_n w, grad_m = 4 ∫f³ Y_m w
    
    Solve this 22x22 linear system at each Newton step.
    """
    c = coeffs.copy()
    norm_c = np.linalg.norm(c)
    if norm_c < 1e-15:
        return coeffs
    c = c / norm_c
    
    w = weights_2d
    n = 21
    
    for newton_iter in range(100):
        f = _compute_f(c, Y)
        f2 = f**2
        f3 = f**3
        
        # Gradient: grad_m = 4 ∫f³ Y_m w
        grad = np.array([4.0 * np.sum(f3 * Y[m] * w) for m in range(n)])
        
        # Hessian: H_mn = 12 ∫f² Y_m Y_n w
        # Compute as: H = 12 * Y^T * diag(f² * w) * Y
        # Efficiently: for each grid point, outer product of Y values weighted by f² * w
        # H_mn = 12 * sum_{i,j} f²_{ij} * Y_m(i,j) * Y_n(i,j) * w_{ij}
        # = 12 * sum_{ij} (Y_m * sqrt(f²*w))_{ij} * (Y_n * sqrt(f²*w))_{ij}
        
        # Build the matrix: H = 12 * (Y * sqrt(w*f²))^T @ (Y * sqrt(w*f²))
        # Y is (21, 70, 130), w*f² is (70, 130)
        weighted_Y = Y * np.sqrt(np.maximum(f2 * w, 0))[None, :, :]
        H = 12.0 * np.einsum('mij,nij->mn', weighted_Y, weighted_Y)
        
        # KKT system:
        # [H   c] [Δc]   [-grad]
        # [c^T 0] [ν  ] = [  0  ]
        
        A = np.zeros((n+1, n+1))
        A[:n, :n] = H
        A[:n, n] = c
        A[n, :n] = c
        b = np.zeros(n+1)
        b[:n] = -grad
        
        # Solve
        try:
            sol = np.linalg.solve(A, b)
            delta_c = sol[:n]
            nu = sol[n]
        except np.linalg.LinAlgError:
            # Use least squares
            try:
                sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                delta_c = sol[:n]
                nu = sol[n]
            except:
                break
        
        # Line search
        alpha = 1.0
        f4_current = np.sum(f4 := f**4 * w)
        
        for ls_iter in range(30):
            c_trial = c + alpha * delta_c
            c_trial_norm = np.linalg.norm(c_trial)
            if c_trial_norm < 1e-15:
                alpha *= 0.5
                continue
            c_trial = c_trial / c_trial_norm
            
            f_trial = _compute_f(c_trial, Y)
            f4_trial = np.sum(f_trial**4 * w)
            
            if f4_trial < f4_current:
                c = c_trial
                break
            alpha *= 0.5
        
        else:
            break
        
        # Check convergence
        if np.linalg.norm(delta_c) < 1e-12:
            break
    
    return c

def _random_direction():
    """Generate a random unit vector in 21D."""
    v = np.random.randn(21)
    return v / np.linalg.norm(v)

def _structured_directions():
    """Generate diverse structured initial directions."""
    dirs = []
    
    # All ones
    v = np.ones(21) / np.sqrt(21)
    dirs.append(v)
    
    # Only m=0
    v = np.zeros(21); v[10] = 1.0
    dirs.append(v)
    
    # Only |m|=10
    v = np.zeros(21); v[0] = 1.0/np.sqrt(2); v[20] = 1.0/np.sqrt(2)
    dirs.append(v)
    
    # Only |m|>=8
    v = np.zeros(21)
    for m in [8, 9, 10]:
        v[10+m] = 1.0/np.sqrt(6)
        v[10-m] = 1.0/np.sqrt(6)
    dirs.append(v)
    
    # Cosine-weighted
    v = np.array([1.0/np.sqrt(21) for _ in range(21)])
    for i, m in enumerate(range(-10, 11)):
        v[i] = np.cos(np.pi * m / 20) / np.sqrt(21)
    v = v / np.linalg.norm(v)
    dirs.append(v)
    
    # Sine-weighted (emphasize high m)
    v = np.zeros(21)
    for i, m in enumerate(range(-10, 11)):
        v[i] = np.sin(np.pi * (abs(m)+1) / 11)
    v = v / np.linalg.norm(v)
    dirs.append(v)
    
    # Alternating signs
    v = np.zeros(21)
    for i, m in enumerate(range(-10, 11)):
        v[i] = (1.0 if m % 2 == 0 else -1.0) / np.sqrt(21)
    dirs.append(v)
    
    # Random with fixed seed for reproducibility
    rng = np.random.RandomState(42)
    for _ in range(10):
        v = rng.randn(21)
        v = v / np.linalg.norm(v)
        dirs.append(v)
    
    return dirs

def rudin_problem10() -> np.ndarray:
    """
    Find coefficients for degree-10 spherical harmonic minimizing L4/L2 ratio.
    Uses Newton's method on the KKT conditions plus fixed-point iteration,
    with multiple diverse initializations.
    """
    start_time = time.time()
    deadline = 600.0
    
    # Precompute harmonics and weights
    Y, weights_2d = _build_harmonics()
    
    np.random.seed(12345)
    
    best_coeffs = None
    best_score = np.inf
    
    def _update_best(c):
        nonlocal best_coeffs, best_score
        if c is None:
            return
        if not np.all(np.isfinite(c)):
            return
        s = _objective_value(c, Y, weights_2d)
        if s < best_score and np.isfinite(s):
            best_score = s
            best_coeffs = c.copy()
    
    # Phase 1: Try structured directions with both fixed-point and Newton
    for d in _structured_directions():
        if time.time() - start_time > deadline - 120:
            break
        
        # Fixed-point iteration
        c_fp = _minimize_f4_given_norm2(d, Y, weights_2d)
        _update_best(c_fp)
        
        # Newton from the fixed-point result
        if time.time() - start_time < deadline - 60:
            c_newton = _newton_solve(c_fp, Y, weights_2d)
            _update_best(c_newton)
    
    # Phase 2: Random restarts with Newton
    n_restarts = 60
    for i in range(n_restarts):
        if time.time() - start_time > deadline - 90:
            break
        
        # Diverse random starts
        if i < 20:
            # Sparse random
            n_active = np.random.randint(3, 15)
            active_idx = np.random.choice(21, size=n_active, replace=False)
            x0 = np.zeros(21)
            x0[active_idx] = np.random.randn(n_active)
            x0 = x0 / np.linalg.norm(x0)
        elif i < 40:
            # Random with sign structure
            x0 = np.random.choice([-1, 1], size=21) * np.abs(np.random.randn(21))
            x0 = x0 / np.linalg.norm(x0)
        else:
            x0 = _random_direction()
        
        # Newton solve
        c_newton = _newton_solve(x0, Y, weights_2d)
        _update_best(c_newton)
        
        # Also try fixed-point from the Newton result
        c_fp = _minimize_f4_given_norm2(c_newton, Y, weights_2d)
        _update_best(c_fp)
    
    # Phase 3: Local refinement from best solution
    if best_coeffs is not None and time.time() - start_time < deadline - 30:
        # Small perturbations + Newton
        for i in range(30):
            if time.time() - start_time > deadline - 15:
                break
            x0 = best_coeffs + np.random.randn(21) * 0.1
            x0 = x0 / np.linalg.norm(x0)
            c_newton = _newton_solve(x0, Y, weights_2d)
            _update_best(c_newton)
    
    # Final: if we have a best, do one more high-precision Newton
    if best_coeffs is not None and time.time() - start_time < deadline - 10:
        c_final = _newton_solve(best_coeffs, Y, weights_2d)
        _update_best(c_final)
    
    # Validate and return
    if best_coeffs is None:
        return np.ones(21)
    
    if len(best_coeffs) != 21 or not np.all(np.isfinite(best_coeffs)):
        return np.ones(21)
    
    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3:
        best_coeffs = best_coeffs / max_abs if max_abs > 0 else np.ones(21)
    elif max_abs > 1e5:
        best_coeffs = best_coeffs * (1e5 / max_abs)
    
    if not np.all(np.isfinite(best_coeffs)):
        return np.ones(21)
    
    return best_coeffs.astype(np.float64)
