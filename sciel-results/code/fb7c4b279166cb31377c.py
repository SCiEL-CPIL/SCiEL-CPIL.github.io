# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import factorial
import time

def rudin_problem10() -> np.ndarray:
    """
    Find real coefficients c_m for m = -10..10 of a real degree-10 spherical harmonic
    f = sum_m c_m Y_m that MINIMIZE the L4/L2 norm ratio on the unit sphere.
    
    The evaluation measures 1/R where R = ||f||_4/||f||_2, so higher is better.
    """
    k = 10
    n_coeffs = 2 * k + 1  # 21
    
    # Time budget: 630s wall-clock, keep 30s safety margin
    deadline = time.time() + 600
    
    # --- Precompute quadrature and basis matrices ---
    # 70-point Gauss-Legendre in cos(theta), 130 equally spaced phi
    n_theta = 70
    n_phi = 130
    
    # Gauss-Legendre nodes and weights in [-1, 1]
    x_theta, w_theta = np.polynomial.legendre.leggauss(n_theta)
    
    # Equally spaced phi in [0, 2*pi)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    
    # Build 2D grid
    cos_theta, phi_grid = np.meshgrid(x_theta, phi, indexing='ij')
    # cos_theta shape: (n_theta, n_phi), phi_grid shape: (n_theta, n_phi)
    
    # Weights: w_theta[i] * (2*pi/n_phi) for each phi point
    # Total measure should be 4*pi
    w_phi = 2 * np.pi / n_phi
    weights_2d = np.outer(w_theta, np.ones(n_phi)) * w_phi
    # Verify: sum should be 4*pi
    # assert np.isclose(weights_2d.sum(), 4*np.pi), f"Weight sum {weights_2d.sum()} != 4*pi"
    
    # Flatten for vector operations
    cos_theta_flat = cos_theta.ravel()  # (n_theta * n_phi,)
    phi_flat = phi_grid.ravel()
    weights_flat = weights_2d.ravel()
    n_pts = n_theta * n_phi
    
    # --- Precompute associated Legendre polynomials P_10^m(cos_theta) ---
    # Using recurrence relation with Condon-Shortley phase
    def associated_legendre(l, m, x):
        """Compute P_l^m(x) with Condon-Shortley phase."""
        m = abs(m)
        if m > l:
            return np.zeros_like(x)
        if m == 0:
            # P_l^0
            p_mm = np.ones_like(x)
            if l == 0:
                return p_mm
            # P_1^0
            p_lm1 = x.copy()
            if l == 1:
                return p_lm1
            # P_2^0
            p_ll = (2 * 1 + 1) * x * p_lm1 - p_mm
            p_mm = p_lm1
            p_lm1 = p_ll
            for ll in range(2, l + 1):
                p_ll = ((2 * ll + 1) * x * p_lm1 - (ll - 1 + m) * p_mm) / (ll + m)
                p_mm = p_lm1
                p_lm1 = p_ll
            return p_ll
        else:
            # m > 0
            # Start with P_m^m = (-1)^m (2m-1)!! (1-x^2)^(m/2)
            p_mm = np.ones_like(x)
            if m > 0:
                # (1-x^2)^(m/2)
                one_minus_x2 = np.maximum(1 - x**2, 0)
                for j in range(m):
                    p_mm *= np.sqrt(one_minus_x2)
                # (-1)^m (2m-1)!!
                cs_phase = (-1.0) ** m
                double_fact = 1.0
                for j in range(1, m + 1):
                    double_fact *= (2 * j - 1)
                p_mm *= cs_phase * double_fact
            
            if l == m:
                return p_mm
            # P_{m+1}^m = (2m+1) x P_m^m
            p_lm1 = (2 * m + 1) * x * p_mm
            if l == m + 1:
                return p_lm1
            # P_{m+2}^m
            p_ll = ((2 * (m + 1) + 1) * x * p_lm1 - m * p_mm) / (m + 1)
            p_mm = p_lm1
            p_lm1 = p_ll
            for ll in range(m + 2, l + 1):
                p_ll = ((2 * ll + 1) * x * p_lm1 - (ll - 1 + m) * p_mm) / (ll + m)
                p_mm = p_lm1
                p_lm1 = p_ll
            return p_ll
    
    # Precompute P_10^m(cos_theta) for m = 0, 1, ..., 10
    # Shape: (n_pts, 11) for m = 0..10
    P_m = np.zeros((n_pts, k + 1))
    for m in range(k + 1):
        P_m[:, m] = associated_legendre(k, m, cos_theta_flat)
    
    # Normalization constants N_m = sqrt(21/(4*pi) * (10-m)!/(10+m)!)
    # 21 = 2*(2k+1)/4 = (2k+1)/2 * 2 = ... actually 21 = (2*10+1)/2 * 2? 
    # N_m = sqrt((2k+1)/(4*pi) * (k-m)!/(k+m)!)
    # For k=10: (2*10+1)/(4*pi) = 21/(4*pi)
    N_m = np.zeros(k + 1)
    for m in range(k + 1):
        N_m[m] = np.sqrt(21.0 / (4.0 * np.pi) * factorial(k - m) / factorial(k + m))
    
    # Build basis matrix B of shape (n_pts, 21)
    # Column m (0-indexed) corresponds to harmonic index m_idx = m - 10
    # For m_idx = 0 (m=0): Y_0 = N_0 * P_10^0
    # For m_idx > 0 (m > 0): Y_m = sqrt(2) * N_m * P_10^m * cos(m*phi)
    # For m_idx < 0 (m < 0): Y_m = sqrt(2) * N_|m| * P_10^{|m|} * sin(|m|*phi)
    B = np.zeros((n_pts, n_coeffs))
    
    # m = 0 -> index 10
    B[:, 10] = N_m[0] * P_m[:, 0]
    
    for m in range(1, k + 1):
        # Positive m -> index 10 + m
        B[:, 10 + m] = np.sqrt(2.0) * N_m[m] * P_m[:, m] * np.cos(m * phi_flat)
        # Negative m -> index 10 - m
        B[:, 10 - m] = np.sqrt(2.0) * N_m[m] * P_m[:, m] * np.sin(m * phi_flat)
    
    # Precompute weighted B: B_w = B * weights_flat[:, None]
    # For L2: f = B @ c, ||f||_2^2 = c^T B^T W B c where W = diag(weights)
    # For L4: ||f||_4^4 = sum_i w_i f_i^4
    # Let's precompute B^T W B for L2
    BW = B * weights_flat[:, None]  # (n_pts, 21)
    M2 = B.T @ BW  # (21, 21) = B^T W B
    
    # --- Objective and gradient ---
    def objective_and_grad(c):
        """Compute R = ||f||_4/||f||_2 and its gradient w.r.t. c."""
        f = B @ c  # (n_pts,)
        
        # L2 norm squared
        f2 = f * f
        norm2_sq = f2 @ weights_flat  # sum w_i f_i^2
        if norm2_sq < 1e-30:
            return 1e10, np.zeros_like(c)
        
        # L4 norm to the 4th power
        f4 = f2 * f2
        norm4_4 = f4 @ weights_flat  # sum w_i f_i^4
        
        # R = ||f||_4 / ||f||_2 = (norm4_4)^(1/4) / (norm2_sq)^(1/2)
        # Let's work with R^2 = norm4_4 / norm2_sq^2 to avoid 4th roots
        # Actually, let's just compute R directly
        norm4 = np.sqrt(np.sqrt(norm4_4))
        norm2 = np.sqrt(norm2_sq)
        R = norm4 / norm2
        
        # Gradient of R = ||f||_4 / ||f||_2
        # d/dc (||f||_4 / ||f||_2) = (1/||f||_2) d||f||_4/dc - (||f||_4/||f||_2^2) d||f||_2/dc
        # d||f||_4/dc = (1/4) ||f||_4^{-3} * 4 * sum w_i f_i^3 * df_i/dc
        #             = ||f||_4^{-3} * sum w_i f_i^3 * B_i
        # d||f||_2/dc = (1/2) ||f||_2^{-1} * 2 * sum w_i f_i * B_i
        #             = ||f||_2^{-1} * sum w_i f_i * B_i
        
        # Let A = sum w_i f_i^3 * B_i  (21-vector)
        # Let D = sum w_i f_i * B_i    (21-vector)
        A = B.T @ (weights_flat * f2 * f)  # B^T (w * f^3)
        D = B.T @ (weights_flat * f)       # B^T (w * f)
        
        # dR/dc = (1/norm2) * (A / norm4^3) - (norm4/norm2^2) * (D / norm2)
        #       = A / (norm2 * norm4^3) - norm4 * D / (norm2^3)
        #       = (A * norm2^2 - norm4^2 * D) / (norm2^3 * norm4^3)
        # Simplify: = (A - (norm4/norm2)^2 * D) / (norm2 * norm4^3) * norm2
        # Let's just compute directly
        grad_R = A / (norm2 * norm4**3) - (norm4 / norm2**3) * D
        
        return R, grad_R
    
    # --- Trust region / Newton-type method ---
    # We want to minimize R(c) subject to |c| in [1e-3, 1e5] range
    # Use a projected Newton/trust-region approach
    
    # Precompute B^T W B (already have M2)
    # Hessian of R is complex; use quasi-Newton or Gauss-Newton approximation
    
    # For the Gauss-Newton approximation of the Hessian of R:
    # R = ||f||_4 / ||f||_2
    # Let g = log(R) = 0.5 log(norm4_4) - log(norm2_sq)
    # Minimizing R is equivalent to minimizing g (monotone transform)
    # dg/dc = 0.5 * (1/norm4_4) * 4 * sum w_i f_i^3 B_i - (1/norm2_sq) * 2 * sum w_i f_i B_i
    #       = 2 * A / norm4_4 - 2 * D / norm2_sq
    # Hessian of g:
    # d2g/dc2 = 2 * d(A/norm4_4)/dc - 2 * d(D/norm2_sq)/dc
    # d(A_j)/dc_k = sum_i w_i 3 f_i^2 B_ij B_ik
    # d(D_j)/dc_k = sum_i w_i B_ij B_ik = (M2)_jk
    
    # This is getting complex. Let's use L-BFGS-B with analytic gradient, which is known to work well.
    
    from scipy.optimize import minimize
    
    def obj(c):
        R, _ = objective_and_grad(c)
        return R
    
    def grad(c):
        _, g = objective_and_grad(c)
        return g
    
    # Multiple restarts with different initial conditions
    best_R = np.inf
    best_c = np.ones(n_coeffs)
    
    rng = np.random.default_rng(42)
    
    # Initial guesses
    init_list = []
    init_list.append(np.ones(n_coeffs))
    init_list.append(np.zeros(n_coeffs))
    init_list.append(np.zeros(n_coeffs))
    init_list[-1][10] = 1.0  # Only m=0
    
    # Random inits
    for _ in range(10):
        c_init = rng.standard_normal(n_coeffs)
        c_init /= np.linalg.norm(c_init)
        init_list.append(c_init)
    
    # Symmetric patterns
    # All even m
    c_even = np.zeros(n_coeffs)
    for idx in range(0, n_coeffs, 2):
        c_even[idx] = 1.0
    init_list.append(c_even)
    
    # All odd m
    c_odd = np.zeros(n_coeffs)
    for idx in range(1, n_coeffs, 2):
        c_odd[idx] = 1.0
    init_list.append(c_odd)
    
    # Alternating signs
    c_alt = np.zeros(n_coeffs)
    for idx in range(n_coeffs):
        c_alt[idx] = 1.0 if idx % 2 == 0 else -1.0
    init_list.append(c_alt)
    
    for c_init in init_list:
        if time.time() > deadline - 60:
            break
        
        try:
            res = minimize(
                obj, c_init, jac=grad,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            R_res = res.fun
            if R_res < best_R:
                best_R = R_res
                best_c = res.x.copy()
        except Exception:
            pass
    
    # Additional random restarts if time permits
    n_extra = 0
    while time.time() < deadline - 90:
        n_extra += 1
        c_init = rng.standard_normal(n_coeffs)
        c_init /= np.linalg.norm(c_init)
        try:
            res = minimize(
                obj, c_init, jac=grad,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            R_res = res.fun
            if R_res < best_R:
                best_R = R_res
                best_c = res.x.copy()
        except Exception:
            pass
        if n_extra >= 20:
            break
    
    # Final polish with tighter tolerances
    if time.time() < deadline - 30 and np.isfinite(best_R):
        try:
            res = minimize(
                obj, best_c, jac=grad,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res.fun < best_R:
                best_R = res.fun
                best_c = res.x.copy()
        except Exception:
            pass
    
    # Validate and return
    if not np.all(np.isfinite(best_c)):
        best_c = np.ones(n_coeffs)
    
    max_abs = np.max(np.abs(best_c))
    if max_abs < 1e-3:
        best_c = best_c / max_abs * 1e-2
    elif max_abs > 1e5:
        best_c = best_c / max_abs * 1e4
    
    return best_c.astype(np.float64)

# EVOLVE-BLOCK-END
