"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    def get_sq_dists(pts):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        sq_dists = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        return sq_dists[iu]

    def score_raw(pts):
        d_sq = get_sq_dists(pts)
        d_min = np.sqrt(d_sq.min())
        d_max = np.sqrt(d_sq.max())
        if d_min < 1e-12:
            return 1e10
        return (d_max / d_min) ** 2

    def normalize(pts):
        pts = pts - np.mean(pts, axis=0)
        scale = np.sqrt(np.mean(np.sum(pts**2, axis=1)))
        if scale > 1e-9:
            pts = pts / scale
        return pts

    # Fourier parameterization: points on a perturbed circle
    # p_i = (1 + a*cos(m*theta_i) + b*sin(m*theta_i)) * (cos(theta_i), sin(theta_i))
    # where theta_i = 2*pi*i/N
    # We optimize the Fourier coefficients a, b for modes m=2,3,4
    
    def fourier_to_points(coeffs):
        """Convert Fourier coefficients to point positions.
        coeffs: array of shape (n_modes, 3) where each row is [a_m, b_m, phase_m]
        """
        n_modes = len(coeffs)
        theta = 2 * np.pi * np.arange(N) / N
        
        # Base radius 1
        r = np.ones(N)
        angle_offset = np.zeros(N)
        
        for m_idx, (a, b, phase) in enumerate(coeffs):
            m = m_idx + 2  # Start from mode 2 (mode 1 would just rotate)
            r += a * np.cos(m * theta + phase) + b * np.sin(m * theta + phase)
        
        # Ensure r is positive (add offset if needed)
        r = np.maximum(r, 0.1)
        
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        return np.column_stack([x, y])

    def smooth_objective_fourier(coeffs):
        pts = fourier_to_points(coeffs)
        d_sq = get_sq_dists(pts)
        d = np.sqrt(d_sq)
        
        k = 10.0
        d_max_val = np.max(d)
        d_min_val = np.min(d)
        
        exp_term_max = np.exp(k * (d - d_max_val))
        soft_max = d_max_val + (1.0 / k) * np.log(np.sum(exp_term_max))
        
        exp_term_min = np.exp(-k * (d - d_min_val))
        soft_min = d_min_val - (1.0 / k) * np.log(np.sum(exp_term_min))
        
        if soft_min < 1e-12:
            return 1e10
        
        ratio = soft_max / soft_min
        return ratio ** 2

    # Initializations for Fourier coefficients
    n_modes = 3  # Modes 2, 3, 4
    n_params = n_modes * 3  # a, b, phase for each mode
    
    inits = []
    
    # 1. Zero coefficients (perfect circle)
    inits.append(np.zeros(n_params))
    
    # 2. Small random perturbations
    for _ in range(10):
        coeffs = np.random.normal(0, 0.05, n_params)
        inits.append(coeffs)
    
    # 3. Larger perturbations
    for _ in range(5):
        coeffs = np.random.normal(0, 0.15, n_params)
        inits.append(coeffs)
    
    # 4. Structured initializations
    # Mode 2 dominant (ellipse-like)
    c = np.zeros(n_params)
    c[0] = 0.1  # a_2
    c[2] = 0.0  # phase_2
    inits.append(c)
    
    # Mode 3 dominant (triangular perturbation)
    c = np.zeros(n_params)
    c[3] = 0.08  # a_3
    c[5] = 0.0  # phase_3
    inits.append(c)
    
    # Mixed modes
    c = np.zeros(n_params)
    c[0] = 0.05
    c[3] = 0.03
    c[6] = 0.02
    inits.append(c)
    
    best_coeffs = None
    best_score = 1e10
    best_pts = None
    
    for x0 in inits:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(
                smooth_objective_fourier,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            final_pts = fourier_to_points(res.x)
            raw_score = score_raw(final_pts)
            
            if raw_score < best_score:
                best_score = raw_score
                best_coeffs = res.x
                best_pts = final_pts
        except Exception:
            continue
    
    # Also try direct optimization from good Fourier solutions
    if best_pts is not None:
        if time.time() < deadline - 20:
            try:
                # Refine with direct point optimization
                def direct_objective(x):
                    pts = x.reshape(N, 2)
                    d_sq = get_sq_dists(pts)
                    d = np.sqrt(d_sq)
                    k = 10.0
                    d_max_val = np.max(d)
                    d_min_val = np.min(d)
                    exp_term_max = np.exp(k * (d - d_max_val))
                    soft_max = d_max_val + (1.0 / k) * np.log(np.sum(exp_term_max))
                    exp_term_min = np.exp(-k * (d - d_min_val))
                    soft_min = d_min_val - (1.0 / k) * np.log(np.sum(exp_term_min))
                    if soft_min < 1e-12:
                        return 1e10
                    ratio = soft_max / soft_min
                    return ratio ** 2
                
                res2 = minimize(
                    direct_objective,
                    best_pts.ravel(),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                final_pts2 = res2.x.reshape(N, 2)
                raw_score2 = score_raw(final_pts2)
                
                if raw_score2 < best_score:
                    best_score = raw_score2
                    best_pts = final_pts2
            except Exception:
                pass
    
    if best_pts is None:
        # Fallback to perfect circle
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_pts

# EVOLVE-BLOCK-END
