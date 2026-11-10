<<<<<<< SEARCH
import numpy as np
from scipy.special import lpmv, factorial
from scipy.optimize import minimize
import time

def rudin_problem10() -> np.ndarray:
    """
    Find coefficients for degree-10 spherical harmonic minimizing L4/L2 ratio.
    Strategy:
    1. Precompute basis matrix A on the specified quadrature grid.
    2. Define objective R(c) = ||f||_4 / ||f||_2 and its analytic gradient.
    3. Generate a large, diverse set of initial candidates using symmetry-aware
       constructions and random sampling.
    4. Polish each candidate with L-BFGS-B using the analytic gradient.
    5. Return the best valid coefficient vector found within the time budget.
    """
    start_time = time.time()
    deadline = start_time + 570  # Safety margin

    k = 10
    n_m = 2 * k + 1  # 21 coefficients

    # 1. Setup Quadrature Grid
    # 70-point Gauss-Legendre in mu = cos(theta)
    mu, w = np.polynomial.legendre.leggauss(70)
    # 130 equally spaced phi
    n_phi = 130
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    d_phi = 2 * np.pi / n_phi

    # Flatten grid
    mu_flat, phi_flat = np.meshgrid(mu, phi, indexing='ij')
    mu_flat = mu_flat.flatten()
    phi_flat = phi_flat.flatten()
    w_flat = np.repeat(w, n_phi)
    
    n_points = len(mu_flat)

    # 2. Construct Basis Functions
    A = np.zeros((n_points, n_m))
    
    def get_norm(m):
        m_abs = abs(m)
        val = (21.0 / (4.0 * np.pi)) * (factorial(10 - m_abs) / factorial(10 + m_abs))
        return np.sqrt(val)
    
    # m = 0
    m = 0
    idx = 10 
    P = lpmv(0, 10, mu_flat)
    N = get_norm(0)
    A[:, idx] = N * P
    
    # m > 0
    for m in range(1, k + 1):
        idx_pos = m + 10
        idx_neg = -m + 10
        P = lpmv(m, 10, mu_flat)
        N = get_norm(m)
        sqrt2_N = np.sqrt(2.0) * N
        A[:, idx_pos] = sqrt2_N * P * np.cos(m * phi_flat)
        A[:, idx_neg] = sqrt2_N * P * np.sin(m * phi_flat)
        
    # 3. Objective and Gradient
    def objective_and_grad(c):
        norm_c = np.linalg.norm(c)
        if norm_c < 1e-12:
            return 1e10, np.zeros(n_m)
        u = c / norm_c
        f_vals = A @ u
        f2 = f_vals ** 2
        f4 = f2 ** 2
        norm2_sq = np.sum(w_flat * f2) * d_phi
        norm4_4 = np.sum(w_flat * f4) * d_phi
        
        if norm2_sq < 1e-12:
            return 1e10, np.zeros(n_m)
            
        N2 = np.sqrt(norm2_sq)
        N4 = norm4_4 ** 0.25
        R = N4 / N2
        
        # Analytic gradient of R with respect to u
        sum_wf_A = A.T @ (w_flat * f_vals) * d_phi
        sum_wf3_A = A.T @ (w_flat * f_vals**3) * d_phi
        
        grad_u = (N4**3) * sum_wf3_A - (N4 / N2**3) * sum_wf_A
        
        # Chain rule for c -> u = c/||c||
        u_dot_grad = np.dot(u, grad_u)
        grad_c = (grad_u - u * u_dot_grad) / norm_c
        
        return R, grad_c

    # 4. Generate Initial Candidates
    candidates = []
    
    # Candidate 1: Pure m=0 (Axisymmetric)
    c0 = np.zeros(n_m)
    c0[10] = 1.0
    candidates.append(c0)
    
    # Candidates 2-21: Pure m>0 modes (cos and sin parts)
    for m in range(1, 11):
        c_cos = np.zeros(n_m)
        c_cos[m + 10] = 1.0
        candidates.append(c_cos)
        
        c_sin = np.zeros(n_m)
        c_sin[-m + 10] = 1.0
        candidates.append(c_sin)
        
    # Candidates 22-41: Pairs of m modes (sum of two different m)
    for m1 in range(0, 11):
        for m2 in range(m1 + 1, 11):
            if time.time() > deadline - 60:
                break
            c_pair = np.zeros(n_m)
            c_pair[10 + m1] = 1.0
            c_pair[10 + m2] = 1.0
            c_pair /= np.linalg.norm(c_pair)
            candidates.append(c_pair)
            
    # Candidates 42+: Random normalized vectors with different seeds
    rng = np.random.default_rng(12345)
    n_random = 50
    for _ in range(n_random):
        if time.time() > deadline - 60:
            break
        c_rand = rng.standard_normal(n_m)
        c_rand /= np.linalg.norm(c_rand)
        candidates.append(c_rand)

    # 5. Optimize Each Candidate
    best_c = None
    best_val = np.inf
    
    # Sort candidates by initial objective value to prioritize good starts
    candidate_vals = []
    for c_init in candidates:
        if time.time() > deadline - 60:
            break
        val, _ = objective_and_grad(c_init)
        candidate_vals.append((val, c_init))
    
    candidate_vals.sort(key=lambda x: x[0])
    
    for val_init, c_init in candidate_vals:
        if time.time() > deadline - 5:
            break
            
        try:
            res = minimize(
                objective_and_grad, 
                c_init, 
                jac=True, 
                method='L-BFGS-B', 
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            val = res.fun
            if val < best_val and np.all(np.isfinite(res.x)):
                best_val = val
                best_c = res.x.copy()
        except Exception:
            # If optimization fails, check the initial point itself
            if val_init < best_val:
                best_val = val_init
                best_c = c_init.copy()

    # 6. Final Validity Check and Scaling
    if best_c is None or not np.all(np.isfinite(best_c)):
        # Fallback to m=0
        best_c = np.zeros(n_m)
        best_c[10] = 1.0
        
    max_abs = np.max(np.abs(best_c))
    if max_abs < 1e-3 or max_abs > 1e5:
        if max_abs > 1e5:
            best_c = best_c * (1e4 / max_abs)
        elif max_abs < 1e-3:
            best_c = best_c * (1e-2 / max_abs)
            
    return best_c
=======
import numpy as np
from scipy.special import lpmv, factorial
from scipy.optimize import minimize
import time

def rudin_problem10() -> np.ndarray:
    """
    Find coefficients for degree-10 spherical harmonic minimizing L4/L2 ratio.
    Strategy:
    1. Precompute basis matrix A and weighted basis B on the quadrature grid.
    2. Define objective R(c) = ||f||_4 / ||f||_2 with precomputed gradient terms.
    3. Generate diverse initial candidates (pure modes, pairs, triplets, random).
    4. Polish each candidate with L-BFGS-B then sphere-constrained GD.
    5. Return the best valid coefficient vector found within the time budget.
    """
    start_time = time.time()
    deadline = start_time + 570  # Safety margin

    k = 10
    n_m = 2 * k + 1  # 21 coefficients

    # 1. Setup Quadrature Grid
    mu, w = np.polynomial.legendre.leggauss(70)
    n_phi = 130
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    d_phi = 2 * np.pi / n_phi

    mu_flat, phi_flat = np.meshgrid(mu, phi, indexing='ij')
    mu_flat = mu_flat.flatten()
    phi_flat = phi_flat.flatten()
    w_flat = np.repeat(w, n_phi)
    
    n_points = len(mu_flat)

    # 2. Construct Basis Functions and precompute weighted versions
    A = np.zeros((n_points, n_m))
    Aw = np.zeros((n_points, n_m))  # A with weights pre-applied
    
    def get_norm(m):
        m_abs = abs(m)
        val = (21.0 / (4.0 * np.pi)) * (factorial(10 - m_abs) / factorial(10 + m_abs))
        return np.sqrt(val)
    
    m = 0
    idx = 10 
    P = lpmv(0, 10, mu_flat)
    N = get_norm(0)
    A[:, idx] = N * P
    Aw[:, idx] = w_flat * A[:, idx] * d_phi
    
    for m in range(1, k + 1):
        idx_pos = m + 10
        idx_neg = -m + 10
        P = lpmv(m, 10, mu_flat)
        N = get_norm(m)
        sqrt2_N = np.sqrt(2.0) * N
        A[:, idx_pos] = sqrt2_N * P * np.cos(m * phi_flat)
        A[:, idx_neg] = sqrt2_N * P * np.sin(m * phi_flat)
        Aw[:, idx_pos] = w_flat * A[:, idx_pos] * d_phi
        Aw[:, idx_neg] = w_flat * A[:, idx_neg] * d_phi
        
    # 3. Objective and Gradient using precomputed weighted basis
    def objective_and_grad(c):
        norm_c = np.linalg.norm(c)
        if norm_c < 1e-12:
            return 1e10, np.zeros(n_m)
        u = c / norm_c
        f_vals = A @ u
        f2 = f_vals ** 2
        f4 = f2 ** 2
        norm2_sq = np.sum(w_flat * f2) * d_phi
        norm4_4 = np.sum(w_flat * f4) * d_phi
        
        if norm2_sq < 1e-12:
            return 1e10, np.zeros(n_m)
            
        N2 = np.sqrt(norm2_sq)
        N4 = norm4_4 ** 0.25
        R = N4 / N2
        
        # Precompute f_vals**3 once
        f3 = f2 * f_vals
        sum_wf_A = Aw @ u * d_phi  # Wait, Aw already has w*d_phi
        # Actually Aw already includes w_flat * d_phi, so:
        sum_wf_A = Aw @ u
        sum_wf3_A = A.T @ (w_flat * f3) * d_phi
        
        grad_u = (N4**3) * sum_wf3_A - (N4 / N2**3) * sum_wf_A
        
        u_dot_grad = np.dot(u, grad_u)
        grad_c = (grad_u - u * u_dot_grad) / norm_c
        
        return R, grad_c

    def objective_only(c):
        norm_c = np.linalg.norm(c)
        if norm_c < 1e-12:
            return 1e10
        u = c / norm_c
        f_vals = A @ u
        f2 = f_vals ** 2
        f4 = f2 ** 2
        norm2_sq = np.sum(w_flat * f2) * d_phi
        norm4_4 = np.sum(w_flat * f4) * d_phi
        if norm2_sq < 1e-12:
            return 1e10
        N2 = np.sqrt(norm2_sq)
        N4 = norm4_4 ** 0.25
        return N4 / N2

    # Sphere-constrained gradient descent polish
    def sphere_gd_polish(c_init, max_iter=500, tol=1e-14):
        """Polish using gradient descent on the sphere."""
        c = c_init.copy()
        norm_c = np.linalg.norm(c)
        if norm_c < 1e-12:
            return c
        u = c / norm_c
        
        best_u = u.copy()
        best_val = objective_only(u)
        
        lr = 0.01
        for i in range(max_iter):
            if time.time() > deadline - 10:
                break
            f_vals = A @ u
            f2 = f_vals ** 2
            f4 = f2 ** 2
            norm2_sq = np.sum(w_flat * f2) * d_phi
            norm4_4 = np.sum(w_flat * f4) * d_phi
            
            if norm2_sq < 1e-12:
                break
                
            N2 = np.sqrt(norm2_sq)
            N4 = norm4_4 ** 0.25
            
            f3 = f2 * f_vals
            sum_wf_A = Aw @ u
            sum_wf3_A = A.T @ (w_flat * f3) * d_phi
            
            grad_u = (N4**3) * sum_wf3_A - (N4 / N2**3) * sum_wf_A
            
            # Project gradient to tangent space of sphere
            u_dot_grad = np.dot(u, grad_u)
            grad_tangent = grad_u - u * u_dot_grad
            
            # Backtracking line search
            best_step_val = best_val
            best_step_u = u.copy()
            step = lr
            for ls in range(20):
                u_new = u - step * grad_tangent
                u_new_norm = np.linalg.norm(u_new)
                if u_new_norm < 1e-12:
                    break
                u_new = u_new / u_new_norm
                val_new = objective_only(u_new)
                if val_new < best_step_val:
                    best_step_val = val_new
                    best_step_u = u_new.copy()
                    break
                step *= 0.5
            
            if np.linalg.norm(best_step_u - u) < tol:
                break
            u = best_step_u
            if best_step_val < best_val:
                best_val = best_step_val
                best_u = u.copy()
        
        return best_u

    # 4. Generate Initial Candidates
    candidates = []
    
    # Pure m=0
    c0 = np.zeros(n_m)
    c0[10] = 1.0
    candidates.append(c0)
    
    # Pure m>0 modes
    for m in range(1, 11):
        c_cos = np.zeros(n_m)
        c_cos[m + 10] = 1.0
        candidates.append(c_cos)
        
        c_sin = np.zeros(n_m)
        c_sin[-m + 10] = 1.0
        candidates.append(c_sin)
        
    # Pairs of m modes
    for m1 in range(0, 11):
        for m2 in range(m1 + 1, 11):
            if time.time() > deadline - 60:
                break
            c_pair = np.zeros(n_m)
            c_pair[10 + m1] = 1.0
            c_pair[10 + m2] = 1.0
            c_pair /= np.linalg.norm(c_pair)
            candidates.append(c_pair)
            
    # Triplets of m modes (low-m biased)
    for m1 in range(0, 8):
        for m2 in range(m1 + 1, 9):
            for m3 in range(m2 + 1, 10):
                if time.time() > deadline - 60:
                    break
                c_trip = np.zeros(n_m)
                c_trip[10 + m1] = 1.0
                c_trip[10 + m2] = 1.0
                c_trip[10 + m3] = 1.0
                c_trip /= np.linalg.norm(c_trip)
                candidates.append(c_trip)
            
    # Random normalized vectors
    rng = np.random.default_rng(12345)
    n_random = 60
    for _ in range(n_random):
        if time.time() > deadline - 60:
            break
        c_rand = rng.standard_normal(n_m)
        c_rand /= np.linalg.norm(c_rand)
        candidates.append(c_rand)
    
    # Low-m biased random: emphasize lower m components
    for _ in range(30):
        if time.time() > deadline - 60:
            break
        c_rand = rng.standard_normal(n_m)
        # Bias toward low m
        for i in range(n_m):
            m_val = i - 10
            c_rand[i] *= np.exp(-0.1 * abs(m_val))
        c_rand /= np.linalg.norm(c_rand)
        candidates.append(c_rand)

    # 5. Optimize Each Candidate
    best_c = None
    best_val = np.inf
    
    candidate_vals = []
    for c_init in candidates:
        if time.time() > deadline - 30:
            break
        val, _ = objective_and_grad(c_init)
        candidate_vals.append((val, c_init))
    
    candidate_vals.sort(key=lambda x: x[0])
    
    for val_init, c_init in candidate_vals:
        if time.time() > deadline - 15:
            break
            
        try:
            res = minimize(
                objective_and_grad, 
                c_init, 
                jac=True, 
                method='L-BFGS-B', 
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            val = res.fun
            if val < best_val and np.all(np.isfinite(res.x)):
                best_val = val
                best_c = res.x.copy()
                
            # Sphere-constrained GD polish
            if np.all(np.isfinite(res.x)):
                polished = sphere_gd_polish(res.x, max_iter=300)
                polished_val = objective_only(polished)
                if polished_val < best_val:
                    best_val = polished_val
                    best_c = polished.copy()
        except Exception:
            if val_init < best_val:
                best_val = val_init
                best_c = c_init.copy()

    # 6. Final Validity Check and Scaling
    if best_c is None or not np.all(np.isfinite(best_c)):
        best_c = np.zeros(n_m)
        best_c[10] = 1.0
        
    max_abs = np.max(np.abs(best_c))
    if max_abs < 1e-3 or max_abs > 1e5:
        if max_abs > 1e5:
            best_c = best_c * (1e4 / max_abs)
        elif max_abs < 1e-3:
            best_c = best_c * (1e-2 / max_abs)
            
    return best_c
>>>>>>> REPLACE

I need to fix the `sum_wf_A` calculation since `Aw` already includes `w_flat * d_phi`:

