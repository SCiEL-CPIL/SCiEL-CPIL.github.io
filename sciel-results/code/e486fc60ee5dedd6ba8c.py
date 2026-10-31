import numpy as np
import time
from scipy.optimize import minimize, root

_DEADLINE = time.time() + 195

# Precompute I[n] = integral_{-inf}^{inf} x^{2n} exp(-x^2) dx
I = np.zeros(25)
I[0] = np.sqrt(np.pi)
for n in range(1, 25):
    I[n] = I[n-1] * (2.0*n - 1.0) / (2.0*n)

# Coefficients for Hermite polynomials in power basis x^{2k}
H0_c = np.array([1, 0, 0, 0, 0, 0, 0])
H4_c = np.array([3, -6, 1, 0, 0, 0, 0])
H8_c = np.array([105, -420, 210, -28, 1, 0, 0])
H12_c = np.array([945, -7560, 13860, -6930, 1155, -66, 1])

c12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])

col0 = H0_c + c12[0]*H12_c
col1 = H4_c + c12[1]*H12_c
col2 = H8_c + c12[2]*H12_c

A_mat = np.column_stack((col0, col1, col2)) # Shape (7, 3)

def _get_p(a):
    return A_mat @ a

def _compute_bound_and_grad(a):
    p = _get_p(a)
    q = p * np.array([1, -1, 1, -1, 1, -1, 1])
    
    Mx0 = 0.0
    Mx1 = 0.0
    Mxi0 = 0.0
    Mxi1 = 0.0
    
    # Vectorized moment calculation
    # Mx0 = p @ M0 @ p
    # Mx1 = p @ M1 @ p
    # Where M0[i,j] = I[i+j], M1[i,j] = I[i+j+1]
    M0 = np.zeros((7,7))
    M1 = np.zeros((7,7))
    for i in range(7):
        for j in range(7):
            M0[i,j] = I[i+j]
            M1[i,j] = I[i+j+1]
            
    Mx0 = p @ M0 @ p
    Mx1 = p @ M1 @ p
    Mxi0 = q @ M0 @ q
    Mxi1 = q @ M1 @ q
    
    if Mx0 <= 1e-14 or Mxi0 <= 1e-14:
        return 1e18, np.zeros(3)
        
    vx = Mx1 / Mx0
    vxi = Mxi1 / Mxi0
    
    if vx <= 1e-14 or vxi <= 1e-14:
        return 1e18, np.zeros(3)
        
    prod = vx * vxi
    if prod <= 0:
        return 1e18, np.zeros(3)
        
    bound = np.sqrt(prod)
    
    if not np.isfinite(bound):
        return 1e18, np.zeros(3)
        
    # Gradient
    # dMx0 = 2 * M0 @ p
    # dMx1 = 2 * M1 @ p
    # dMxi0 = 2 * M0 @ q
    # dMxi1 = 2 * M1 @ q
    
    dMx0 = 2.0 * (M0 @ p)
    dMx1 = 2.0 * (M1 @ p)
    dMxi0 = 2.0 * (M0 @ q)
    dMxi1 = 2.0 * (M1 @ q)
    
    # d(p_k) / d(a_m) = A_mat[k, m]
    # d(Mx0)/d(a_m) = sum_k dMx0_k * A_mat[k, m] = dMx0 @ A_mat[:, m]
    # Vectorized: dMx0_da = A_mat.T @ dMx0
    
    dMx0_da = A_mat.T @ dMx0
    dMx1_da = A_mat.T @ dMx1
    dMxi0_da = A_mat.T @ dMxi0
    dMxi1_da = A_mat.T @ dMxi1
    
    inv_Mx0_sq = 1.0 / (Mx0 * Mx0)
    inv_Mxi0_sq = 1.0 / (Mxi0 * Mxi0)
    
    dvx = (dMx1_da * Mx0 - Mx1 * dMx0_da) * inv_Mx0_sq
    dvxi = (dMxi1_da * Mxi0 - Mxi1 * dMxi0_da) * inv_Mxi0_sq
        
    factor = 0.5 / bound
    grad = factor * (vxi * dvx + vx * dvxi)
    
    if not np.all(np.isfinite(grad)):
        return bound, np.zeros(3)
        
    return bound, grad

def _sphere_obj(params):
    """
    Objective function in spherical coordinates (theta, phi).
    theta in [0, pi], phi in [0, 2*pi]
    """
    theta, phi = params
    a0 = np.sin(theta) * np.cos(phi)
    a4 = np.sin(theta) * np.sin(phi)
    a8 = np.cos(theta)
    a = np.array([a0, a4, a8])
    val, _ = _compute_bound_and_grad(a)
    return val

def _sphere_grad(params):
    """
    Gradient of the objective in spherical coordinates.
    """
    theta, phi = params
    a0 = np.sin(theta) * np.cos(phi)
    a4 = np.sin(theta) * np.sin(phi)
    a8 = np.cos(theta)
    a = np.array([a0, a4, a8])
    
    val, grad_a = _compute_bound_and_grad(a)
    
    # Chain rule
    # da/dtheta = [cos(theta)cos(phi), cos(theta)sin(phi), -sin(theta)]
    # da/dphi = [-sin(theta)sin(phi), sin(theta)cos(phi), 0]
    
    da_dtheta = np.array([
        np.cos(theta) * np.cos(phi),
        np.cos(theta) * np.sin(phi),
        -np.sin(theta)
    ])
    
    da_dphi = np.array([
        -np.sin(theta) * np.sin(phi),
        np.sin(theta) * np.cos(phi),
        0.0
    ])
    
    grad_theta = np.dot(grad_a, da_dtheta)
    grad_phi = np.dot(grad_a, da_dphi)
    
    return np.array([grad_theta, grad_phi])

def construct():
    """
    Find three coefficients (a0, a4, a8) that minimize the uncertainty product.
    Returns array of shape (3,).
    """
    
    # Seed value
    seed_params = np.array([-1.13, 0.04, 0.000308642])
    seed_val, _ = _compute_bound_and_grad(seed_params)
    
    best_val = seed_val
    best_params = seed_params.copy()
    
    deadline_margin = 10  # seconds
    
    # Multi-start optimization on the sphere
    rng = np.random.default_rng(42)
    
    # Generate starts: uniform on sphere
    num_starts = 200
    starts = []
    for _ in range(num_starts):
        # Uniform on sphere
        z = rng.normal()
        x = rng.normal()
        y = rng.normal()
        norm = np.sqrt(x*x + y*y + z*z)
        if norm < 1e-9: continue
        u = x/norm
        v = y/norm
        w = z/norm
        # Convert to spherical
        theta = np.arccos(np.clip(w, -1.0, 1.0))
        phi = np.arctan2(v, u)
        if phi < 0: phi += 2*np.pi
        starts.append([theta, phi])
        
    # Also include seed converted to sphere
    norm_seed = np.linalg.norm(seed_params)
    if norm_seed > 1e-9:
        u_s = seed_params[0]/norm_seed
        v_s = seed_params[1]/norm_seed
        w_s = seed_params[2]/norm_seed
        theta_s = np.arccos(np.clip(w_s, -1.0, 1.0))
        phi_s = np.arctan2(v_s, u_s)
        if phi_s < 0: phi_s += 2*np.pi
        starts.append([theta_s, phi_s])
        
    candidates = []
    
    for start in starts:
        if time.time() > _DEADLINE - deadline_margin:
            break
        try:
            res = minimize(_sphere_obj, start, jac=_sphere_grad, method='L-BFGS-B', 
                           bounds=[(0, np.pi), (0, 2*np.pi)], 
                           options={'maxiter': 100, 'ftol': 1e-12})
            if res.success:
                theta, phi = res.x
                a0 = np.sin(theta) * np.cos(phi)
                a4 = np.sin(theta) * np.sin(phi)
                a8 = np.cos(theta)
                a_sol = np.array([a0, a4, a8])
                
                # Polish in Cartesian coordinates with SLSQP constraint ||a||=1
                # This ensures high precision on the constraint
                def cart_obj(a):
                    return _compute_bound_and_grad(a)[0]
                
                def cart_grad(a):
                    return _compute_bound_and_grad(a)[1]
                
                cons = [{'type': 'eq', 'fun': lambda a: np.sum(a*a) - 1.0, 'jac': lambda a: 2.0*a}]
                
                res2 = minimize(cart_obj, a_sol, jac=cart_grad, method='SLSQP', 
                                constraints=cons, options={'maxiter': 100, 'ftol': 1e-14})
                
                if res2.success and res2.fun < best_val:
                    best_val = res2.fun
                    best_params = res2.x
                    candidates.append(res2.x.copy())
                    
        except Exception:
            pass

    # Final high-precision polish on the best candidate
    if time.time() < _DEADLINE - deadline_margin and np.all(np.isfinite(best_params)):
        try:
            # One more SLSQP pass with tighter tolerance
            def cart_obj(a):
                return _compute_bound_and_grad(a)[0]
            def cart_grad(a):
                return _compute_bound_and_grad(a)[1]
            cons = [{'type': 'eq', 'fun': lambda a: np.sum(a*a) - 1.0, 'jac': lambda a: 2.0*a}]
            
            res_final = minimize(cart_obj, best_params, jac=cart_grad, method='SLSQP', 
                                 constraints=cons, options={'maxiter': 200, 'ftol': 1e-16})
            if res_final.success and res_final.fun < best_val:
                best_val = res_final.fun
                best_params = res_final.x
        except Exception:
            pass

    if not np.all(np.isfinite(best_params)):
        return seed_params

    return np.array(best_params)
