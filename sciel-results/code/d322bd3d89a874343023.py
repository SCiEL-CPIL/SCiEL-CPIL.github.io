"""
Broadband antireflection coating design via analytic-gradient L-BFGS-B with diverse starts.

Strategy:
- Compute exact analytic gradients of mean reflectance w.r.t. each thickness
  via chain-rule propagation through the characteristic matrix.
- Use L-BFGS-B (from scipy) with those analytic gradients for fast convergence.
- Multi-start from diverse physically-motivated initial designs.
- Polish with a few extra L-BFGS-B restarts and coordinate-descent micro-tuning.
- Select the best valid result within the time budget.
"""
import numpy as np
import time
from scipy.optimize import minimize

def _compute_reflectance_and_grad(wavelengths_nm, indices, thicknesses_nm, ambient_idx, substrate_idx):
    """
    Compute reflectance (fraction, 0-1) at each wavelength and its gradient
    w.r.t. each thickness, using the characteristic matrix method with
    analytic chain-rule differentiation.
    """
    n_layers = len(indices)
    n_wl = len(wavelengths_nm)
    
    if n_layers == 0:
        r0 = (ambient_idx - substrate_idx) / (ambient_idx + substrate_idx)
        return np.full(n_wl, r0 * r0), np.zeros(n_layers)
    
    k = 2.0 * np.pi / wavelengths_nm  # (n_wl,)
    
    # Store per-layer B matrices and their d/dt_i for gradient computation
    # B_i = [[cos(delta_i), 1j*sin(delta_i)/n_i],
    #        [1j*n_i*sin(delta_i), cos(delta_i)]]
    # d B_i / d t_i = [[-k*n_i*sin(delta_i), 1j*k*n_i*cos(delta_i)/n_i],
    #                  [1j*k*n_i^2*cos(delta_i), -k*n_i*sin(delta_i)]]
    
    # Forward pass: compute M = B_1 @ B_2 @ ... @ B_n  (M is 2x2 per wavelength)
    # We'll store the running product from the left: P_i = B_1 @ ... @ B_i
    # and from the right: Q_i = B_{i+1} @ ... @ B_n
    # Then dM/dt_i = Q_i @ (dB_i/dt_i) @ P_{i-1}  (with P_0 = I, Q_{n+1} = I)
    
    # For efficiency, store all B_i and dB_i/dt_i
    B_list = []       # list of (n_wl, 2, 2) arrays
    dB_list = []      # list of (n_wl, 2, 2) arrays
    
    for i in range(n_layers):
        n_i = indices[i]
        d_i = thicknesses_nm[i]
        delta = k * n_i * d_i  # (n_wl,)
        
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        
        # B_i
        B = np.zeros((n_wl, 2, 2), dtype=complex)
        B[:, 0, 0] = cos_d
        B[:, 0, 1] = 1j * sin_d / n_i
        B[:, 1, 0] = 1j * n_i * sin_d
        B[:, 1, 1] = cos_d
        
        # dB_i / d t_i
        # d(delta)/d(t_i) = k * n_i
        dcos_d = -np.sin(delta) * (k * n_i)
        dsin_d = np.cos(delta) * (k * n_i)
        
        dB = np.zeros((n_wl, 2, 2), dtype=complex)
        dB[:, 0, 0] = dcos_d
        dB[:, 0, 1] = 1j * dsin_d / n_i
        dB[:, 1, 0] = 1j * n_i * dsin_d
        dB[:, 1, 1] = dcos_d
        
        B_list.append(B)
        dB_list.append(dB)
    
    # Forward products: P_i = B_0 @ B_1 @ ... @ B_{i-1}  (P_0 = I)
    P = [None] * (n_layers + 1)
    P[0] = np.eye(2, dtype=complex)  # scalar identity, will broadcast
    P[0] = np.broadcast_to(np.eye(2, dtype=complex), (n_wl, 2, 2)).copy()
    
    for i in range(n_layers):
        # P_{i+1} = P_i @ B_i
        P[i+1] = np.einsum('...ab,...bc->...ac', P[i], B_list[i])
    
    # Backward products: Q_i = B_i @ B_{i+1} @ ... @ B_{n-1}  (Q_n = I)
    Q = [None] * (n_layers + 1)
    Q[n_layers] = np.broadcast_to(np.eye(2, dtype=complex), (n_wl, 2, 2)).copy()
    
    for i in range(n_layers - 1, -1, -1):
        # Q_i = B_i @ Q_{i+1}
        Q[i] = np.einsum('...ab,...bc->...ac', B_list[i], Q[i+1])
    
    # M = P_n = Q_0
    M = P[n_layers]  # (n_wl, 2, 2)
    M00 = M[:, 0, 0]
    M01 = M[:, 0, 1]
    M10 = M[:, 1, 0]
    M11 = M[:, 1, 1]
    
    Y_out = substrate_idx
    Y_in = (M10 + M11 * Y_out) / (M00 + M01 * Y_out)
    
    r = (ambient_idx - Y_in) / (ambient_idx + Y_in)
    R = np.abs(r) ** 2
    
    # Gradient of R w.r.t. M elements
    # R = |r|^2, r = (n_a - Y_in)/(n_a + Y_in)
    # Y_in = (M10 + M11*Y_out)/(M00 + M01*Y_out)
    
    # Let's compute dY_in/dM00, dY_in/dM01, dY_in/dM10, dY_in/dM11
    num = M10 + M11 * Y_out
    den = M00 + M01 * Y_out
    
    dYdM00 = -num / (den ** 2)
    dYdM01 = -Y_out * num / (den ** 2)
    dYdM10 = 1.0 / den
    dYdM11 = Y_out / den
    
    # r = (n_a - Y)/(n_a + Y)
    # dr/dY = [-(n_a+Y) - (n_a-Y)] / (n_a+Y)^2 = -2*n_a / (n_a+Y)^2
    drdY = -2.0 * ambient_idx / (ambient_idx + Y_in) ** 2
    
    # R = |r|^2 = r * conj(r)
    # dR/dY = conj(r) * dr/dY + r * conj(dr/dY)
    # But since we're dealing with complex, let's do it properly:
    # R = r * r_conj
    # dR/dY = r_conj * dr/dY + r * d(r_conj)/dY
    # d(r_conj)/dY = conj(dr/dY) if Y is real... but Y is complex.
    # Actually, for complex variables, we need to be careful.
    # Let's use: R = |r|^2, dR = 2*Re(conj(r) * dr)
    # dr = drdY * dY
    # So dR/dY = 2*Re(conj(r) * drdY)
    
    dRdY = 2.0 * np.real(np.conj(r) * drdY)
    
    # Now dR/dM00 = dRdY * dYdM00, etc.
    dRdM00 = dRdY * dYdM00
    dRdM01 = dRdY * dYdM01
    dRdM10 = dRdY * dYdM10
    dRdM11 = dRdY * dYdM11
    
    # Mean reflectance: f = (1/n_wl) * sum(R)
    # df/dt_i = (1/n_wl) * sum_wl dR/dt_i(wl)
    # dR/dt_i = dR/dM @ dM/dt_i
    # dM/dt_i = Q_i @ dB_i/dt_i @ P_{i-1}
    
    grad = np.zeros(n_layers)
    for i in range(n_layers):
        # dM/dt_i = Q_i @ dB_i/dt_i @ P_{i-1}
        # Shape: (n_wl, 2, 2)
        dMdt = np.einsum('...ab,...bc,...cd->...ad', Q[i], dB_list[i], P[i])
        
        # dR/dt_i = dRdM00 * dMdt[:,0,0] + dRdM01 * dMdt[:,0,1] + dRdM10 * dMdt[:,1,0] + dRdM11 * dMdt[:,1,1]
        dRdt = dRdM00 * dMdt[:, 0, 0] + dRdM01 * dMdt[:, 0, 1] + \
               dRdM10 * dMdt[:, 1, 0] + dRdM11 * dMdt[:, 1, 1]
        
        grad[i] = np.mean(dRdt)
    
    return R, grad

def _objective_and_grad(thicknesses, indices, wavelengths_nm, ambient_idx, substrate_idx,
                        max_layer_nm, max_total_nm):
    """
    Objective and gradient. Returns (f, grad). If constraints violated, returns (100, zeros).
    """
    thicknesses = np.asarray(thicknesses, dtype=float)
    if np.any(thicknesses < -1e-6) or np.any(thicknesses > max_layer_nm + 1e-6):
        return 100.0, np.zeros_like(thicknesses)
    if np.sum(thicknesses) > max_total_nm + 1e-6:
        return 100.0, np.zeros_like(thicknesses)
    
    thicknesses_clipped = np.clip(thicknesses, 0.0, max_layer_nm)
    if np.sum(thicknesses_clipped) > max_total_nm:
        return 100.0, np.zeros_like(thicknesses)
    
    R, grad = _compute_reflectance_and_grad(wavelengths_nm, indices, thicknesses_clipped,
                                             ambient_idx, substrate_idx)
    f = np.mean(R) * 100.0
    grad_percent = grad * 100.0
    return f, grad_percent

def _objective_only(thicknesses, indices, wavelengths_nm, ambient_idx, substrate_idx,
                    max_layer_nm, max_total_nm):
    """Objective only (no gradient)."""
    f, _ = _objective_and_grad(thicknesses, indices, wavelengths_nm, ambient_idx, substrate_idx,
                                max_layer_nm, max_total_nm)
    return f

def _lbfgsb_optimize(f, df, x0, bounds, max_iter=500, time_deadline=None):
    """
    L-BFGS-B optimization using scipy.
    """
    n = len(x0)
    lo_bounds = np.array([b[0] for b in bounds])
    hi_bounds = np.array([b[1] for b in bounds])
    x0 = np.clip(np.asarray(x0, dtype=float), lo_bounds, hi_bounds)
    
    # Check time
    if time_deadline is not None and time.time() > time_deadline:
        return x0, f(x0)
    
    result = minimize(
        f, x0, jac=df, method='L-BFGS-B',
        bounds=bounds, options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    
    # Check time
    if time_deadline is not None and time.time() > time_deadline:
        return x0, f(x0)
    
    if result.success or result.fun < f(x0):
        x_opt = np.clip(result.x, lo_bounds, hi_bounds)
        f_opt = f(x_opt)
        return x_opt, f_opt
    else:
        return x0, f(x0)

def _coordinate_descent(f, x0, bounds, max_iter=50, time_deadline=None):
    """
    Coordinate descent for final refinement.
    """
    n = len(x0)
    lo_bounds = np.array([b[0] for b in bounds])
    hi_bounds = np.array([b[1] for b in bounds])
    x = np.clip(np.asarray(x0, dtype=float), lo_bounds, hi_bounds)
    fval = f(x)
    
    for it in range(max_iter):
        if time_deadline is not None and time.time() > time_deadline:
            break
        
        improved = False
        for i in range(n):
            best_xi = x[i]
            best_fi = fval
            
            for delta in [-0.1, -0.05, -0.01, 0.01, 0.05, 0.1]:
                xi_new = x[i] + delta
                if xi_new < lo_bounds[i] or xi_new > hi_bounds[i]:
                    continue
                x_trial = x.copy()
                x_trial[i] = xi_new
                if np.sum(x_trial) > bounds[i][1] * n:  # rough check
                    continue
                f_trial = f(x_trial)
                if f_trial < best_fi:
                    best_fi = f_trial
                    best_xi = xi_new
            
            if best_xi != x[i]:
                x[i] = best_xi
                fval = best_fi
                improved = True
        
        if not improved:
            break
    
    return x, fval

def _generate_initial_designs(indices, n_layers, center_wl, max_total_nm, max_layer_nm, n_starts=20):
    """
    Generate diverse, physically-motivated initial designs.
    """
    designs = []
    
    # Quarter-wave stack
    qw = np.array([center_wl / (4.0 * n_i) for n_i in indices])
    if np.sum(qw) <= max_total_nm:
        designs.append(qw.copy())
    else:
        designs.append(qw * (max_total_nm / np.sum(qw)))
    
    # Half-wave stack
    hw = np.array([center_wl / (2.0 * n_i) for n_i in indices])
    if np.sum(hw) <= max_total_nm:
        designs.append(hw.copy())
    else:
        designs.append(hw * (max_total_nm / np.sum(hw)))
    
    # Various fractions of quarter-wave
    for frac in [0.5, 0.75, 1.25, 1.5]:
        trial = np.array([frac * center_wl / (4.0 * n_i) for n_i in indices])
        if np.sum(trial) <= max_total_nm:
            designs.append(trial)
        else:
            designs.append(trial * (max_total_nm / np.sum(trial)))
    
    # Random designs with physical constraints
    rng = np.random.RandomState(42)
    for _ in range(n_starts - len(designs)):
        trial = rng.uniform(0.1, 1.0, size=n_layers)
        trial = trial * (center_wl / 4.0) / np.array(indices)
        if np.sum(trial) > max_total_nm:
            trial = trial * (max_total_nm / np.sum(trial))
        trial = np.clip(trial, 0, max_layer_nm)
        designs.append(trial)
    
    valid_designs = []
    for d in designs:
        d = np.clip(d, 0, max_layer_nm)
        if np.sum(d) <= max_total_nm:
            valid_designs.append(d)
    
    if not valid_designs:
        valid_designs.append(np.array([center_wl / (4.0 * indices[0])]))
    
    return valid_designs

def design(problem):
    """
    Design a broadband antireflection coating.
    Returns: (labels, thicknesses)
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    ambient_idx = problem["ambient_index"]
    substrate_idx = problem["substrate_index"]
    materials = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    wavelengths_nm = np.linspace(band_nm[0], band_nm[1], 401)
    center_wl = 0.5 * (band_nm[0] + band_nm[1])
    
    best_labels = None
    best_thicknesses = None
    best_score = np.inf
    
    # Try different layer counts and patterns
    for n_layers in [7, 9, 11, 13]:
        if n_layers > max_layers or time.time() > deadline - 30:
            continue
        
        labels_1 = ['L' if i % 2 == 0 else 'H' for i in range(n_layers)]
        labels_2 = ['H' if i % 2 == 0 else 'L' for i in range(n_layers)]
        
        for labels in [labels_1, labels_2]:
            if time.time() > deadline - 25:
                break
            
            indices = [materials[label] for label in labels]
            bounds = [(0.0, float(max_layer_nm))] * n_layers
            
            def make_f(indices):
                def f(x):
                    return _objective_only(x, indices, wavelengths_nm, ambient_idx, substrate_idx,
                                           max_layer_nm, max_total_nm)
                return f
            
            def make_df(indices):
                def df(x):
                    _, g = _objective_and_grad(x, indices, wavelengths_nm, ambient_idx, substrate_idx,
                                                max_layer_nm, max_total_nm)
                    return g
                return df
            
            f = make_f(indices)
            df = make_df(indices)
            
            initial_designs = _generate_initial_designs(indices, n_layers, center_wl,
                                                        max_total_nm, max_layer_nm, n_starts=20)
            
            best_x = None
            best_f = np.inf
            
            for x0 in initial_designs:
                if time.time() > deadline - 20:
                    break
                
                x0 = np.clip(x0, 0, max_layer_nm)
                if np.sum(x0) > max_total_nm:
                    x0 = x0 * (max_total_nm / np.sum(x0))
                
                lbfgsb_deadline = min(time.time() + 8.0, deadline - 15)
                try:
                    x_opt, f_opt = _lbfgsb_optimize(f, df, x0, bounds, max_iter=300,
                                                     time_deadline=lbfgsb_deadline)
                    if f_opt < best_f:
                        best_f = f_opt
                        best_x = x_opt.copy()
                except Exception:
                    pass
            
            if best_x is not None and best_f < best_score:
                best_score = best_f
                best_labels = labels.copy()
                best_thicknesses = best_x.copy()
                
                # Polish with coordinate descent
                if time.time() < deadline - 10:
                    cd_deadline = min(time.time() + 5.0, deadline - 5)
                    try:
                        x_cd, f_cd = _coordinate_descent(f, best_x, bounds, max_iter=30,
                                                         time_deadline=cd_deadline)
                        if f_cd < best_score:
                            best_score = f_cd
                            best_thicknesses = x_cd.copy()
                    except Exception:
                        pass
    
    if best_labels is None:
        return ["L"], [center_wl / (4.0 * materials["L"])]
    
    thicknesses = np.array(best_thicknesses)
    thicknesses = np.clip(thicknesses, 0, max_layer_nm)
    if np.sum(thicknesses) > max_total_nm:
        thicknesses = thicknesses * (max_total_nm / np.sum(thicknesses))
    
    return best_labels, thicknesses.tolist()
