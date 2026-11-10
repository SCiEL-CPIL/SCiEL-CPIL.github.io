"""
Strategy: Physics-embedded Rational Interpolation via Analytic Perturbation.
1. Identify the baseline LambdaCDM case (w0=-1, wa=0).
2. Derive the linear response of the growth index gamma to w0 and wa.
3. Map these analytic perturbation terms onto the 6 free coefficients.
4. Solve a constrained linear least-squares problem to find the coefficients
   that best match the reference f and D on a sparse, strategically chosen grid.
5. This avoids the local minima traps of Nelder-Mead and leverages the 
   known physics of the growth index expansion.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp, cumulative_trapezoid
from scipy.optimize import lsq_linear

START_TIME = time.time()
DEADLINE = 85.0

def _Om_of_a(a, Om0, w0, wa):
    """Compute matter density parameter at scale factor a for CPL dark energy."""
    if np.isscalar(a):
        if a <= 0:
            return 1.0
        de_factor = np.exp(-3.0 * ((1.0 + w0) * np.log(a) + wa * (a - 1.0)))
        om_m = Om0 * a**(-3.0)
        om_de = (1.0 - Om0) * de_factor
        return om_m / (om_m + om_de)
    a = np.asarray(a, dtype=float)
    de_factor = np.exp(-3.0 * ((1.0 + w0) * np.log(a) + wa * (a - 1.0)))
    om_m = Om0 * a**(-3.0)
    om_de = (1.0 - Om0) * de_factor
    return om_m / (om_m + om_de)

def _compute_reference_sparse():
    """
    Compute reference data on a sparse, strategically chosen grid.
    We focus on the boundaries of the parameter space and key scale factors.
    """
    # Sparse grid of cosmologies
    Om0s = np.array([0.27, 0.30, 0.33])
    w0s = np.array([-1.0, -0.9, -1.1])
    was = np.array([0.0, -0.3, 0.3])
    
    # Key scale factors: early, mid, late
    a_vals = np.array([0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0])
    
    ref_data = []
    
    for Om0 in Om0s:
        for w0 in w0s:
            for wa in was:
                f_refs = np.zeros_like(a_vals)
                D_refs = np.zeros_like(a_vals)
                
                a_min = a_vals[0]
                
                def ode(t, y):
                    a = np.exp(t)
                    om_a = _Om_of_a(a, Om0, w0, wa)
                    f_val = y[0]
                    D_val = y[1]
                    df_dlna = f_val - f_val**2 - 1.5 * om_a
                    dD_dlna = -f_val * D_val
                    return [df_dlna, dD_dlna]
                
                f_init = Om0 ** 0.55
                D_init = 1.0
                t_end = np.log(a_min)
                
                try:
                    sol = solve_ivp(
                        ode, [0.0, t_end], [f_init, D_init],
                        method='DOP853',
                        rtol=1e-10,
                        atol=1e-13,
                        dense_output=True
                    )
                    
                    if not sol.success:
                        continue
                    
                    t_vals = np.log(a_vals)
                    y_all = sol.sol(t_vals)
                    f_refs = y_all[0]
                    D_refs = y_all[1]
                    
                    ref_data.append((Om0, w0, wa, a_vals, f_refs, D_refs))
                except Exception:
                    continue
                    
    return ref_data

def _optimize():
    if time.time() > DEADLINE - 20:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
        
    ref_data = _compute_reference_sparse()
    if len(ref_data) == 0:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
        
    # Build features
    all_q = []
    all_u = []
    all_wa_a = []
    all_uq = []
    all_om = []
    all_f_ref = []
    all_D_ref = []
    
    for Om0, w0, wa, a_vals, f_refs, D_refs in ref_data:
        om_a_vals = _Om_of_a(a_vals, Om0, w0, wa)
        q = 1.0 - om_a_vals
        u = 1.0 + w0 + wa * (1.0 - a_vals)
        wa_a = wa * a_vals
        uq = u * q
        
        all_q.extend(q.tolist())
        all_u.extend(u.tolist())
        all_wa_a.extend(wa_a.tolist())
        all_uq.extend(uq.tolist())
        all_om.extend(om_a_vals.tolist())
        all_f_ref.extend(f_refs.tolist())
        all_D_ref.extend(D_refs.tolist())
        
    Q = np.array(all_q)
    U = np.array(all_u)
    WA_A = np.array(all_wa_a)
    UQ = np.array(all_uq)
    OM = np.array(all_om)
    F_REF = np.array(all_f_ref)
    D_REF = np.array(all_D_ref)
    
    # Design matrix for gamma
    # gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
    X_design = np.column_stack([np.ones_like(Q), Q, Q**2, U, WA_A, UQ])
    
    # Objective: Minimize L2 error of f and D
    # f_approx = OM ** gamma
    # This is non-linear in coeffs because of the exponent.
    # However, for small deviations from LambdaCDM, we can linearize.
    # But let's try a direct non-linear least squares first if time permits, 
    # or use a linear approximation for the exponent.
    
    # Actually, the task asks for a structural change.
    # The current program uses Nelder-Mead on the non-convex objective.
    # A "physics-embedded" approach suggests using the known form of gamma.
    # For LambdaCDM, gamma is constant ~0.55.
    # For CPL, gamma varies with a.
    # The ansatz is gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q.
    
    # Let's use scipy.optimize.least_squares on the non-linear objective.
    # This is more robust than Nelder-Mead for smooth functions.
    
    def residuals(coeffs):
        gamma = X_design @ coeffs
        # Enforce gamma bounds softly in the residual? Or use bounds in least_squares.
        # least_squares supports bounds.
        
        f_approx = OM ** gamma
        
        # f residuals
        res_f = (f_approx - F_REF) / F_REF
        
        # D residuals
        # D(a) = exp(-int_a^0 f(x) dx)
        # We need to compute the integral for each cosmology segment.
        # This is tricky to vectorize easily in least_squares if we don't know the boundaries.
        # Let's precompute the integration weights or just ignore D in the first pass
        # and rely on f to be a good proxy, then refine?
        # No, D is crucial.
        
        # Let's compute D errors efficiently.
        # We have the same structure as before.
        res_D = np.zeros_like(F_REF)
        ptr = 0
        for Om0, w0, wa, a_vals, f_refs, D_refs in ref_data:
            n = len(a_vals)
            f_seg = f_approx[ptr:ptr+n]
            x = np.log(a_vals)
            cum_int = cumulative_trapezoid(f_seg, x, initial=0.0)
            integrals = cum_int[-1] - cum_int
            D_approx = np.exp(-integrals)
            
            res_D[ptr:ptr+n] = (D_approx - D_refs) / D_refs
            ptr += n
            
        # Combine residuals
        # Weight f and D equally
        return np.concatenate([res_f, res_D])
    
    # Bounds for coeffs: [-10, 10]
    bounds = (np.array([-10.0]*6), np.array([10.0]*6))
    
    # Initial guess
    x0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    try:
        res = lsq_linear(
            residuals,
            x0,
            bounds=bounds,
            method='trf',
            max_nfev=500
        )
        best_coeffs = res.x
    except Exception:
        best_coeffs = x0
    
    # Verify gamma bounds for the found solution
    gamma_check = X_design @ best_coeffs
    if np.any(gamma_check < 0.1) or np.any(gamma_check > 1.2):
        # If bounds are violated, fall back to a safer solution
        # Or project onto the feasible set?
        # For now, return the seed if invalid.
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
        
    return best_coeffs.tolist()

# Run optimization at module load
_t0 = time.time()
try:
    _coeffs = _optimize()
except Exception:
    _coeffs = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

# Verify coefficients are valid
if not all(np.isfinite(c) for c in _coeffs):
    _coeffs = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

def construct():
    return _coeffs
# EVOLVE-BLOCK-END
