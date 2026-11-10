"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _Om_of_a(a, Om0, w0, wa):
    """Compute matter density parameter at scale factor a for CPL dark energy."""
    de_factor = np.exp(-3 * ((1 + w0) * np.log(a) + wa * (a - 1)))
    om_m = Om0 * a**(-3)
    om_de = (1 - Om0) * de_factor
    return om_m / (om_m + om_de)

def _f_reference(a, Om0, w0, wa):
    """Reference growth rate using a numerical ODE solve (fast version)."""
    if a >= 1.0:
        return Om0 ** 0.55
    
    # Use a coarse grid for speed
    n_steps = 50
    ln_a = np.linspace(0.0, np.log(a), n_steps + 1)
    f = Om0 ** 0.55
    
    for i in range(n_steps):
        a_i = np.exp(ln_a[i])
        om_i = _Om_of_a(a_i, Om0, w0, wa)
        df_dlna = f - f**2 - 1.5 * om_i
        f += df_dlna * (ln_a[i+1] - ln_a[i])
    
    return f

def _error_for_cosmology(coeffs, Om0, w0, wa, a_vals):
    """Compute max relative error for f and D for one cosmology."""
    c0, c1, c2, c3, c4, c5 = coeffs
    max_err = 0.0
    
    # Compute f and D for all a values
    f_approx_list = []
    f_ref_list = []
    D_approx = 1.0
    D_ref = 1.0
    ln_a_prev = 0.0
    
    for a in a_vals:
        om_a = _Om_of_a(a, Om0, w0, wa)
        if om_a <= 0 or om_a >= 1:
            continue
            
        q = 1 - om_a
        u = 1 + w0 + wa * (1 - a)
        
        gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
        
        if gamma < 0.1 or gamma > 1.2:
            return 1e10
            
        f_approx = om_a ** gamma
        f_ref = _f_reference(a, Om0, w0, wa)
        
        if abs(f_ref) > 1e-10:
            rel_err_f = abs(f_approx - f_ref) / abs(f_ref)
            max_err = max(max_err, rel_err_f)
        
        f_approx_list.append((a, f_approx))
        f_ref_list.append((a, f_ref))
    
    # Compute D(a)/D(1) using trapezoidal rule
    if len(f_approx_list) > 1:
        D_approx = 1.0
        D_ref = 1.0
        for i in range(1, len(f_approx_list)):
            a1, f1 = f_approx_list[i-1]
            a2, f2 = f_approx_list[i]
            a1r, f1r = f_ref_list[i-1]
            a2r, f2r = f_ref_list[i]
            
            # Integrate f(exp(x)) dx from x1 to x2 where x = ln(a)
            # f(exp(x)) = f(a)
            # Use trapezoidal: (f(a1) + f(a2)) / 2 * (ln(a2) - ln(a1))
            if a1 > 0 and a2 > 0:
                da_ln = np.log(a2) - np.log(a1)
                D_approx *= np.exp(-(f1 + f2) / 2 * da_ln)
                D_ref *= np.exp(-(f1r + f2r) / 2 * da_ln)
        
        if abs(D_ref) > 1e-10:
            rel_err_D = abs(D_approx - D_ref) / abs(D_ref)
            max_err = max(max_err, rel_err_D)
    
    return max_err

def _optimize():
    """Optimize coefficients."""
    test_cosmos = [
        (0.30, -1.0, 0.0),
        (0.30, -0.5, 0.0),
        (0.30, -1.0, 0.5),
        (0.30, -1.0, -0.5),
        (0.25, -0.9, 0.3),
        (0.35, -1.1, 0.2),
    ]
    a_vals = np.linspace(0.05, 1.0, 30)
    
    def obj(coeffs):
        max_e = 0.0
        for Om0, w0, wa in test_cosmos:
            e = _error_for_cosmology(coeffs, Om0, w0, wa, a_vals)
            max_e = max(max_e, e)
        return max_e
    
    # Simple Nelder-Mead
    x = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    simplex = [x.copy()]
    for i in range(6):
        xi = x.copy()
        xi[i] += 0.1
        simplex.append(xi)
    
    for _ in range(200):
        vals = [(obj(s), i) for i, s in enumerate(simplex)]
        vals.sort()
        
        # Reflect
        worst = simplex[vals[-1][1]]
        centroid = np.mean([simplex[i] for i, _ in vals[:-1]], axis=0)
        reflected = centroid + 1.0 * (centroid - worst)
        
        if obj(reflected) < vals[0][0]:
            simplex[vals[-1][1]] = reflected
        else:
            # Contraction
            contracted = centroid + 0.5 * (worst - centroid)
            if obj(contracted) < vals[-1][0]:
                simplex[vals[-1][1]] = contracted
            else:
                # Shrink
                for i in range(1, len(simplex)):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
    
    best = min(simplex, key=lambda s: obj(s))
    return np.clip(best, -10, 10)

_t0 = time.time()
try:
    _coeffs = _optimize().tolist()
except Exception:
    _coeffs = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

def construct():
    return _coeffs
# EVOLVE-BLOCK-END
