"""Cosmology: universal luminosity-distance quadrature.
SEARCH approach: Minimax adaptive sampling. We start with a diverse set of
cosmological models. We evaluate the worst-case error of the current quadrature
rule on this set. We then identify the specific model (and redshift) that
maximizes the error and add it to the training set. We re-optimize the rule
against this growing set of "hardest" models. This focuses optimization effort
on the specific configurations that are currently most difficult, rather than
spending compute on models that are already well- approximated.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

START = time.time()
DEADLINE = 90.0
MARGIN = 6.0
EPS = 1e-8
RNG_SEED = 12345

def _fallback():
    n = 8
    return {"nodes": (np.arange(n) + 0.5) / n, "weights": np.full(n, 1.0 / n)}

def _model_E(a, w0, wa):
    """CPL E(a): Om_m=1, Om_l=0, E=(1-a^3)^(1/3), w=w0+wa(1-a)."""
    # Ensure numerical stability for a in [0, 1]
    a = np.clip(a, 1e-6, 1.0 - 1e-6)
    x = (1.0 - a ** 3) ** (1.0 / 3.0)
    w = w0 + wa * (1.0 - a)
    # Handle case where exponent might cause issues, though for flat models it's usually fine
    # E = (1-a^3)^(1/3) * (1-a^3)^((3(1+w)-1)/3 - 1/3) ... 
    # Actually E^2 = 1/a^2 - 1 = (1-a^3)/a^3 + Omega_L(a)
    # For CPL: Omega_L(a) = Omega_L0 * a^(-3(1+w0+wa)) * exp(3wa(a-1))
    # But the prompt's _model_E uses a specific simplified form:
    # E=(1-a^3)^(1/3) * (1-a^p)^C ... wait, the prompt's _model_E is:
    # x = (1.0 - a ** 3) ** (1.0 / 3.0)
    # w = w0 + wa * (1.0 - a)
    # return x ** ((3.0 * (1.0 + w) - 1.0) / 3.0)
    # This looks like a specific parametric choice, possibly not standard CPL but a "time-varying" proxy.
    # I will stick to the prompt's definition of _model_E to ensure consistency with the "seed" strategy's definition of the space,
    # but I will expand the search range of w0, wa.
    
    # Wait, the prompt's _model_E returns x ** ((3.0 * (1.0 + w) - 1.0) / 3.0).
    # If w=-1, exponent is (0)/3 = 0, E=1. This is flat LCDM.
    # If w varies, E varies.
    
    return x ** ((3.0 * (1.0 + w) - 1.0) / 3.0)

def _model_E2(a, p, C):
    """Generalized time-varying dark energy: E=(1-a^3)^(1/3)*(1-a^p)^C."""
    a = np.clip(a, 1e-6, 1.0 - 1e-6)
    x = (1.0 - a ** 3) ** (1.0 / 3.0)
    return x * (1.0 - a ** p) ** C

def _relerr(nodes, weights, z, Efun):
    """Calculate relative error for a single redshift and model."""
    a = 1.0 / (1.0 + z * nodes)
    E = Efun(a)
    quad = z * np.sum(weights / E)
    
    # Reference: High-order Gauss-Legendre
    # We use 100 points for a very accurate reference
    xi, wi = np.polynomial.legendre.leggauss(100)
    a_ref = 1.0 / (1.0 + z * 0.5 * (xi + 1.0))
    E_ref = Efun(a_ref)
    ref = z * 0.5 * np.sum(wi / E_ref)
    
    if ref == 0:
        return 0.0
    return abs(quad - ref) / ref

def _build_initial_library():
    """Return a diverse initial set of models."""
    lib = []
    # CPL family
    for w0 in (-1.5, -1.0, -0.5, 0.0, 0.5):
        for wa in (-1.0, 0.0, 1.0):
            lib.append({"type": "cpl", "w0": w0, "wa": wa})
    # Generalized
    for p in (1.0, 2.0, 4.0):
        for C in (-1.0, 0.0, 1.0):
            lib.append({"type": "gv", "p": p, "C": C})
    return lib

def _get_Efun(model):
    if model["type"] == "cpl":
        w0, wa = model["w0"], model["wa"]
        return lambda a: _model_E(a, w0, wa)
    elif model["type"] == "gv":
        p, C = model["p"], model["C"]
        return lambda a: _model_E2(a, p, C)
    else:
        raise ValueError("Unknown model type")

def _optimize_rule(nodes0, weights0, hard_set, deadline):
    """Optimize nodes and weights to minimize worst-case error on hard_set."""
    best_err = float('inf')
    best_nodes = nodes0.copy()
    best_weights = weights0.copy()
    
    # Parameterize weights as exp(w) to ensure positivity
    xv0 = np.concatenate([nodes0, np.log(weights0)])
    
    def objective(xv):
        nodes = xv[:8]
        w = np.exp(xv[8:16])
        w = w / np.sum(w)
        
        # Penalty for non-increasing nodes
        d = np.diff(nodes)
        pen = 0.0
        for di in d:
            if di < EPS:
                pen += (EPS - di) ** 2 * 1e6
        
        # Evaluate on hard set
        max_err = 0.0
        for item in hard_set:
            model = item["model"]
            z = item["z"]
            Efun = _get_Efun(model)
            e = _relerr(nodes, w, z, Efun)
            if e > max_err:
                max_err = e
        
        return max_err + pen

    # Nelder-Mead is robust for this small dimension
    res = minimize(objective, xv0, method="Nelder-Mead",
                   options={"maxiter": 500, "xatol": 1e-8, "fatol": 1e-10, "adaptive": True})
    
    if res.x is not None:
        nodes = res.x[:8]
        w = np.exp(res.x[8:16])
        w = w / np.sum(w)
        if np.all(np.diff(nodes) > 1e-6) and np.all(w > 0):
            err = objective(res.x)
            if err < best_err:
                best_err = err
                best_nodes = nodes.copy()
                best_weights = w.copy()
    
    return best_nodes, best_weights, best_err

def construct():
    fallback = _fallback()
    best_nodes = fallback["nodes"].copy()
    best_weights = fallback["weights"].copy()
    
    # Initial library
    library = _build_initial_library()
    
    # Redshifts to test
    zs = [0.1, 0.3, 0.5, 1.0, 2.0, 3.0]
    
    # Initialize hard set with a few random points from the library
    hard_set = []
    for model in library[:5]: # Take a subset initially
        for z in zs[:3]:
            hard_set.append({"model": model, "z": z})
    
    # Iterative Minimax Search
    # We will iterate a few times:
    # 1. Optimize rule on hard_set
    # 2. Evaluate rule on ALL library models and ALL zs
    # 3. Find the (model, z) with max error
    # 4. Add it to hard_set
    # 5. Repeat if time permits
    
    current_nodes = best_nodes
    current_weights = best_weights
    
    for iteration in range(10): # Max 10 iterations
        if time.time() > DEADLINE - MARGIN:
            break
        
        # 1. Optimize
        current_nodes, current_weights, curr_err = _optimize_rule(
            current_nodes, current_weights, hard_set, DEADLINE
        )
        
        # Update best if valid
        if curr_err < float('inf'):
            best_nodes = current_nodes
            best_weights = current_weights
        
        # 2. Evaluate on full library to find the "hardest" case
        max_err_found = -1.0
        hardest_item = None
        
        for model in library:
            Efun = _get_Efun(model)
            for z in zs:
                e = _relerr(current_nodes, current_weights, z, Efun)
                if e > max_err_found:
                    max_err_found = e
                    hardest_item = {"model": model, "z": z}
        
        # 3. Add hardest case to hard_set
        if hardest_item is not None:
            # Check if it's already in hard_set (simple check)
            already_there = False
            for h in hard_set:
                if h["model"] == hardest_item["model"] and h["z"] == hardest_item["z"]:
                    already_there = True
                    break
            if not already_there:
                hard_set.append(hardest_item)
        
        # Optional: If the max error is very small, we can stop early or expand library
        if max_err_found < 1e-4:
            # We have a very good rule, try to refine with more diverse models?
            # For now, just break to save time, as we likely converged.
            break
            
    # Final validation and return
    if np.all(np.diff(best_nodes) > 0) and np.all(best_weights > 0):
        best_weights = best_weights / np.sum(best_weights)
        return {"nodes": best_nodes, "weights": best_weights}
    
    return fallback
# EVOLVE-BLOCK-END
