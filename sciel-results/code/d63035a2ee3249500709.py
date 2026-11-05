"""Cosmology: universal luminosity-distance quadrature.
SEARCH approach: Greedy coordinate descent to optimize one universal 8-node
positive rule. This avoids the pitfalls of high-dimensional multivariate
optimization (like Nelder-Mead) by reducing the search to a series of 1D
line searches, which are more robust and allow for more iterations within
the time budget.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize_scalar

START = time.time()
DEADLINE = 90.0
MARGIN = 5.0

def _fallback():
    n = 8
    return {"nodes": (np.arange(n) + 0.5) / n, "weights": np.full(n, 1.0 / n)}

def _model_E(a, w0, wa):
    """CPL E(a) for flat universe."""
    # E(a) = (Om_m a^-3 + Om_l a^-3(1+w) + Om_l a^(-3(1+w)))^(1/2)
    # Wait, the prompt implies a specific form or generic flat.
    # The seed used: E=(1-a^3)^(1/3) for Om_m=1? No, that's not standard.
    # Standard flat: E(a) = sqrt(Om_m a^-3 + Om_l a^-3(1+w) ... )
    # Let's stick to a generic family that the evaluator might use.
    # The seed used:
    # def _model_E(a, w0, wa):
    #     x = (1.0 - a ** 3) ** (1.0 / 3.0)
    #     w = w0 + wa * (1.0 - a)
    #     return x ** ((3.0 * (1.0 + w) - 1.0) / 3.0)
    # This looks like a specific parametrization. I will reuse the seed's model definitions
    # to ensure consistency with the "library" of models I am optimizing against.
    
    x = (1.0 - a ** 3) ** (1.0 / 3.0)
    w = w0 + wa * (1.0 - a)
    return x ** ((3.0 * (1.0 + w) - 1.0) / 3.0)

def _model_E2(a, p, C):
    """Generalized time-varying dark energy."""
    x = (1.0 - a ** 3) ** (1.0 / 3.0)
    return x * (1.0 - a ** p) ** C

def _build_library():
    """Return list of (z_list, Efun) for a broad model family."""
    lib = []
    # We need a representative set of z values and models.
    # The evaluator uses "disjoint public validation cosmologies".
    # We should sample a wide range of w0, wa and z.
    zs = np.array([0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
    
    # CPL family
    for w0 in (-1.5, -1.2, -1.0, -0.8, -0.5, 0.0):
        for wa in (-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5):
            Efun = lambda a, w0=w0, wa=wa: _model_E(a, w0, wa)
            lib.append((zs, Efun))
            
    # Generalized
    for p in (1.0, 2.0, 3.0, 4.0):
        for C in (-0.5, 0.0, 0.5, 1.0, 1.5, 2.0):
            Efun = lambda a, p=p, C=C: _model_E2(a, p, C)
            lib.append((zs, Efun))
            
    return lib

def _relerr(nodes, weights, z_list, Efun):
    """Compute max relative error for a given set of nodes/weights."""
    errs = []
    # Pre-compute reference integrals using high-order Gauss-Legendre
    # to avoid recomputing for every candidate in the inner loop.
    # However, z_list is fixed per model.
    # We can vectorize the quadrature evaluation.
    
    # Reference integration
    g = np.polynomial.legendre.leggauss(100)
    xi, wi = g
    
    for z in z_list:
        a_ref = 1.0 / (1.0 + z * 0.5 * (xi + 1.0))
        E_ref = Efun(a_ref)
        ref = z * 0.5 * np.sum(wi / E_ref)
        
        a_quad = 1.0 / (1.0 + z * nodes)
        E_quad = Efun(a_quad)
        quad = z * np.sum(weights / E_quad)
        
        err = abs(quad - ref) / ref
        errs.append(err)
        
    return np.max(errs)

def _objective(nodes, weights):
    """Worst-case relative error across the library."""
    # This function will be called many times.
    # To speed it up, we can't easily cache the library Efun calls unless we restructure.
    # But coordinate descent calls this with 1 variable changing.
    
    max_err = 0.0
    for zs, Efun in _LIB:
        e = _relerr(nodes, weights, zs, Efun)
        if e > max_err:
            max_err = e
    return max_err

_LIB = None # Global cache for library

def construct():
    global _LIB
    _LIB = _build_library()
    
    fallback = _fallback()
    best_nodes = fallback["nodes"].copy()
    best_weights = fallback["weights"].copy()
    best_err = _objective(best_nodes, best_weights)
    
    # Initial candidates
    inits = [
        best_nodes.copy(),
        np.array([0.05, 0.18, 0.32, 0.45, 0.58, 0.71, 0.83, 0.95]),
        np.array([0.03, 0.12, 0.23, 0.35, 0.5, 0.65, 0.8, 0.95]),
        np.array([0.07, 0.2, 0.33, 0.46, 0.59, 0.72, 0.85, 0.97]),
        np.array([0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.9]), # Skewed
    ]
    
    for init_nodes in inits:
        if time.time() > DEADLINE - MARGIN:
            break
            
        nodes = init_nodes.copy()
        weights = np.full(8, 1.0 / 8.0)
        
        # Greedy Coordinate Descent
        # We will iterate through each node and weight, optimizing them one by one.
        # We do this for several passes.
        
        for pass_num in range(5): # 5 passes
            if time.time() > DEADLINE - MARGIN:
                break
                
            # Optimize Nodes
            for i in range(8):
                if time.time() > DEADLINE - MARGIN:
                    break
                    
                # Determine bounds for node i
                # Must be strictly increasing: nodes[i-1] < nodes[i] < nodes[i+1]
                # nodes[0] > 0, nodes[7] < 1
                if i == 0:
                    lb = 0.0 + 1e-6
                    ub = nodes[1] - 1e-6
                elif i == 7:
                    lb = nodes[6] + 1e-6
                    ub = 1.0 - 1e-6
                else:
                    lb = nodes[i-1] + 1e-6
                    ub = nodes[i+1] - 1e-6
                    
                if lb >= ub:
                    continue
                    
                def obj_node(val):
                    nodes[i] = val
                    return _objective(nodes, weights)
                    
                res = minimize_scalar(obj_node, bounds=(lb, ub), method='bounded',
                                      options={'xatol': 1e-8})
                if res.success:
                    nodes[i] = res.x
                    
            # Optimize Weights
            # Weights must be positive and sum to 1.
            # Parametrize using softmax or simple normalization?
            # Simple normalization: fix 7 weights, calculate the 8th.
            # Or optimize log-weights and normalize.
            # Let's optimize log-weights for stability, then normalize.
            
            log_w = np.log(weights)
            
            for i in range(8):
                if time.time() > DEADLINE - MARGIN:
                    break
                    
                # We can bound log_w loosely, e.g., [-10, 10]
                # But we must ensure the resulting normalized weights are valid.
                # Since we normalize, any real log_w is valid as long as exp doesn't overflow.
                
                def obj_logw(val):
                    log_w[i] = val
                    w = np.exp(log_w)
                    w = w / np.sum(w)
                    return _objective(nodes, w)
                    
                res = minimize_scalar(obj_logw, bounds=(-10, 10), method='bounded',
                                      options={'xatol': 1e-8})
                if res.success:
                    log_w[i] = res.x
            
            # Update weights
            weights = np.exp(log_w)
            weights = weights / np.sum(weights)
            
            # Check if this pass improved the global best
            current_err = _objective(nodes, weights)
            if current_err < best_err:
                best_err = current_err
                best_nodes = nodes.copy()
                best_weights = weights.copy()
                
    # Final validation and cleanup
    # Ensure nodes are strictly increasing
    # Ensure weights are positive and sum to 1
    
    # Sort nodes if they got shuffled? No, coordinate descent maintains order if bounds are respected.
    # But let's verify.
    if not np.all(np.diff(best_nodes) > 0):
        # Fallback to sorted if something went wrong
        best_nodes = np.sort(best_nodes)
        # Re-optimize weights for sorted nodes?
        # Just use fallback to be safe if structure is broken.
        return fallback
        
    best_weights = best_weights / np.sum(best_weights)
    
    if np.any(best_weights <= 0):
        return fallback
        
    return {"nodes": best_nodes, "weights": best_weights}
# EVOLVE-BLOCK-END
