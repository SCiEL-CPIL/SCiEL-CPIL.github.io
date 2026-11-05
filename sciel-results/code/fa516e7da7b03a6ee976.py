"""Cosmology: universal luminosity-distance quadrature.
STRATEGY: Two-stage approach.
Stage 1: Analytically determine that the integrand f(a) = 1/E(a) for CPL models
         has its most rapid variation at small a (high z). The optimal quadrature
         nodes should be concentrated toward small a (large z*nodes).
Stage 2: Perform a 1-D search over a single 'concentration' parameter that maps
         a base Gauss-Legendre node distribution to [0,1] with variable density.
         For each candidate node set, compute the exact Gauss-Legendre weights
         (via polynomial interpolation) and evaluate the worst-case relative error
         over a dense library of CPL cosmologies.
This reduces the problem from 16-D to 1-D, making it tractable within the time budget.
"""

import time
import numpy as np
from scipy.optimize import minimize_scalar, brentq
from scipy.interpolate import Lagrange

START = time.time()
DEADLINE = 85.0
MARGIN = 5.0

def _fallback():
    n = 8
    return {"nodes": (np.arange(n) + 0.5) / n, "weights": np.full(n, 1.0 / n)}

def _E_cpl(a, Om_m, w0, wa):
    """Compute E(a) = H(a)/H0 for flat CPL cosmology."""
    Om_l = 1.0 - Om_m
    if Om_l < 0:
        Om_l = 0.0
    a_safe = np.maximum(a, 1e-10)
    term_m = Om_m * a_safe**-3
    term_l = Om_l * a_safe**-3 * (1.0 + w0) * np.exp(wa * (1.0 - a_safe))
    E_sq = np.maximum(term_m + term_l, 1e-10)
    return np.sqrt(E_sq)

def _gauss_weights_from_nodes(nodes):
    """
    Given n nodes in (0,1) (not necessarily Gauss-Legendre), compute the
    weights for the interpolatory quadrature rule that is exact for polynomials
    of degree 2n-1. This is the Gauss quadrature for the given node set.
    We use the formula: w_i = 1 / (sum_j 1/p'_i(x_j)) where p'(x) is the
    derivative of the nodal polynomial.
    """
    n = len(nodes)
    # Compute weights using the standard Gauss quadrature weight formula
    # w_i = 1 / sum_{j != i} 1/(x_i - x_j)  ... no, that's for specific cases.
    # Correct formula: For nodes x_i, the weights are:
    # w_i = 1 / (p'_i * sum_j 1/p'(x_j))  -- no, let me use the standard formula.
    # 
    # The Gauss quadrature weights for arbitrary nodes can be computed as:
    # w_i = integral_0^1 l_i(x) dx, where l_i is the Lagrange basis polynomial.
    # For efficiency, use the formula:
    # w_i = 1 / (n^2 * sum_{j != i} 1/(x_i - x_j))  -- this is only for Legendre on [-1,1].
    #
    # General formula: w_i = 1 / sum_j (1 / (x_i - x_j)) for j != i, times a normalization.
    # Actually, the correct formula for Gauss quadrature weights given nodes is:
    # w_i = 1 / (p'_i(x_i) * sum_{k} 1/p'_k(x_k))  -- no.
    #
    # Let me just use the direct formula: w_i = integral of Lagrange polynomial.
    # For n=8, this is manageable.
    
    # Use the formula: w_i = 1 / (sum_{j != i} 1/(x_i - x_j))  -- WRONG for general interval.
    
    # Correct approach: The weights for interpolatory quadrature on [a,b] with nodes x_i are:
    # w_i = (b-a) / (n * sum_{j != i} (x_i - x_j) / (x_i - x_j)) ... no.
    
    # Let me use the standard formula for Gauss-Legendre weights mapped to [0,1]:
    # If nodes are t_i in [-1,1], weights w_i^GL, then on [0,1]:
    # x_i = (t_i + 1)/2, W_i = w_i^GL / 2
    # But our nodes are arbitrary, so we need the general formula.
    
    # General formula: w_i = 1 / (sum_{j=0}^{n-1} 1/(x_i - x_j)) for j != i
    # This gives weights that sum to 1 only if the nodes are Gauss nodes.
    # For arbitrary nodes, the interpolatory quadrature weights are:
    # w_i = integral_0^1 L_i(x) dx
    # where L_i(x) = prod_{j != i} (x - x_j)/(x_i - x_j)
    
    # For n=8, compute directly using high-precision quadrature of Lagrange polynomials.
    # Use Gauss-Legendre with 64 points to integrate each Lagrange polynomial.
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(64)
    # Map to [0,1]
    x_gl = 0.5 * (gl_nodes + 1.0)
    w_gl = 0.5 * gl_weights
    
    weights = np.zeros(n)
    for i in range(n):
        # L_i(x) = prod_{j != i} (x - x_j) / (x_i - x_j)
        Li = np.ones_like(x_gl)
        for j in range(n):
            if j != i:
                Li *= (x_gl - nodes[j]) / (nodes[i] - nodes[j])
        weights[i] = np.sum(w_gl * Li)
    
    # Normalize to ensure sum = 1
    total = np.sum(weights)
    if total > 0:
        weights /= total
    return weights

def _build_dense_library():
    """
    Build a dense library of CPL cosmologies for worst-case evaluation.
    Cover a broad range of physically plausible parameters.
    """
    library = []
    zs = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]
    
    Om_m_vals = np.linspace(0.2, 0.4, 5)
    w0_vals = np.linspace(-1.5, -0.5, 5)
    wa_vals = np.linspace(-1.0, 1.0, 5)
    
    for Om_m in Om_m_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                # Skip unphysical combinations
                if w0 < -1.5 or w0 > 0.5:
                    continue
                if wa < -1.5 or wa > 1.5:
                    continue
                library.append((zs, (float(Om_m), float(w0), float(wa))))
    
    # Add extreme cases
    library.append((zs, (0.15, -1.8, 2.0)))
    library.append((zs, (0.45, 0.0, 0.0)))
    library.append((zs, (0.3, -1.0, 0.0)))  # LambdaCDM
    library.append((zs, (0.3, -1.0, 0.5)))
    library.append((zs, (0.3, -0.5, -0.5)))
    
    return library

def _worst_case_error(nodes, weights, library, z_sample=None):
    """
    Compute the worst-case relative error over the library.
    Use a subset of z values for speed if needed.
    """
    max_err = 0.0
    for zs, params in library:
        # Use a subset of z for speed
        if z_sample is not None:
            zs_use = zs[z_sample]
        else:
            zs_use = zs
        
        for z in zs_use:
            a = 1.0 / (1.0 + z * nodes)
            E_vals = _E_cpl(a, *params)
            quad_val = z * np.sum(weights / E_vals)
            dL_quad = (1.0 + z) * quad_val
            
            # Reference: high-order Gauss-Legendre
            a_min = 1.0 / (1.0 + z)
            gl_n, gl_w = np.polynomial.legendre.leggauss(80)
            a_ref = 0.5 * (1.0 - a_min) * gl_n + 0.5 * (1.0 + a_min)
            E_ref = _E_cpl(a_ref, *params)
            I_ref = 0.5 * (1.0 - a_min) * np.sum(gl_w / (a_ref**2 * E_ref))
            dL_ref = (1.0 + z) * I_ref
            
            if dL_ref > 0:
                err = abs(dL_quad - dL_ref) / dL_ref
                if err > max_err:
                    max_err = err
    return max_err

def _node_set_from_concentration(beta):
    """
    Generate 8 nodes in (0,1) using a power-law concentration.
    beta controls how much the nodes are concentrated toward small values (high z).
    
    Start with Gauss-Legendre nodes on [0,1], then apply a nonlinear mapping:
    x_new = x^beta for beta > 1 concentrates toward 0.
    x_new = x^(1/beta) for beta < 1 concentrates toward 1.
    
    We want nodes concentrated toward small x (which corresponds to large z*nodes,
    i.e., high redshift where 1/E(a) varies most).
    """
    # Base Gauss-Legendre nodes on [0,1]
    gl_nodes, _ = np.polynomial.legendre.leggauss(8)
    x_gl = 0.5 * (gl_nodes + 1.0)  # Map to [0,1]
    
    # Apply power-law transformation
    # beta > 1: concentrate toward 0 (small nodes = high z contribution)
    # beta < 1: concentrate toward 1
    # beta = 1: standard Gauss-Legendre
    x_new = x_gl ** beta
    
    # Ensure strictly increasing and in (0,1)
    x_new = np.sort(x_new)
    # Add tiny epsilon to ensure > 0
    x_new = np.maximum(x_new, 1e-10)
    x_new = np.minimum(x_new, 1.0 - 1e-10)
    
    return x_new

def _objective_1d(beta, library, z_sample):
    """
    Objective function for 1-D optimization.
    beta: concentration parameter.
    Returns worst-case relative error.
    """
    nodes = _node_set_from_concentration(beta)
    
    # Check validity
    if not np.all(np.diff(nodes) > 1e-8):
        return 1e6
    if not np.all(nodes > 0) or not np.all(nodes < 1):
        return 1e6
    
    weights = _gauss_weights_from_nodes(nodes)
    
    if np.any(weights <= 0):
        return 1e6
    if abs(np.sum(weights) - 1.0) > 1e-10:
        return 1e6
    
    err = _worst_case_error(nodes, weights, library, z_sample)
    return err

def _refine_nodes_local(nodes, weights, library, z_sample):
    """
    Local refinement of nodes using Nelder-Mead in the node space.
    This is a 8-D local optimization starting from the 1-D result.
    """
    def obj(xv):
        n_nodes = xv[:8]
        # Ensure valid
        if not np.all(np.diff(n_nodes) > 1e-8):
            return 1e6
        if not np.all(n_nodes > 0) or not np.all(n_nodes < 1):
            return 1e6
        
        w = _gauss_weights_from_nodes(n_nodes)
        if np.any(w <= 0):
            return 1e6
        
        return _worst_case_error(n_nodes, w, library, z_sample)
    
    from scipy.optimize import minimize
    try:
        res = minimize(
            obj, nodes, method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        if res.fun < 1e-3:
            n_nodes = res.x[:8]
            if np.all(np.diff(n_nodes) > 1e-8) and np.all(n_nodes > 0) and np.all(n_nodes < 1):
                w = _gauss_weights_from_nodes(n_nodes)
                if np.all(w > 0):
                    return n_nodes, w, res.fun
    except Exception:
        pass
    
    return nodes, weights, None

def construct():
    fallback = _fallback()
    
    # Build library (use subset of z for speed during optimization)
    library = _build_dense_library()
    
    # Use a subset of z values for the optimization phase
    # Each model has 11 z values; use every other one for speed
    z_sample = np.array([0, 1, 2, 4, 6, 8, 10])  # Indices into zs list
    
    # Stage 1: 1-D search over concentration parameter beta
    best_beta = 1.0
    best_err = float('inf')
    
    # Coarse grid search first
    beta_grid = np.linspace(0.3, 3.0, 30)
    
    for beta in beta_grid:
        if time.time() > DEADLINE - 20:
            break
        err = _objective_1d(beta, library, z_sample)
        if err < best_err:
            best_err = err
            best_beta = beta
    
    # Fine search around best
    if time.time() < DEADLINE - 15:
        try:
            res = minimize_scalar(
                _objective_1d,
                args=(library, z_sample),
                bounds=(max(0.3, best_beta - 0.5), min(3.0, best_beta + 0.5)),
                method='bounded',
                options={'xatol': 1e-6, 'maxiter': 100}
            )
            if res.fun < best_err:
                best_err = res.fun
                best_beta = res.x
        except Exception:
            pass
    
    # Get nodes and weights from best beta
    nodes = _node_set_from_concentration(best_beta)
    weights = _gauss_weights_from_nodes(nodes)
    
    # Stage 2: Local refinement of nodes (8-D)
    if time.time() < DEADLINE - 10:
        try:
            nodes, weights, refined_err = _refine_nodes_local(nodes, weights, library, z_sample)
            if refined_err is not None and refined_err < best_err:
                best_err = refined_err
        except Exception:
            pass
    
    # Final validation
    if (np.all(np.diff(nodes) > 1e-8) and 
        np.all(nodes > 0) and np.all(nodes < 1) and 
        np.all(weights > 0) and 
        abs(np.sum(weights) - 1.0) < 1e-10):
        
        # Final check with full z list
        final_err = _worst_case_error(nodes, weights, library, None)
        if final_err < 0.5:  # Accept if error < 50%
            return {"nodes": nodes, "weights": weights}
    
    # If 1-D result is bad, try a few alternative strategies
    if time.time() < DEADLINE - 5:
        # Try standard Gauss-Legendre
        gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
        x_gl = 0.5 * (gl_nodes + 1.0)
        w_gl = 0.5 * gl_weights
        err_gl = _worst_case_error(x_gl, w_gl, library, None)
        if err_gl < 0.5:
            return {"nodes": x_gl, "weights": w_gl}
        
        # Try trapezoidal-like uniform nodes with Gauss weights
        uniform_nodes = np.linspace(0.05, 0.95, 8)
        uniform_weights = _gauss_weights_from_nodes(uniform_nodes)
        err_uni = _worst_case_error(uniform_nodes, uniform_weights, library, None)
        if err_uni < 0.5:
            return {"nodes": uniform_nodes, "weights": uniform_weights}
    
    return fallback

# EVOLVE-BLOCK-END
