"""Rewritten program for luminosity-distance quadrature optimization."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import quad


def _E_cpl(a, w0, wa):
    """CPL dark energy model: E(a) = (a^-3 + (1+w0+wa)*a^3 + (w0+wa)*a^2 + ...)^0.5
    Actually for flat universe with CPL:
    E^2(a) = (1-a)^3(1+3w0+3wa) + a^3 + a(1+wa)*((1-a)^3 - 1)
    More standard form:
    E^2(a) = a^-3 * Om_m + a^-(3(1+w0+wa)) * Om_DE * (1 + wa*(1-a))
    For flat: Om_m + Om_DE = 1, Om_r negligible at late times but we'll include it.
    Let's use: E^2(a) = Om_m*a^-3 + Om_r*a^-4 + Om_DE*a^(-3*(1+w0+wa)) * (1+wa*(1-a))
    with Om_m + Om_r + Om_DE = 1
    """
    Om_m = 0.3
    Om_r = 5e-5
    Om_DE = 1.0 - Om_m - Om_r
    w = w0 + wa * (1 - a)
    return np.sqrt(Om_m * a**(-3) + Om_r * a**(-4) + Om_DE * a**(-3*(1+w0+wa)) * (1 + wa*(1-a)))


def _integrand(z, w0, wa):
    """The integrand for luminosity distance: z / E(a) where a = 1/(1+z)"""
    a = 1.0 / (1.0 + z)
    return z / _E_cpl(a, w0, wa)


def _true_dL(z, w0, wa):
    """Compute true dimensionless luminosity distance using adaptive quadrature.
    dL/(c/H0) = (1+z) * integral_0^z z'/E(a') dz'
    """
    val, _ = quad(_integrand, 0, z, args=(w0, wa), limit=200, epsabs=1e-12, epsrel=1e-12)
    return (1 + z) * val


def _quadrature_dL(z, nodes, weights, w0, wa):
    """Compute quadrature approximation of dimensionless luminosity distance."""
    a_vals = 1.0 / (1.0 + z * nodes)
    E_vals = _E_cpl(a_vals, w0, wa)
    integral = np.sum(weights / E_vals)
    return z * integral


def _worst_rel_error(nodes, weights, test_zs, test_cosmos):
    """Compute worst relative error (in percent) over test set."""
    max_err = 0.0
    for (w0, wa) in test_cosmos:
        for z in test_zs:
            true_val = _true_dL(z, w0, wa)
            quad_val = _quadrature_dL(z, nodes, weights, w0, wa)
            if true_val > 1e-10:
                rel_err = abs(quad_val - true_val) / true_val * 100.0
                if rel_err > max_err:
                    max_err = rel_err
    return max_err


def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    # Test cosmologies (disjoint from training, for evaluation)
    # We'll train on a set and evaluate on another
    train_cosmos = [
        (0.0, 0.0),      # LCDM
        (-1.0, 0.0),     # Cosmological constant
        (0.5, -1.0),     # Quintessence-like
        (-1.5, 0.5),     # Phantom-like
        (0.0, -1.0),     # CPL with wa=-1
        (-0.5, 0.5),     # Another CPL
    ]
    
    test_cosmos = [
        (0.1, -0.5),
        (-0.3, 0.8),
        (0.3, -0.2),
        (-1.2, 0.3),
        (0.0, 0.5),
        (-0.8, -0.4),
    ]
    
    test_zs = np.array([0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0])
    
    # Initial guess: midpoint rule
    n = 8
    nodes_init = (np.arange(n) + 0.5) / n
    weights_init = np.full(n, 1.0 / n)
    
    # Parameterize: nodes must be strictly increasing in (0,1)
    # Use logit transform: x_i = sigmoid(t_i), ensure ordering by t_i < t_{i+1}
    # Or simpler: parameterize as cumulative sums of positive gaps
    # Let's use: nodes = cumulative sum of exp(p_i) normalized, but that's complex.
    # Alternative: parameterize by 8 values, enforce ordering via sorting won't work for optimization.
    # Better: use the fact that nodes are in (0,1) and strictly increasing.
    # Parameterize as: nodes_i = sigmoid(y_i), with y_0 < y_1 < ... < y_7
    # We can parameterize y_i = y_0 + sum_{j=0}^{i-1} exp(d_j)
    
    def _params_to_nodes_weights(params):
        """Convert 16 parameters to 8 nodes and 8 weights."""
        y = params[:8]
        w_log = params[8:]
        
        # Ensure y is strictly increasing by adding a small epsilon-based ordering
        # Actually, let's just use the raw y and sigmoid, then sort and reindex
        # This is problematic. Instead, let's use a different parameterization.
        
        # New approach: nodes are determined by 7 gaps g_1..g_7 > 0 and a base
        # nodes_0 = g_0 / (g_0 + ... + g_7)
        # nodes_i = (g_0 + ... + g_i) / (g_0 + ... + g_7)
        # This ensures strictly increasing nodes in (0,1)
        
        gaps = np.exp(params[:7])  # 7 gaps
        total = np.sum(gaps)
        cumsum = np.cumsum(gaps)
        nodes = cumsum / total
        
        # Weights: positive, sum to 1
        w_raw = np.exp(params[7:])  # 8 weights
        weights = w_raw / np.sum(w_raw)
        
        return nodes, weights
    
    # Initial parameters
    # For midpoint: nodes = [1/16, 3/16, ..., 15/16]
    # gaps = differences: 2/16, 2/16, ... so all gaps equal
    # nodes_0 = g0/sum = 1/8 * ... wait, let me recalculate
    # If all 7 gaps are equal: g, then nodes = [g, 2g, 3g, 4g, 5g, 6g, 7g] / (7g) = [1/7, 2/7, ..., 1]
    # That's not midpoint. Midpoint has 8 nodes: 1/16, 3/16, ..., 15/16
    # gaps: 3/16, 2/16, 2/16, 2/16, 2/16, 2/16, 2/16 -> not all equal
    # Let's just use a reasonable initial guess
    
    # Actually, let me use a simpler approach: directly optimize nodes and weights with constraints
    # using scipy's minimize with bounds
    
    def _objective(params):
        if time.time() > deadline:
            return 1e10
        nodes, weights = _params_to_nodes_weights(params)
        return _worst_rel_error(nodes, weights, test_zs, train_cosmos)
    
    # Initial parameters
    # For midpoint-like: let's just start with log-spaced or something reasonable
    # nodes ~ [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.7, 0.85]
    # gaps = [0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.15, ...] wait, 7 gaps for 8 nodes
    # nodes_0 = g0/sum, nodes_1 = (g0+g1)/sum, etc.
    # For nodes = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.7, 0.85]:
    # g0 = 0.05, g1 = 0.1, g2 = 0.1, g3 = 0.1, g4 = 0.1, g5 = 0.1, g6 = 0.15, g7 = 0.15
    # Wait, we have 7 gaps for 8 nodes: g0..g6
    # nodes_0 = g0/S, nodes_1 = (g0+g1)/S, ..., nodes_7 = S/S = 1
    # So the last node is always 1! That's not what we want.
    
    # Let me fix: 8 nodes require 8 gaps or a different parameterization.
    # nodes_i = (g0 + ... + g_i) / (g0 + ... + g_{i+1}) for i=0..7, with g_8 = 0
    # Or: use 8 gaps, nodes_i = cumsum(gaps[:i+1]) / cumsum(gaps)
    # Then nodes_7 = 1. Still last node is 1.
    
    # Better: nodes in (0,1), strictly increasing. Use 8 independent parameters with sigmoid.
    # nodes_i = sigmoid(y_i), and enforce y_0 < y_1 < ... < y_7 by construction.
    # y_i = base + i * step + jitter_i, where jitter is small
    # Or: y_0 = y, y_i = y_{i-1} + exp(d_i) for i=1..7
    # So 8 parameters for y: y_0, d_1, ..., d_7 (all real, d_i > 0 via exp)
    
    def _params_to_nodes_weights_v2(params):
        y0 = params[0]
        d = np.exp(params[1:8])  # 7 positive gaps in y-space
        y = np.empty(8)
        y[0] = y0
        for i in range(1, 8):
            y[i] = y[i-1] + d[i-1]
        nodes = 1.0 / (1.0 + np.exp(-y))  # sigmoid
        # Clip to (0,1) strictly
        nodes = np.clip(nodes, 1e-8, 1 - 1e-8)
        # Ensure strictly increasing (sigmoid is monotonic, so this should hold)
        
        w_raw = np.exp(params[8:])  # 8 positive weights
        weights = w_raw / np.sum(w_raw)
        return nodes, weights
    
    # Initial: y values for midpoint nodes [1/16, 3/16, ..., 15/16]
    # sigmoid^{-1}(x) = log(x/(1-x))
    mid_nodes = (np.arange(8) + 0.5) / 8
    y_init = np.log(mid_nodes / (1 - mid_nodes))
    d_init = np.diff(y_init)  # should be positive
    y0_init = y_init[0]
    
    params_init = np.concatenate([[y0_init], d_init, np.zeros(8)])  # weights init to 1
    
    # Optimize
    best_params = params_init.copy()
    best_nodes, best_weights = _params_to_nodes_weights_v2(params_init)
    best_score = _worst_rel_error(best_nodes, best_weights, test_zs, train_cosmos)
    
    # Try differential evolution first (global)
    bounds = [(-10, 10)] + [(-5, 5)] * 7 + [(-5, 5)] * 8
    try:
        if time.time() < deadline - 10:
            de_result = differential_evolution(
                _objective, bounds, seed=42, maxiter=50, tol=1e-6,
                popsize=15, polish=True, workers=1
            )
            if de_result.fun < best_score:
                best_params = de_result.x
                best_nodes, best_weights = _params_to_nodes_weights_v2(best_params)
                best_score = de_result.fun
    except Exception:
        pass
    
    # Refine with L-BFGS-B
    try:
        if time.time() < deadline - 5:
            res = minimize(
                _objective, best_params, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 100, 'ftol': 1e-10}
            )
            if res.fun < best_score:
                best_params = res.x
                best_nodes, best_weights = _params_to_nodes_weights_v2(best_params)
                best_score = res.fun
    except Exception:
        pass
    
    # Final validation on test set
    test_error = _worst_rel_error(best_nodes, best_weights, test_zs, test_cosmos)
    
    # Ensure constraints are met
    # Nodes strictly increasing in [0,1]
    nodes = np.clip(best_nodes, 1e-10, 1 - 1e-10)
    # Ensure strictly increasing
    for i in range(1, 8):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + 1e-10
    # Re-clip
    nodes = np.clip(nodes, 1e-10, 1 - 1e-10)
    
    # Weights positive, sum to 1
    weights = np.maximum(best_weights, 1e-15)
    weights = weights / np.sum(weights)
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
