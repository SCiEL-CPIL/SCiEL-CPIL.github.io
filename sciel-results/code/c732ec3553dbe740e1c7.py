"""
Strategy: Minimize worst-case error in integrand approximation.
The task asks for a quadrature rule for dL/(c/H0) = (1+z) * z * sum w_i / E(a_i).
Let f(z, t) = 1/E(z*t) where t in [0,1].
The approximation is z * sum w_i f(z, t_i).
This is equivalent to approximating the function g_z(t) = f(z, t) on [0,1]
using a quadrature rule.
However, the "universal" requirement means nodes/weights are fixed.
The error depends on how well sum w_i f(z, t_i) approximates \int_0^1 f(z, t) dt.
Since f(z, t) is smooth for reasonable z and cosmologies, we can assume
f(z, t) can be well-approximated by a low-order polynomial in t for fixed z.
If we choose nodes and weights that are exact for polynomials up to degree 7,
we get a Gaussian quadrature rule.
However, the "worst-case" over cosmologies suggests we should optimize for
the specific functional form of 1/E(z).
1/E(z) = [Omega_m (1+z)^3 + Omega_de (1+z)^(3(1+w0+wa*z))]^{-1/2}.
Let's expand this in powers of t (where z' = z*t).
1/E(z*t) is a function of (1+z*t).
For small z, it's smooth. For large z, it varies.
A better approach:
The integral is I(z) = \int_0^1 \frac{dt}{E(z t)}.
We want \sum w_i \frac{1}{E(z t_i)} \approx I(z).
Let's use a deterministic search over a grid of candidate nodes (e.g., Gauss-Legendre,
Gauss-Jacobi, or Chebyshev nodes) and optimize only the weights via least squares
on a large set of (z, w0, wa) samples. This is much faster than optimizing 16 parameters
with differential evolution.
Actually, even simpler:
Use Gauss-Legendre nodes (which are optimal for polynomials) and see if that's enough.
If not, optimize weights for those fixed nodes.
Or, use a coarse grid search for nodes and least squares for weights.
Given the 90s limit, a full 16-param DE is risky.
Let's try:
1. Fix nodes to Gauss-Legendre nodes on [0,1].
2. Optimize weights using least squares to minimize error on a representative set of cosmologies and redshifts.
3. If time permits, refine nodes with a local optimizer (L-BFGS-B) starting from GL nodes.

Why this is different:
The current program uses Differential Evolution on 16 variables (8 nodes, 8 weights)
with a slow objective function (adaptive integration for reference).
My strategy fixes the structure (nodes) or uses a much faster objective (analytic/numeric
integration with fixed steps) and uses a deterministic/local optimizer for weights,
potentially refining nodes. This avoids the stochastic global search overhead.
"""

import time
import numpy as np
from scipy.optimize import minimize, least_squares
from scipy.integrate import quad
from scipy.special import roots_legendre

_START_TIME = time.time()
_DEADLINE = _START_TIME + 85.0

def _E_of_z(z, w0, wa, Omega_m=0.3):
    """Compute E(z) = H(z)/H0."""
    Omega_de = 1.0 - Omega_m
    # Handle z=0 carefully if needed, but 1/E(0)=1
    term_m = Omega_m * (1 + z)**3
    term_de = Omega_de * (1 + z)**(3 * (1 + w0 + wa * z))
    return np.sqrt(term_m + term_de)

def _true_integral(z, w0, wa, Omega_m=0.3):
    """Compute \int_0^1 dt / E(z*t) using quad."""
    if z == 0:
        return 1.0 # Limit as z->0 of \int_0^1 dt/E(0) = 1
    def integrand(t):
        return 1.0 / _E_of_z(z * t, w0, wa, Omega_m)
    val, _ = quad(integrand, 0, 1, limit=100)
    return val

def _approx_integral(z, w0, wa, nodes, weights, Omega_m=0.3):
    """Compute \sum w_i / E(z*t_i)."""
    if z == 0:
        return np.sum(weights) # Should be 1
    z_nodes = z * nodes
    E_vals = _E_of_z(z_nodes, w0, wa, Omega_m)
    return np.dot(weights, 1.0 / E_vals)

def _compute_errors(nodes, weights, test_set, Omega_m=0.3):
    """Compute relative errors for a set of (z, w0, wa)."""
    errors = []
    for z, w0, wa in test_set:
        true_val = _true_integral(z, w0, wa, Omega_m)
        approx_val = _approx_integral(z, w0, wa, nodes, weights, Omega_m)
        if true_val > 1e-10:
            rel_err = abs(approx_val - true_val) / true_val
            errors.append(rel_err)
    return np.array(errors)

def construct():
    # Deadline guard
    if time.time() > _DEADLINE - 10:
        # Fallback: Gauss-Legendre nodes and weights
        n = 8
        nodes, weights = roots_legendre(n)
        nodes = (nodes + 1) / 2  # Map [-1,1] to [0,1]
        weights = weights / 2
        return {"nodes": nodes, "weights": weights}

    # 1. Generate test set
    # Cover a range of w0, wa and z
    w0_vals = np.linspace(-1.5, -0.5, 5)
    wa_vals = np.linspace(-0.5, 0.5, 5)
    z_vals = np.logspace(-1, np.log10(5.0), 10) # 0.1 to 5.0
    
    test_set = []
    for w0 in w0_vals:
        for wa in wa_vals:
            for z in z_vals:
                test_set.append((z, w0, wa))
                
    # 2. Initial guess: Gauss-Legendre
    n = 8
    gl_nodes, gl_weights = roots_legendre(n)
    init_nodes = (gl_nodes + 1) / 2
    init_weights = gl_weights / 2
    
    # 3. Optimize weights for fixed nodes (Least Squares)
    # We want to minimize sum (approx - true)^2
    # This is linear in weights!
    # approx = sum w_i / E(z*t_i)
    # Let A[z_idx, i] = 1/E(z_idx * t_i)
    # Let b[z_idx] = true_integral(z_idx)
    # Minimize || A w - b ||^2
    # This can be solved analytically with np.linalg.lstsq
    
    # Build matrix A
    # Number of test points
    num_tests = len(test_set)
    A = np.zeros((num_tests, n))
    b = np.zeros(num_tests)
    
    for idx, (z, w0, wa) in enumerate(test_set):
        b[idx] = _true_integral(z, w0, wa)
        for i in range(n):
            A[idx, i] = 1.0 / _E_of_z(z * init_nodes[i], w0, wa)
            
    # Solve least squares
    # We want weights > 0 and sum to 1.
    # First, get unconstrained LS solution
    try:
        w_ls, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        w_ls = init_weights
        
    # Normalize to sum to 1
    if np.sum(w_ls) > 1e-12:
        w_ls = w_ls / np.sum(w_ls)
        
    # Check if any weights are negative. If so, use init_weights or project.
    if np.any(w_ls < 1e-6):
        # Fallback to GL weights if LS produces negatives
        w_opt = init_weights
    else:
        w_opt = w_ls
        
    # 4. Refine nodes and weights jointly using local optimization
    # Use L-BFGS-B to minimize worst-case error (or sum of squared errors)
    # Objective: sum of squared relative errors
    def objective(params):
        if time.time() > _DEADLINE - 5:
            return 1e10
        nodes_p = params[:n]
        weights_p = params[n:]
        
        # Enforce constraints: nodes in [0,1] increasing, weights > 0, sum=1
        # L-BFGS-B handles bounds, but not equality constraints or monotonicity directly.
        # We can parameterize:
        # nodes: use cumulative sum of positive gaps? Or just bound them and hope?
        # Better: optimize gaps.
        # Let gaps = params[:n-1] (positive), node_1 = params[n-1] (positive)
        # This is complex. Let's just bound nodes in [0,1] and weights in [0, 1].
        # And rely on the fact that GL nodes are well-spaced.
        
        # Simple bounds check
        if np.any(nodes_p < 0) or np.any(nodes_p > 1):
            return 1e10
        if np.any(weights_p < 0):
            return 1e10
            
        # Sort nodes and weights? No, keep them associated.
        # If we don't sort, the optimizer might swap them.
        # To ensure strict increase, we can penalize inversions or just assume
        # the local optimum near GL nodes maintains order.
        
        # Compute errors
        errs = []
        for z, w0, wa in test_set:
            true_val = _true_integral(z, w0, wa)
            z_nodes = z * nodes_p
            E_vals = _E_of_z(z_nodes, w0, wa)
            approx_val = np.dot(weights_p, 1.0 / E_vals)
            if true_val > 1e-10:
                rel_err = (approx_val - true_val) / true_val
                errs.append(rel_err**2)
                
        return sum(errs)
        
    # Initial point for joint optimization
    x0 = np.concatenate([init_nodes, w_opt])
    
    # Bounds
    # Nodes: [0, 1]
    # Weights: [1e-6, 1.0]
    bounds = [(1e-6, 1 - 1e-6)] * n + [(1e-6, 1.0)] * n
    
    try:
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                          options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-8})
        
        if result.success or result.fun < 1e-4:
            opt_nodes = result.x[:n]
            opt_weights = result.x[n:]
            
            # Normalize weights
            opt_weights = opt_weights / np.sum(opt_weights)
            
            # Ensure strict increase
            # If not, sort
            sort_idx = np.argsort(opt_nodes)
            opt_nodes = opt_nodes[sort_idx]
            opt_weights = opt_weights[sort_idx]
            
            # Check strict increase
            diffs = np.diff(opt_nodes)
            if np.any(diffs <= 0):
                # Fallback
                return {"nodes": init_nodes, "weights": w_opt}
                
            # Final check: is this better than the LS-only solution?
            err_joint = objective(np.concatenate([opt_nodes, opt_weights]))
            err_ls = objective(np.concatenate([init_nodes, w_opt]))
            
            if err_joint < err_ls:
                return {"nodes": opt_nodes, "weights": opt_weights}
            else:
                return {"nodes": init_nodes, "weights": w_opt}
        else:
            return {"nodes": init_nodes, "weights": w_opt}
            
    except Exception:
        return {"nodes": init_nodes, "weights": w_opt}

# EVOLVE-BLOCK-END
