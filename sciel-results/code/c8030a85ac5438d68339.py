"""Cosmology: universal luminosity-distance quadrature.
STRATEGY: Minimax optimization over a diverse cosmology library using alternating
LP for weights and Nelder-Mead for nodes, with a broader range of dark energy
parameters (w0, wa) to better approximate the worst-case error across disjoint
validation cosmologies.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, linprog

START = time.time()
DEADLINE = 90.0
MARGIN = 5.0

def _fallback():
    n = 8
    return {"nodes": (np.arange(n) + 0.5) / n, "weights": np.full(n, 1.0 / n)}

def _E_cpl(a, w0, wa):
    """CPL E(a) for flat universe.
    H^2/H0^2 = Omega_m a^-3 + Omega_L a^(-3(1+w))
    For CPL: w(a) = w0 + wa(1-a)
    Omega_L(a) = Omega_L0 * a^(-3(1+w0+wa)) * exp(3wa(a-1))
    """
    a = np.clip(a, 1e-8, 1.0)
    Omega_m = 0.3
    Omega_L0 = 0.7
    exp_term = np.exp(3.0 * wa * (a - 1.0))
    pow_term = a ** (-3.0 * (1.0 + w0 + wa))
    Omega_L = Omega_L0 * pow_term * exp_term
    E2 = a ** (-3.0) + Omega_L
    return np.sqrt(E2)

def _integrate(z, nodes, weights, Efun):
    """Compute the quadrature approximation."""
    a = 1.0 / (1.0 + z * nodes)
    E = Efun(a)
    return (1.0 + z) * z * np.sum(weights / E)

def _ref_integrate(z, Efun):
    """High-accuracy reference integration using Gauss-Legendre."""
    n = 64
    xi, wi = np.polynomial.legendre.leggauss(n)
    u = 0.5 * (xi + 1.0)
    a = 1.0 / (1.0 + z * u)
    E = Efun(a)
    val = z * np.sum(wi / E)
    return (1.0 + z) * val

def _build_library():
    """Build a diverse library of (w0, wa) pairs for minimax optimization."""
    # Focus on challenging dark energy models
    # w0 in [-2, 0], wa in [-2, 2]
    w0_vals = np.linspace(-2.0, 0.0, 7)
    wa_vals = np.linspace(-2.0, 2.0, 9)
    lib = []
    for w0 in w0_vals:
        for wa in wa_vals:
            # Skip some trivial cases to focus on challenging ones
            if abs(w0 + 1.0) < 0.1 and abs(wa) < 0.1:
                continue
            lib.append((w0, wa))
    return lib

def _get_error(nodes, weights, zs, lib):
    """Compute worst-case relative error for the given nodes/weights."""
    max_err = 0.0
    for w0, wa in lib:
        Efun = lambda a, w0=w0, wa=wa: _E_cpl(a, w0, wa)
        for z in zs:
            app = _integrate(z, nodes, weights, Efun)
            ref = _ref_integrate(z, Efun)
            if ref == 0:
                continue
            err = abs(app - ref) / ref
            if err > max_err:
                max_err = err
    return max_err

def _optimize_weights(nodes, zs, lib):
    """Optimize weights to minimize L_infinity error using linear programming."""
    n_nodes = len(nodes)
    n_constr = len(zs) * len(lib)
    
    c = np.zeros(n_nodes + 1)
    c[-1] = 1.0  # Minimize t
    
    A_ub = []
    b_ub = []
    
    for w0, wa in lib:
        Efun = lambda a, w0=w0, wa=wa: _E_cpl(a, w0, wa)
        for z in zs:
            a = 1.0 / (1.0 + z * nodes)
            E = Efun(a)
            A_row = (1.0 + z) * z / E
            
            ref = _ref_integrate(z, Efun)
            if ref == 0:
                continue
            
            # Constraint 1: sum_j A_{i,j} w_j - ref_i <= t * ref_i
            row1 = np.zeros(n_nodes + 1)
            row1[:n_nodes] = A_row
            row1[-1] = -ref
            A_ub.append(row1)
            b_ub.append(ref)
            
            # Constraint 2: -sum_j A_{i,j} w_j + ref_i <= t * ref_i
            row2 = np.zeros(n_nodes + 1)
            row2[:n_nodes] = -A_row
            row2[-1] = -ref
            A_ub.append(row2)
            b_ub.append(-ref)
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    bounds = [(0, None)] * n_nodes + [(0, None)]
    
    A_eq = np.zeros((1, n_nodes + 1))
    A_eq[0, :n_nodes] = 1.0
    b_eq = np.array([1.0])
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if res.status == 0:
        weights = res.x[:n_nodes]
        weights = np.clip(weights, 1e-12, None)
        weights = weights / np.sum(weights)
        return weights
    else:
        return np.ones(n_nodes) / n_nodes

def _optimize_nodes(weights, zs, lib, current_nodes):
    """Optimize nodes to minimize L_infinity error using Nelder-Mead."""
    def objective(x):
        nodes = x
        if nodes[0] < 1e-6 or nodes[-1] > 1 - 1e-6:
            return 1e10
        if np.any(np.diff(nodes) < 1e-6):
            return 1e10
        return _get_error(nodes, weights, zs, lib)
    
    res = minimize(objective, current_nodes, method='Nelder-Mead', 
                   options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10})
    
    if res.success:
        nodes = res.x
        if nodes[0] > 1e-6 and nodes[-1] < 1 - 1e-6 and np.all(np.diff(nodes) > 1e-6):
            return nodes
    return current_nodes

def construct():
    fallback = _fallback()
    best_nodes = fallback["nodes"].copy()
    best_weights = fallback["weights"].copy()
    best_err = float('inf')
    
    # Build diverse cosmology library
    lib = _build_library()
    
    # Redshifts to test
    zs = np.linspace(0.05, 3.0, 40)
    
    # Initial nodes: Gauss-Legendre on [0, 1]
    xi, _ = np.polynomial.legendre.leggauss(8)
    nodes = 0.5 * (xi + 1.0)
    nodes = np.sort(nodes)
    weights = np.ones(8) / 8.0
    
    # Alternating optimization
    for iteration in range(30):
        if time.time() > DEADLINE - MARGIN:
            break
        
        # 1. Optimize weights for fixed nodes
        weights = _optimize_weights(nodes, zs, lib)
        
        # 2. Evaluate error
        curr_err = _get_error(nodes, weights, zs, lib)
        
        if curr_err < best_err:
            best_err = curr_err
            best_nodes = nodes.copy()
            best_weights = weights.copy()
        
        # 3. Optimize nodes for fixed weights
        new_nodes = _optimize_nodes(weights, zs, lib, nodes)
        
        if np.allclose(new_nodes, nodes, atol=1e-6):
            break
        nodes = new_nodes
    
    if np.all(np.diff(best_nodes) > 0) and np.all(best_weights > 0):
        best_weights = best_weights / np.sum(best_weights)
        return {"nodes": best_nodes, "weights": best_weights}
    
    return fallback
# EVOLVE-BLOCK-END
