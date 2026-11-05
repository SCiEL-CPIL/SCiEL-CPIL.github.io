"""Cosmology: universal luminosity-distance quadrature.
STRATEGY: Chebyshev approximation for dominant CPL family.
Instead of a global minimax over many models, we focus on the flat CPL family
with w0=-1 and varying wa, which typically produces the most challenging
integrands for fixed nodes. We use an alternating optimization scheme:
1. Fix nodes, optimize weights to minimize L_infinity error on a dense set of
   challenging redshifts.
2. Fix weights, optimize nodes (via gradient-free search) to reduce the error.
This structured approach avoids the high-dimensional non-convexity of the
full minimax problem and converges faster to a robust rule.
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
    For w0=-1, wa=var, w(a) = -1 + wa(1-a).
    Omega_L(a) = Omega_L0 * a^(-3(1+w0+wa)) * exp(3wa(a-1))
    Since w0=-1, 1+w0=0.
    Omega_L(a) = Omega_L0 * a^(-3wa) * exp(3wa(a-1))
    E^2 = a^-3 + Omega_L0 * a^(-3wa) * exp(3wa(a-1))
    """
    a = np.clip(a, 1e-8, 1.0)
    Omega_m = 0.3
    Omega_L0 = 0.7
    # To ensure flatness and stability, we use the standard CPL form
    w = w0 + wa * (1.0 - a)
    # Integral form for Omega_L:
    # Omega_L(a) = Omega_L0 * a^(-3(1+w0+wa)) * exp(3wa(a-1))
    # For w0=-1:
    exp_term = np.exp(3.0 * wa * (a - 1.0))
    pow_term = a ** (-3.0 * wa)
    Omega_L = Omega_L0 * pow_term * exp_term
    E2 = a ** (-3.0) + Omega_L
    return np.sqrt(E2)

def _integrate(z, nodes, weights, Efun):
    """Compute the quadrature approximation."""
    a = 1.0 / (1.0 + z * nodes)
    E = Efun(a)
    # The formula is dL/(c/H0) = (1+z) * integral_0^z dz' / E(a(z'))
    # Wait, the prompt says: dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i)
    # This implies the integral is approximated by z * sum(w_i / E(a_i))
    # Let's check the variable change.
    # dL = c/H0 * (1+z) * integral_0^z dz' / E(z')
    # If we change variable to a=1/(1+z), z goes 0->z_max, a goes 1->1/(1+z_max).
    # The prompt's formula is specific: (1+z)*z*sum(w/E).
    # This looks like an approximation of the integral \int_0^z dz'/E(z') by z * sum(w/E).
    # Let's stick to the prompt's formula exactly.
    return (1.0 + z) * z * np.sum(weights / E)

def _ref_integrate(z, Efun):
    """High-accuracy reference integration using Gauss-Legendre."""
    # Integrate \int_0^z dz' / E(1/(1+z'))
    # Change variable to u in [0, 1]: z' = z * u
    # dz' = z du
    # Integral = z * \int_0^1 du / E(1/(1+z*u))
    n = 64
    xi, wi = np.polynomial.legendre.leggauss(n)
    u = 0.5 * (xi + 1.0)
    a = 1.0 / (1.0 + z * u)
    E = Efun(a)
    val = z * np.sum(wi / E)
    return (1.0 + z) * val

def _get_error(nodes, weights, zs, wa_vals):
    """Compute worst-case relative error for the given nodes/weights."""
    max_err = 0.0
    for wa in wa_vals:
        Efun = lambda a: _E_cpl(a, -1.0, wa)
        for z in zs:
            app = _integrate(z, nodes, weights, Efun)
            ref = _ref_integrate(z, Efun)
            if ref == 0:
                continue
            err = abs(app - ref) / ref
            if err > max_err:
                max_err = err
    return max_err

def _optimize_weights(nodes, zs, wa_vals):
    """Optimize weights to minimize L_infinity error."""
    # We want to minimize max_i |e_i| where e_i are relative errors.
    # This is a linear programming problem if we fix the nodes and linearize the error.
    # However, the error is not linear in weights because of the 1/E term?
    # No, app = (1+z) * z * sum(w_i / E_i). This IS linear in w_i.
    # ref is constant.
    # err_i = | (1+z_i) z_i sum(w_j / E_{ij}) - ref_i | / ref_i
    # Let A_{i,j} = (1+z_i) z_i / E_{ij}. Then app_i = sum_j A_{i,j} w_j.
    # We want to minimize t such that |app_i - ref_i| <= t * ref_i for all i.
    # |sum_j A_{i,j} w_j - ref_i| <= t * ref_i
    # -t * ref_i <= sum_j A_{i,j} w_j - ref_i <= t * ref_i
    # -t * ref_i + ref_i <= sum_j A_{i,j} w_j <= t * ref_i + ref_i
    # ref_i (1 - t) <= sum_j A_{i,j} w_j <= ref_i (1 + t)
    
    n_nodes = len(nodes)
    n_constr = len(zs) * len(wa_vals)
    
    # Variables: w_0, ..., w_7, t
    # w must be positive. t must be non-negative.
    # Sum w = 1.
    
    c = np.zeros(n_nodes + 1)
    c[-1] = 1.0  # Minimize t
    
    A_ub = []
    b_ub = []
    
    idx = 0
    for wa in wa_vals:
        Efun = lambda a: _E_cpl(a, -1.0, wa)
        for z in zs:
            a = 1.0 / (1.0 + z * nodes)
            E = Efun(a)
            A_row = (1.0 + z) * z / E
            
            ref = _ref_integrate(z, Efun)
            if ref == 0:
                idx += 1
                continue
            
            # Constraint 1: sum_j A_{i,j} w_j - ref_i <= t * ref_i
            # sum_j A_{i,j} w_j - t * ref_i <= ref_i
            row1 = np.zeros(n_nodes + 1)
            row1[:n_nodes] = A_row
            row1[-1] = -ref
            A_ub.append(row1)
            b_ub.append(ref)
            
            # Constraint 2: -sum_j A_{i,j} w_j + ref_i <= t * ref_i
            # -sum_j A_{i,j} w_j - t * ref_i <= -ref_i
            row2 = np.zeros(n_nodes + 1)
            row2[:n_nodes] = -A_row
            row2[-1] = -ref
            A_ub.append(row2)
            b_ub.append(-ref)
            
            idx += 1
            
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Bounds: w_i >= 0, t >= 0
    bounds = [(0, None)] * n_nodes + [(0, None)]
    
    # Equality constraint: sum w = 1
    A_eq = np.zeros((1, n_nodes + 1))
    A_eq[0, :n_nodes] = 1.0
    b_eq = np.array([1.0])
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if res.status == 0:
        weights = res.x[:n_nodes]
        # Ensure numerical stability
        weights = np.clip(weights, 1e-12, None)
        weights = weights / np.sum(weights)
        return weights
    else:
        # Fallback to uniform if LP fails
        return np.ones(n_nodes) / n_nodes

def _optimize_nodes(weights, zs, wa_vals, current_nodes):
    """Optimize nodes to minimize L_infinity error."""
    def objective(x):
        nodes = x
        # Ensure nodes are in [0, 1] and increasing
        if nodes[0] < 1e-6 or nodes[-1] > 1 - 1e-6:
            return 1e10
        if np.any(np.diff(nodes) < 1e-6):
            return 1e10
        return _get_error(nodes, weights, zs, wa_vals)
    
    # Use Nelder-Mead
    res = minimize(objective, current_nodes, method='Nelder-Mead', 
                   options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
    
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
    
    # Focus on CPL family with w0=-1, wa in [-1, 1]
    # This covers a wide range of dark energy behaviors including phantom and quintessence
    wa_vals = np.linspace(-1.0, 1.0, 11)
    # Redshifts to test: focus on low to medium z where errors are often largest
    # due to the (1+z) factor and the integral accumulation
    zs = np.linspace(0.1, 3.0, 30)
    
    # Initial nodes: Gauss-Legendre on [0, 1]
    xi, _ = np.polynomial.legendre.leggauss(8)
    nodes = 0.5 * (xi + 1.0)
    nodes = np.sort(nodes)
    weights = np.ones(8) / 8.0
    
    # Alternating optimization
    for iteration in range(20):
        if time.time() > DEADLINE - MARGIN:
            break
            
        # 1. Optimize weights for fixed nodes
        weights = _optimize_weights(nodes, zs, wa_vals)
        
        # 2. Evaluate error
        curr_err = _get_error(nodes, weights, zs, wa_vals)
        
        if curr_err < best_err:
            best_err = curr_err
            best_nodes = nodes.copy()
            best_weights = weights.copy()
            
        # 3. Optimize nodes for fixed weights
        new_nodes = _optimize_nodes(weights, zs, wa_vals, nodes)
        
        # Check if nodes changed significantly
        if np.allclose(new_nodes, nodes, atol=1e-6):
            # Converged
            break
        nodes = new_nodes
        
    # Final validation
    if np.all(np.diff(best_nodes) > 0) and np.all(best_weights > 0):
        best_weights = best_weights / np.sum(best_weights)
        return {"nodes": best_nodes, "weights": best_weights}
    
    return fallback
# EVOLVE-BLOCK-END
