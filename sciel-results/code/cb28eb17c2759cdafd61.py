"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution
from scipy.integrate import quad
import sys

# Deadline guard
START_TIME = time.time()
DEADLINE = 90  # seconds

# Representative cosmologies (w0, wa) for training/validation of the rule
# We use a diverse set to ensure the rule is "universal"
COSMOLOGIES = [
    (-1.0, 0.0),   # LambdaCDM
    (-1.0, 0.5),   # Quintessence-like
    (-0.9, 0.3),   # General CPL
    (-1.1, 0.0),   # Phantom
    (-0.5, 0.0),   # Matter dominated limit
    (-1.0, -0.5),  # Different wa sign
    (-0.8, 0.2),   # Intermediate
    (-1.2, 0.1),   # More phantom
    (-0.95, 0.1),  # Near LCDM
    (-1.0, 0.2),   # Mild quintessence
]

def E_z(z, w0, wa):
    """Calculate E(z) = H(z)/H0 for flat universe with CPL w(z) = w0 + wa*(1 - 1/(1+z))"""
    if z < 0:
        return 1.0
    # E(z)^2 = (1+z)^3 * (1+z)^(-3(1+w0)) * (1 - wa*(1 - 1/(1+z)))^(1.5*wa)
    # Simplified: E(z)^2 = (1+z)^(-3*w0) * (1 - wa*(1 - 1/(1+z)))^(1.5*wa)
    # Note: (1 - 1/(1+z)) = z/(1+z)
    term1 = (1 + z) ** (-3 * w0)
    # Handle the term (1 - wa * z / (1+z))
    base = 1 - wa * z / (1 + z)
    if base < 0:
        # This can happen for some parameters, but usually w0, wa are chosen such that E is real.
        # For safety in optimization, return a large value or handle carefully.
        # In standard CPL, we assume the term is positive.
        base = 1e-12
    term2 = base ** (1.5 * wa)
    return np.sqrt(term1 * term2)

def true_integral(z, w0, wa):
    """Calculate the true integral part: int_0^1 dx / E(z*x)"""
    if z == 0:
        return 1.0
    def integrand(x):
        return 1.0 / E_z(z * x, w0, wa)
    val, _ = quad(integrand, 0, 1, limit=100)
    return val

def quadrature_approx(nodes, weights, z, w0, wa):
    """Calculate the quadrature approximation: sum w_i / E(z * x_i)"""
    if z == 0:
        return 1.0
    # x_i are nodes in [0,1]
    # z * x_i is the redshift
    z_vals = z * nodes
    E_vals = np.array([E_z(zv, w0, wa) for zv in z_vals])
    return np.sum(weights / E_vals)

def objective(params):
    """
    Objective function to minimize: worst-case relative error.
    params: [x1, ..., x8, w1, ..., w8]
    Constraints:
    - x_i strictly increasing in [0,1]
    - w_i > 0
    - sum(w_i) = 1 (handled by normalization or penalty)
    """
    if time.time() > DEADLINE - 5:
        return 1e9  # Return large value to stop optimization

    x = params[:8]
    w = params[8:]
    
    # Enforce strict increase and bounds for x
    # Use a penalty method or transformation.
    # For differential_evolution, bounds are handled, but strict increase is not.
    # We will sort x and penalize if not strictly increasing or if duplicates exist.
    # Actually, it's better to define x in terms of sorted unique points.
    # Let's just penalize violations.
    
    if np.any(x < 0) or np.any(x > 1):
        return 1e9
    
    # Check strict increase
    if np.any(np.diff(x) <= 1e-6):
        return 1e9
        
    # Check weights positive
    if np.any(w <= 0):
        return 1e9
        
    # Normalize weights to sum to 1
    w_sum = np.sum(w)
    if w_sum < 1e-6:
        return 1e9
    w_norm = w / w_sum
    
    # If weights don't sum to 1 initially, the error is calculated with normalized weights
    # The problem requires sum to 1. We optimize raw weights and normalize.
    
    max_rel_err = 0.0
    
    # Test on a grid of redshifts and cosmologies
    # Redshifts: 0.1 to 3.0 (logarithmic or linear)
    z_vals = np.linspace(0.05, 3.0, 15)
    
    for z in z_vals:
        for w0, wa in COSMOLOGIES:
            true_val = true_integral(z, w0, wa)
            if true_val < 1e-10:
                continue
            approx_val = quadrature_approx(x, w_norm, z, w0, wa)
            rel_err = abs(approx_val - true_val) / true_val
            if rel_err > max_rel_err:
                max_rel_err = rel_err
                
    # Add penalty if weights sum is far from 1 (though we normalize, it's good to encourage sum=1 in raw space if we don't normalize, but we do normalize)
    # Actually, if we normalize, the "sum to 1" constraint is satisfied by the normalized vector.
    # The optimization space is effectively 8 x_i and 7 w_i (since sum is fixed).
    # But differential evolution works on fixed dimension.
    # We can just return max_rel_err.
    
    return max_rel_err

def construct():
    # Try to optimize. If it takes too long, fallback to midpoint.
    best_params = None
    best_score = 1e9
    
    try:
        # Bounds for x: [0, 1] for each
        # Bounds for w: [0.01, 1.0] for each (positive)
        bounds = [(0.0, 1.0)] * 8 + [(0.01, 1.0)] * 8
        
        # Use differential evolution for global search
        # maxiter and popsize kept small to fit in time
        # We have ~80s left.
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=50, 
            popsize=15, 
            tol=1e-6, 
            seed=42,
            polish=False  # Skip polish to save time, DE is good enough
        )
        
        if result.fun < best_score:
            best_score = result.fun
            best_params = result.x
            
    except Exception as e:
        pass
        
    # Fallback to midpoint if optimization failed or didn't find anything better
    if best_params is None or best_score > 0.1: # 10% error is bad, midpoint is usually < 1%
         # Use midpoint rule
         n = 8
         x = (np.arange(n) + 0.5) / n
         w = np.full(n, 1.0 / n)
         return {"nodes": x, "weights": w}
        
    # Extract nodes and weights
    x = best_params[:8]
    w = best_params[8:]
    
    # Ensure strict increase and bounds
    # Sort x
    x = np.sort(x)
    # Enforce strict increase by adding small epsilon if needed
    # And ensure within [0,1]
    x = np.clip(x, 1e-6, 1 - 1e-6)
    # Fix duplicates
    for i in range(1, 8):
        if x[i] <= x[i-1]:
            x[i] = x[i-1] + 1e-6
    # Re-clip
    x = np.clip(x, 1e-6, 1 - 1e-6)
    
    # Normalize weights
    w = np.abs(w) # Ensure positive
    w_sum = np.sum(w)
    if w_sum < 1e-12:
        w = np.full(8, 1.0/8.0)
    else:
        w = w / w_sum
        
    return {"nodes": x, "weights": w}
# EVOLVE-BLOCK-END
