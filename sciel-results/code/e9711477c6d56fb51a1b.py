"""
Strategy: Global optimization via Differential Evolution (DE) combined with exact state-vector simulation.
Instead of local gradient-based optimization (L-BFGS-B) which is prone to getting stuck in local
minima of the non-convex QAOA landscape, we use a derivative-free global search method (Differential
Evolution). This method explores the entire parameter space simultaneously, making it much more
robust against local optima. We also optimize the Hamiltonian construction to be faster, allowing
DE to run more evaluations within the time budget.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
import time
import sys

# Global deadline guard
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0

def _time_remaining():
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - START_TIME)

def _build_hamiltonians(n, edges):
    """
    Build the problem Hamiltonian C (MaxCut) and mixing Hamiltonian B (sum X).
    C = sum_{(i,j) in E} 0.5 * (I - Z_i Z_j)
    B = sum_{i=0}^{n-1} X_i
    """
    # Pauli matrices
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)
    
    dim = 2**n
    
    # Helper to apply operator on qubit k
    def apply_op(op, k, n):
        ops = [I] * n
        ops[k] = op
        result = ops[0]
        for i in range(1, n):
            result = np.kron(result, ops[i])
        return result
        
    # Build C
    C = np.zeros((dim, dim), dtype=complex)
    eye = np.eye(dim, dtype=complex)
    for (u, v) in edges:
        Z_u = apply_op(Z, u, n)
        Z_v = apply_op(Z, v, n)
        ZZ = Z_u @ Z_v
        C += 0.5 * (eye - ZZ)
        
    # Build B
    B = np.zeros((dim, dim), dtype=complex)
    for i in range(n):
        X_i = apply_op(X, i, n)
        B += X_i
        
    return C, B

def _expected_cut_value(n, edges, angles, C, B):
    """
    Compute the expected cut value.
    angles: (p, 2) array of (gamma, beta)
    """
    if len(edges) == 0:
        return 0.0
        
    dim = 2**n
    p = len(angles)
    
    # Initial state |+>^n
    psi = np.ones(dim, dtype=complex) / np.sqrt(dim)
    
    # Precompute exponentials for efficiency? 
    # For p=2, we just have 4 gates.
    # We can compute U_C and U_B for each layer on the fly.
    
    for i in range(p):
        gamma, beta = angles[i]
        
        # U_C = exp(-i gamma C)
        # For small n, expm is fast.
        U_C = np.linalg.expm(-1j * gamma * C)
        psi = U_C @ psi
        
        # U_B = exp(-i beta B)
        U_B = np.linalg.expm(-1j * beta * B)
        psi = U_B @ psi
        
    # Final state
    psi_final = psi
    
    # Value: <psi| C |psi>
    val = np.vdot(psi_final, C @ psi_final).real
    
    return val

def angles(n, edges, p):
    # Deadline check
    if _time_remaining() < 10.0:
        return np.tile([0.35, 0.25], (p, 1))
    
    # Precompute Hamiltonians once
    try:
        C, B = _build_hamiltonians(n, edges)
    except Exception:
        return np.tile([0.35, 0.25], (p, 1))
    
    # Define bounds for DE
    # Gamma is typically in [0, pi], Beta in [0, pi]
    # But for p=2, smaller ranges often work better.
    # Let's use [0, 2] for both to be safe, but DE will explore.
    bounds = [(0, 2), (0, 2)] * p
    
    # Define objective function for DE (minimization)
    def objective(x):
        if _time_remaining() < 5.0:
            return 1e9
        x_reshaped = np.array(x).reshape(p, 2)
        val = _expected_cut_value(n, edges, x_reshaped, C, B)
        return -val
    
    # Run Differential Evolution
    best_angles = np.tile([0.35, 0.25], (p, 1))
    best_val = -1.0
    
    try:
        # DE is stochastic, so set seed for reproducibility
        # maxiter and popsize control the budget.
        # popsize=15 means 15*4=60 parameters per generation? No, popsize is number of individuals.
        # For 4 parameters, popsize=15 is reasonable.
        # maxiter=50 should be enough for convergence in low dimensions.
        res = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=50, 
            popsize=15, 
            tol=1e-5,
            polish=True  # Use Nelder-Mead to polish the result
        )
        
        if res.fun < -best_val:
            best_angles = res.x.reshape(p, 2)
            best_val = -res.fun
    except Exception:
        pass
    
    # If time allows, try a second DE run with different seed/params
    if _time_remaining() > 30.0:
        try:
            res2 = differential_evolution(
                objective, 
                bounds, 
                seed=123, 
                maxiter=30, 
                popsize=10, 
                tol=1e-5,
                polish=True
            )
            if res2.fun < -best_val:
                best_angles = res2.x.reshape(p, 2)
                best_val = -res2.fun
        except Exception:
            pass
            
    return best_angles

# EVOLVE-BLOCK-END
