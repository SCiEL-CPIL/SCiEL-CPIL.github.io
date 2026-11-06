"""Quantum information: fixed-depth QAOA parameter design."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = 150.0
_SAFETY = 10.0
_T0 = time.time()

def _remaining():
    return _DEADLINE - _SAFETY - (time.time() - _T0)

def _precompute_cuts(n, edges):
    """Precompute cut values for all 2^n basis states."""
    num_states = 2**n
    cuts = np.zeros(num_states, dtype=np.float64)
    states = np.arange(num_states, dtype=np.int64)
    for u, v in edges:
        bits_u = (states >> u) & 1
        bits_v = (states >> v) & 1
        mask = bits_u != bits_v
        cuts[mask] += 1
    return cuts

def _apply_beta(psi, beta, n):
    """Apply exp(-i beta sum_k X_k) to state vector psi."""
    c = np.cos(beta)
    s = np.sin(beta)
    for k in range(n):
        shape = (2**(n-k-1), 2, 2**k)
        psi_reshaped = psi.reshape(shape)
        p0 = psi_reshaped[:, 0, :].copy()
        p1 = psi_reshaped[:, 1, :].copy()
        psi_reshaped[:, 0, :] = c * p0 - 1j * s * p1
        psi_reshaped[:, 1, :] = -1j * s * p0 + c * p1
        psi = psi_reshaped.reshape(-1)
    return psi

def _expected_cut_4params(n, cuts, gamma1, beta1, gamma2, beta2):
    """Compute expected cut for 2-layer QAOA with 4 independent parameters."""
    psi = np.ones(2**n, dtype=np.complex128) / np.sqrt(2**n)
    
    # Layer 1: Gamma then Beta
    phase1 = np.exp(-1j * gamma1 * cuts)
    psi = psi * phase1
    psi = _apply_beta(psi, beta1, n)
    
    # Layer 2: Gamma then Beta
    phase2 = np.exp(-1j * gamma2 * cuts)
    psi = psi * phase2
    psi = _apply_beta(psi, beta2, n)
    
    # Expected cut
    probs = np.abs(psi)**2
    return np.sum(probs * cuts)

def _optimize_4params(n, cuts, max_iter=300):
    """Optimize 4 QAOA parameters using multi-start Nelder-Mead."""
    best_x = np.array([0.5, 0.5, 0.5, 0.5])
    best_val = -_expected_cut_4params(n, cuts, *best_x)
    
    # Define bounds for parameters: gamma in [0, pi], beta in [0, 2*pi]
    # QAOA angles are periodic.
    lower = np.array([0.0, 0.0, 0.0, 0.0])
    upper = np.array([np.pi, 2*np.pi, np.pi, 2*np.pi])
    
    # Use a fixed number of random restarts, but stop if time runs out
    # We have ~25s per graph. Each NM run is fast for n<=10.
    # Let's try 10-20 restarts.
    
    def obj(x):
        if _remaining() < 2.0:
            return 1e10
        g1, b1, g2, b2 = x
        return -_expected_cut_4params(n, cuts, g1, b1, g2, b2)
    
    # Try multiple starting points
    rng = np.random.default_rng(42)
    n_restarts = 15
    
    for i in range(n_restarts):
        if _remaining() < 5.0:
            break
        
        if i == 0:
            x0 = best_x.copy()
        else:
            x0 = rng.uniform(lower, upper)
        
        try:
            res = minimize(obj, x0, method='Nelder-Mead', 
                           options={'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.fun < 1e9:  # Valid result
                if res.fun < best_val:
                    best_val = res.fun
                    best_x = res.x.copy()
        except Exception:
            pass
    
    return best_x

def angles(n, edges, p):
    if _remaining() < 1.0:
        return np.array([[0.35, 0.25], [0.35, 0.25]], dtype=np.float64)
    
    cuts = _precompute_cuts(n, edges)
    
    # Optimize 4 parameters for p=2
    opt = _optimize_4params(n, cuts)
    
    # Return as (p, 2) array: rows are (gamma, beta)
    result = np.array([
        [opt[0], opt[1]],  # gamma_1, beta_1
        [opt[2], opt[3]]   # gamma_2, beta_2
    ], dtype=np.float64)
    
    return result

# EVOLVE-BLOCK-END
