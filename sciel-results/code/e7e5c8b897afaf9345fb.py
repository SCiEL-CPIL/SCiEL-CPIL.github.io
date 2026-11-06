"""Quantum information: fixed-depth QAOA parameter design."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

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

def _apply_x_rotation(psi, n, beta):
    """Apply exp(-i beta * sum X) to state vector using efficient matrix ops."""
    if abs(beta) < 1e-9:
        return psi
    c = np.cos(beta)
    s = np.sin(beta)
    # Precompute rotation matrix for single qubit: [[c, -1j*s], [-1j*s, c]]
    # Apply to each qubit using reshape and einsum
    for k in range(n):
        shape = (2**(n-k-1), 2, 2**k)
        psi_r = psi.reshape(shape)
        # New state: c*|0> - i*s*|1> for each qubit
        new_psi = np.empty_like(psi_r)
        new_psi[:, 0, :] = c * psi_r[:, 0, :] - 1j * s * psi_r[:, 1, :]
        new_psi[:, 1, :] = -1j * s * psi_r[:, 0, :] + c * psi_r[:, 1, :]
        psi = new_psi.reshape(-1)
    return psi

def _expected_cut(n, cuts, gamma1, beta1, gamma2, beta2):
    """
    Compute expected cut for 2-layer QAOA.
    Optimized state-vector simulation.
    """
    psi = np.ones(2**n, dtype=np.complex128) / np.sqrt(2**n)
    
    # Layer 1
    psi = psi * np.exp(-1j * gamma1 * cuts)
    psi = _apply_x_rotation(psi, n, beta1)
    
    # Layer 2
    psi = psi * np.exp(-1j * gamma2 * cuts)
    psi = _apply_x_rotation(psi, n, beta2)

    probs = np.abs(psi)**2
    return np.sum(probs * cuts)

def _optimize_4params(n, cuts):
    """
    Optimize 4 QAOA parameters using a combination of global and local search.
    """
    
    # Define objective for minimization (negative expected cut)
    def obj(x):
        if _remaining() < 2.0:
            return 1e10
        g1, b1, g2, b2 = x
        val = _expected_cut(n, cuts, g1, b1, g2, b2)
        return -val

    # Bounds
    lower = [0.0, 0.0, 0.0, 0.0]
    upper = [np.pi, np.pi/2, np.pi, np.pi/2]
    
    best_x = None
    best_val = np.inf
    
    # Step 1: Global search with differential evolution if time permits
    if _remaining() > 30.0:
        try:
            res_de = differential_evolution(
                obj, 
                bounds=list(zip(lower, upper)),
                seed=42,
                maxiter=100,
                popsize=10,
                tol=1e-6,
                polish=False
            )
            if res_de.fun < best_val:
                best_val = res_de.fun
                best_x = res_de.x
        except Exception:
            pass
    
    # Step 2: Multi-start local search from diverse initial points
    # Physics-informed initializations based on QAOA literature for MaxCut
    inits = [
        [0.5, 0.5, 0.5, 0.5],
        [0.8, 0.4, 0.6, 0.4],
        [1.0, 0.3, 0.5, 0.5],
        [0.3, 0.6, 0.7, 0.3],
        [0.5, 0.2, 0.8, 0.6],
        [np.pi/4, np.pi/4, np.pi/4, np.pi/4],
        [0.2, 0.8, 0.9, 0.2],
        [1.2, 0.1, 0.4, 0.7],
        # Additional physics-informed points
        [0.7, 0.5, 0.3, 0.7],
        [0.4, 0.3, 0.9, 0.4],
        [0.6, 0.6, 0.4, 0.6],
        [1.1, 0.2, 0.7, 0.5],
    ]
    
    # Add the DE result as an initial point if we have one
    if best_x is not None:
        inits.append(best_x)
    
    for init in inits:
        if _remaining() < 5.0:
            break
        try:
            res = minimize(
                obj, 
                init, 
                method='Nelder-Mead', 
                bounds=list(zip(lower, upper)),
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            continue
            
    # Final high-precision polish with tighter tolerance
    if best_x is not None and _remaining() > 8.0:
        try:
            res_polish = minimize(
                obj, 
                best_x, 
                method='Nelder-Mead', 
                bounds=list(zip(lower, upper)),
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            if res_polish.fun < best_val:
                best_val = res_polish.fun
                best_x = res_polish.x
        except Exception:
            pass
        
        # Additional polish if time permits
        if best_x is not None and _remaining() > 5.0:
            try:
                res_polish2 = minimize(
                    obj, 
                    best_x, 
                    method='Powell',
                    bounds=list(zip(lower, upper)),
                    options={'maxiter': 200, 'xtol': 1e-14, 'ftol': 1e-14}
                )
                if res_polish2.fun < best_val:
                    best_val = res_polish2.fun
                    best_x = res_polish2.x
            except Exception:
                pass
        
    if best_x is None:
        best_x = np.array([0.5, 0.5, 0.5, 0.5])
    
    return best_x

def angles(n, edges, p):
    if _remaining() < 1.0:
        return np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float64)
    
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
