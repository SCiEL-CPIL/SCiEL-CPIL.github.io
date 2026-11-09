import numpy as np
import time
from scipy.optimize import differential_evolution

def _build_hamiltonian(n, coupling, field):
    """
    Build the Hamiltonian H = -coupling * sum Z_i Z_{i+1} - field * sum X_i
    for an open chain of n qubits.
    Qubit i corresponds to bit i (little-endian).
    """
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=complex)
    
    # ZZ terms: -coupling * Z_i Z_{i+1}
    for i in range(n - 1):
        for b in range(dim):
            bit_i = (b >> i) & 1
            bit_ip1 = (b >> (i + 1)) & 1
            eigenval = (-1) ** (bit_i + bit_ip1)
            H[b, b] -= coupling * eigenval
            
    # X terms: -field * sum X_i
    for i in range(n):
        for b in range(dim):
            b_flipped = b ^ (1 << i)
            H[b_flipped, b] -= field
            
    return H

def _ground_energy(n, coupling, field):
    """Compute exact ground energy."""
    H = _build_hamiltonian(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    return eigvals[0].real

def _simulate_state_fast(params, n, layers):
    """
    Fast vectorized simulation of the circuit.
    params: (layers, n)
    """
    dim = 1 << n
    # Initial state |0...0> is index 0
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0
    
    # Pre-compute indices for CZ operations to avoid masking overhead in loop
    # Actually, for n=6, dim=64, it's tiny. Just do it simply.
    
    for l in range(layers):
        # Apply Ry to all qubits
        for q in range(n):
            theta = params[l, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            
            # Reshape to tensor
            # state shape: (2, 2, ..., 2)
            # We want to apply gate on axis q
            # Move axis q to 0
            st = np.moveaxis(state, q, 0)
            
            # Apply rotation
            # R = [[c, -s], [s, c]]
            # new_st[0] = c*st[0] - s*st[1]
            # new_st[1] = s*st[0] + c*st[1]
            new_st = np.empty_like(st)
            new_st[0] = c * st[0] - s * st[1]
            new_st[1] = s * st[0] + c * st[1]
            
            state = np.moveaxis(new_st, 0, q)
            
        # Apply CZ between layers
        if l < layers - 1:
            for q in range(n - 1):
                # CZ flips sign if both q and q+1 are 1
                # Indices where bit q is 1 AND bit q+1 is 1
                # We can precompute these masks or just use bitwise ops on arange
                idx = np.arange(dim)
                mask_q = (idx >> q) & 1
                mask_q1 = (idx >> (q+1)) & 1
                both_one = (mask_q == 1) & (mask_q1 == 1)
                state[both_one] *= -1
                
    return state

def _energy_only(flat_params, n, layers, H, E0):
    """
    Calculate energy gap for a given set of parameters.
    """
    params = flat_params.reshape((layers, n))
    psi = _simulate_state_fast(params, n, layers)
    # Energy = <psi|H|psi>
    h_psi = H @ psi
    energy = np.vdot(psi, h_psi).real
    return energy - E0

def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    E0 = _ground_energy(n, coupling, field)
    H = _build_hamiltonian(n, coupling, field)
    
    # Bounds for angles: [-pi, pi]
    bounds = [(-np.pi, np.pi)] * (layers * n)
    
    # Use Differential Evolution for global search
    # This is robust to local minima.
    # For small n, the function is fast, so we can afford many iterations.
    
    # Strategy: 
    # 1. Run DE with a reasonable maxiter.
    # 2. If time allows, polish the result with L-BFGS-B (optional, but DE polish=True does local optimization).
    
    def objective(flat_params):
        if time.time() > deadline:
            return 1e10 # Penalize timeout
        return _energy_only(flat_params, n, layers, H, E0)
        
    try:
        # initial_popsize=15, maxiter=50 is a good start for 12D problem
        # tol=1e-6
        res = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=100, 
            tol=1e-6,
            polish=True, # Uses L-BFGS-B on the best result
            init='latinhypercube'
        )
        
        if res.success or res.fun < 1e9:
            return res.x.reshape((layers, n))
            
    except Exception:
        pass
        
    # Fallback: Simple random search if DE fails or times out
    # But DE should be fine.
    
    # If we ran out of time or DE failed, return zeros or best found
    # We don't store best found in DE easily if it throws.
    # Let's do a quick random check if we are close to deadline? No, just return zeros.
    
    return np.zeros((layers, n))
