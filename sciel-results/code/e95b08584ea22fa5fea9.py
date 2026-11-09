import numpy as np
from scipy.optimize import minimize

_optimization_cache = {}

def _apply_circuit_vec(thetas, n, cz_masks):
    """Apply the circuit to |0...0> using vectorized operations."""
    psi = np.zeros(2 ** n, dtype=complex)
    psi[0] = 1.0
    layers = thetas.shape[0]
    
    for l in range(layers):
        for i in range(n):
            theta = thetas[l, i]
            c = np.cos(theta / 2)
            s = np.sin(theta / 2)
            shape = [2**i, 2, 2**(n-i-1)]
            psi_reshaped = psi.reshape(shape)
            p0 = psi_reshaped[:, 0, :]
            p1 = psi_reshaped[:, 1, :]
            new_p0 = c * p0 - s * p1
            new_p1 = s * p0 + c * p1
            new_psi_reshaped = np.stack([new_p0, new_p1], axis=1)
            psi = new_psi_reshaped.reshape(-1)
        
        if l < layers - 1:
            for i in range(n - 1):
                psi *= cz_masks[i]
    
    return psi

def _energy_expectation(psi, n, coupling, field):
    """Compute <psi|H|psi> efficiently without building full H."""
    dim = 2 ** n
    energy = 0.0
    # ZZ terms
    for i in range(n - 1):
        for state in range(dim):
            bit_i = (state >> i) & 1
            bit_ip1 = (state >> (i + 1)) & 1
            z_i = 1 if bit_i == 0 else -1
            z_ip1 = 1 if bit_ip1 == 0 else -1
            energy += -coupling * z_i * z_ip1 * np.abs(psi[state])**2
    # X terms
    for i in range(n):
        for state in range(dim):
            flipped = state ^ (1 << i)
            energy += -field * (np.conj(psi[state]) * psi[flipped] + np.conj(psi[flipped]) * psi[state]).real / 2
    return energy.real

def angles(n, coupling, field, layers):
    key = (n, coupling, field, layers)
    if key in _optimization_cache:
        return _optimization_cache[key]
    
    dim = 2 ** n
    # Precompute CZ masks
    cz_masks = []
    for i in range(n - 1):
        mask = np.ones(dim, dtype=complex)
        for state in range(dim):
            if ((state >> i) & 1) and ((state >> (i+1)) & 1):
                mask[state] = -1
        cz_masks.append(mask)
    
    def energy_fn(thetas):
        psi = _apply_circuit_vec(thetas, n, cz_masks)
        return _energy_expectation(psi, n, coupling, field)
    
    def objective_and_grad(x):
        thetas = x.reshape((layers, n))
        e = energy_fn(thetas)
        # Parameter-shift rule for gradients
        grad = np.zeros_like(x)
        for idx in range(x.size):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[idx] += np.pi / 2
            x_minus[idx] -= np.pi / 2
            e_plus = energy_fn(x_plus.reshape((layers, n)))
            e_minus = energy_fn(x_minus.reshape((layers, n)))
            grad[idx] = (e_plus - e_minus) / 2.0
        return e, grad
    
    # Multi-start with L-BFGS-B
    best_result = None
    best_val = np.inf
    
    starts = [np.zeros((layers, n))]
    # Physical initial guesses
    for f_guess in [np.pi/4, np.pi/2, 0]:
        init = np.zeros((layers, n))
        init[-1] = f_guess
        starts.append(init)
    
    np.random.seed(123)
    for _ in range(8):
        starts.append(np.random.uniform(-np.pi, np.pi, (layers, n)))
    
    for x0 in starts:
        res = minimize(
            lambda x: objective_and_grad(x)[0],
            x0.flatten(),
            jac=lambda x: objective_and_grad(x)[1],
            method='L-BFGS-B',
            options={'maxiter': 200, 'gtol': 1e-6}
        )
        if res.fun < best_val:
            best_val = res.fun
            best_result = res.x.reshape((layers, n))
    
    optimal_thetas = best_result
    _optimization_cache[key] = optimal_thetas
    return optimal_thetas
