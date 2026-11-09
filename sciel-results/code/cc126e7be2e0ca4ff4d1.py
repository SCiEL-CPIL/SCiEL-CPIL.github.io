# EVOLVE-BLOCK-START
import numpy as np
import time

def _circuit_state(n, layers, params):
    """Compute the state produced by the circuit using tensor operations.
    params: array of shape (layers, n)
    Qubit i corresponds to bit i (little-endian).
    """
    dim = 1 << n
    # State as tensor: shape (2, 2, ..., 2) with n axes
    # Index: state[i0, i1, ..., i_{n-1}] where iq is the value of qubit q
    state = np.zeros((2,) * n, dtype=np.complex128)
    state[(0,) * n] = 1.0  # |0>^n
    
    for layer in range(layers):
        # Apply Ry to every qubit
        for q in range(n):
            theta = params[layer, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            # Ry = [[c, -s], [s, c]] applied to qubit q
            # new_state[..., 0, ...] = c * old_state[..., 0, ...] - s * old_state[..., 1, ...]
            # new_state[..., 1, ...] = s * old_state[..., 0, ...] + c * old_state[..., 1, ...]
            # Move qubit q to front for easy slicing
            state = np.moveaxis(state, q, 0)
            new_state = np.empty_like(state)
            new_state[0] = c * state[0] - s * state[1]
            new_state[1] = s * state[0] + c * state[1]
            state = np.moveaxis(new_state, 0, q)
        
        # Apply CZ on adjacent pairs (except after last layer)
        if layer < layers - 1:
            for q in range(n - 1):
                # CZ flips sign when both qubits are 1
                # CZ = diag(1, 1, 1, -1) on (q, q+1)
                # Move q and q+1 to front
                state = np.moveaxis(state, q, 0)
                state = np.moveaxis(state, q + 1, 1)
                # Flip sign of state[1, 1, ...]
                state[1, 1] = -state[1, 1]
                # Move back
                state = np.moveaxis(state, 1, q + 1)
                state = np.moveaxis(state, 0, q)
    
    return state.flatten()

def _hamiltonian(n, coupling, field):
    """Build the Hamiltonian matrix H = -coupling * sum Z_i Z_{i+1} - field * sum X_i"""
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.complex128)
    
    # ZZ terms: diagonal
    for i in range(n - 1):
        for b in range(dim):
            bi = (b >> i) & 1
            bj = (b >> (i + 1)) & 1
            zval = (1 - 2 * bi) * (1 - 2 * bj)
            H[b, b] += -coupling * zval
    
    # X terms: off-diagonal
    for i in range(n):
        for b in range(dim):
            b_flip = b ^ (1 << i)
            if b < b_flip:
                H[b, b_flip] += -field
                H[b_flip, b] += -field
    
    return H

def _energy(state, H):
    """Compute <state|H|state>"""
    return np.real(np.vdot(state, H @ state))

def _exact_ground_energy(n, coupling, field):
    """Compute exact ground state energy by diagonalizing H"""
    H = _hamiltonian(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    return eigvals[0]

def _objective(params, n, layers, H, e_ground):
    """Compute objective value."""
    state = _circuit_state(n, layers, params)
    e_circuit = _energy(state, H)
    return (e_circuit - e_ground) / n

def _objective_grad(params, n, layers, H, e_ground):
    """Compute objective and gradient using parameter-shift rule.
    For Ry(theta), d/dtheta <psi|H|psi> = (E(theta+pi/2) - E(theta-pi/2)) / 2
    """
    state = _circuit_state(n, layers, params)
    e_circuit = _energy(state, H)
    obj_val = (e_circuit - e_ground) / n
    
    grad = np.zeros_like(params)
    shift = np.pi / 2.0
    
    for layer in range(layers):
        for q in range(n):
            p_plus = params.copy()
            p_minus = params.copy()
            p_plus[layer, q] += shift
            p_minus[layer, q] -= shift
            
            e_plus = _energy(_circuit_state(n, layers, p_plus), H)
            e_minus = _energy(_circuit_state(n, layers, p_minus), H)
            
            grad[layer, q] = (e_plus - e_minus) / (2.0 * n)
    
    return obj_val, grad

def angles(n, coupling, field, layers):
    """Optimize circuit parameters to minimize mean energy gap per spin."""
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s kill
    
    # Precompute ground energy and Hamiltonian
    e_ground = _exact_ground_energy(n, coupling, field)
    H = _hamiltonian(n, coupling, field)
    
    def obj(params):
        return _objective(params, n, layers, H, e_ground)
    
    def obj_grad(params):
        return _objective_grad(params, n, layers, H, e_ground)
    
    # Initial guess
    x0 = np.zeros((layers, n))
    best_params = x0.copy()
    best_val = obj(x0)
    
    np.random.seed(12345)
    
    n_params = layers * n
    n_restarts = 20
    
    for restart in range(n_restarts):
        if time.time() > deadline - 10:
            break
        
        if restart == 0:
            x_init = x0.copy()
        elif restart == 1:
            x_init = np.full((layers, n), np.pi / 8)
        elif restart == 2:
            x_init = np.full((layers, n), -np.pi / 8)
        elif restart == 3:
            x_init = np.full((layers, n), np.pi / 4)
        elif restart == 4:
            x_init = np.full((layers, n), -np.pi / 4)
        elif restart == 5:
            # Alternating pattern
            x_init = np.tile(np.array([np.pi/6, -np.pi/6])[:n], (layers, 1))
        elif restart == 6:
            x_init = np.tile(np.array([-np.pi/6, np.pi/6])[:n], (layers, 1))
        else:
            x_init = np.random.uniform(-np.pi/2, np.pi/2, (layers, n))
        
        x_flat = x_init.flatten()
        
        def obj_flat(xf):
            return obj(xf.reshape((layers, n)))
        
        def obj_grad_flat(xf):
            val, grad = obj_grad(xf.reshape((layers, n)))
            return val, grad.flatten()
        
        # L-BFGS-B optimization
        try:
            from scipy.optimize import minimize
            res = minimize(obj_grad_flat, x_flat, jac=True, method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 500})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.reshape((layers, n))
        except Exception:
            pass
        
        # Check time
        if time.time() > deadline - 5:
            break
    
    # Final polish with more iterations if time permits
    if time.time() < deadline - 2:
        try:
            x_flat = best_params.flatten()
            from scipy.optimize import minimize
            res = minimize(obj_grad_flat, x_flat, jac=True, method='L-BFGS-B',
                          options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 800})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.reshape((layers, n))
        except Exception:
            pass
    
    return best_params
# EVOLVE-BLOCK-END
