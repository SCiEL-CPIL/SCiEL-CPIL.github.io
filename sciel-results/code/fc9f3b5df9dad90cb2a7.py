"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _exact_ground_energy(n, coupling, field):
    """Compute exact ground state energy via diagonalization."""
    dim = 2 ** n
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    def full_op(op_list):
        result = op_list[0]
        for i in range(1, n):
            result = np.kron(op_list[i], result)
        return result
    
    H = np.zeros((dim, dim), dtype=complex)
    for i in range(n - 1):
        ops = [I] * n
        ops[i] = Z
        ops[i+1] = Z
        H -= coupling * full_op(ops)
    for i in range(n):
        ops = [I] * n
        ops[i] = X
        H -= field * full_op(ops)
    
    evals = np.linalg.eigvalsh(H)
    return evals[0].real


def _apply_circuit(n, layers, theta):
    """Apply circuit, return final state and intermediates for backprop."""
    dim = 2 ** n
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0
    
    I2 = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    def full_op(op_list):
        result = op_list[0]
        for i in range(1, n):
            result = np.kron(op_list[i], result)
        return result
    
    # Store unitaries for backprop
    unitaries = []  # list of (type, index, theta_val) or (type, i, j)
    
    for l in range(layers):
        for i in range(n):
            t = theta[l, i]
            Ry = np.array([[np.cos(t/2), -np.sin(t/2)],
                           [np.sin(t/2), np.cos(t/2)]], dtype=complex)
            ops = [I2] * n
            ops[i] = Ry
            U = full_op(ops)
            state = U @ state
            unitaries.append(('Ry', l, i, t))
        
        if l < layers - 1:
            for i in range(n - 1):
                # CZ on qubits i, i+1
                H_cz = np.zeros((dim, dim), dtype=complex)
                ops = [I2] * n
                H_cz += 0.5 * full_op(ops)
                ops = [I2] * n
                ops[i+1] = Z
                H_cz += 0.5 * full_op(ops)
                ops = [I2] * n
                ops[i] = Z
                H_cz += 0.5 * full_op(ops)
                ops = [I2] * n
                ops[i] = Z
                ops[i+1] = Z
                H_cz -= 0.5 * full_op(ops)
                state = H_cz @ state
                unitaries.append(('CZ', i))
    
    return state, unitaries


def _energy_and_grad(params, n, coupling, field, layers):
    """Compute energy and analytic gradient via backprop."""
    theta = params.reshape(layers, n)
    dim = 2 ** n
    
    I2 = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    def full_op(op_list):
        result = op_list[0]
        for i in range(1, n):
            result = np.kron(op_list[i], result)
        return result
    
    # Build Hamiltonian
    H = np.zeros((dim, dim), dtype=complex)
    for i in range(n - 1):
        ops = [I2] * n
        ops[i] = Z
        ops[i+1] = Z
        H -= coupling * full_op(ops)
    for i in range(n):
        ops = [I2] * n
        ops[i] = X
        H -= field * full_op(ops)
    
    # Forward pass
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0
    
    # Store states before each gate for backprop
    # We'll store the state after each Ry gate
    ry_states = []  # state before each Ry application
    ry_indices = []  # (l, i) for each Ry
    
    gate_order = []  # list of ('Ry', l, i) or ('CZ', i)
    
    for l in range(layers):
        for i in range(n):
            ry_states.append(state.copy())
            ry_indices.append((l, i))
            t = theta[l, i]
            Ry = np.array([[np.cos(t/2), -np.sin(t/2)],
                           [np.sin(t/2), np.cos(t/2)]], dtype=complex)
            ops = [I2] * n
            ops[i] = Ry
            U = full_op(ops)
            state = U @ state
            gate_order.append(('Ry', l, i))
        
        if l < layers - 1:
            for i in range(n - 1):
                H_cz = np.zeros((dim, dim), dtype=complex)
                ops = [I2] * n
                H_cz += 0.5 * full_op(ops)
                ops = [I2] * n
                ops[i+1] = Z
                H_cz += 0.5 * full_op(ops)
                ops = [I2] * n
                ops[i] = Z
                H_cz += 0.5 * full_op(ops)
                ops = [I2] * n
                ops[i] = Z
                ops[i+1] = Z
                H_cz -= 0.5 * full_op(ops)
                state = H_cz @ state
                gate_order.append(('CZ', i))
    
    # Energy
    energy = np.vdot(state, H @ state).real
    
    # Backprop for gradient
    # dE/dtheta = 2 * Re(<psi| H |dpsi/dtheta>)
    # Use parameter-shift: dE/dtheta_i = 2 * Re(<psi| H |U_dagger * dU/dtheta * |psi_before>)
    # For Ry: dU/dtheta = -0.5 * sin(theta/2) * X_i + 0.5 * cos(theta/2) * Y_i ... 
    # Actually, dRy/dtheta = -0.5 * [[sin(t/2), -cos(t/2)], [cos(t/2), -sin(t/2)]]
    # = -0.5 * sin(t/2) * X + 0.5 * cos(t/2) * Y ... let me just compute it directly
    
    grad = np.zeros_like(params)
    
    # Backprop: compute adjoint vector
    # v_final = H @ |psi>
    v = H @ state
    
    # Walk backwards through gates
    # For each gate, update v and compute contribution to gradient
    # We need to be careful with the order
    
    # Let's redo: store all intermediate states
    states_after = []
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0
    states_after.append(state.copy())
    
    for l in range(layers):
        for i in range(n):
            t = theta[l, i]
            Ry = np.array([[np.cos(t/2), -np.sin(t/2)],
                           [np.sin(t/2), np.cos(t/2)]], dtype=complex)
            ops = [I2] * n
            ops[i] = Ry
            U = full_op(ops)
            state = U @ state
            states_after.append(state.copy())
        
        if l < layers - 1:
            for i in range(n - 1):
                H_cz = np.zeros((dim, dim), dtype=complex)
                ops = [I2] * n
                H_cz += 0.5 * full_op(ops)
                ops = [I2] * n
                ops[i+1] = Z
                H_cz += 0.5 * full_op(ops)
                ops = [I2] * n
                ops[i] = Z
                H_cz += 0.5 * full_op(ops)
                ops = [I2] * n
                ops[i] = Z
                ops[i+1] = Z
                H_cz -= 0.5 * full_op(ops)
                state = H_cz @ state
                states_after.append(state.copy())
    
    # Now backprop
    v = H @ states_after[-1]
    
    # Walk backwards
    idx = len(states_after) - 1
    for l in range(layers - 1, -1, -1):
        # CZ gates (if any)
        if l < layers - 1:
            for i in range(n - 2, -1, -1):
                H_cz = np.zeros((dim, dim), dtype=complex)
                ops = [I2] * n
                H_cz += 0.5 * full_op(ops)
                ops = [I2] * n
                ops[i+1] = Z
                H_cz += 0.5 * full_op(ops)
                ops = [I2] * n
                ops[i] = Z
                H_cz += 0.5 * full_op(ops)
                ops = [I2] * n
                ops[i] = Z
                ops[i+1] = Z
                H_cz -= 0.5 * full_op(ops)
                v = H_cz @ v
                idx -= 1
        
        # Ry gates
        for i in range(n - 1, -1, -1):
            t = theta[l, i]
            # dRy/dtheta
            dRy = np.array([[-0.5*np.sin(t/2), -0.5*np.cos(t/2)],
                            [0.5*np.cos(t/2), -0.5*np.sin(t/2)]], dtype=complex)
            ops = [I2] * n
            ops[i] = dRy
            dU = full_op(ops)
            
            # Contribution to gradient
            # dE = 2*Re(<v| dU |psi_before>)
            psi_before = states_after[idx - 1]
            grad[l, i] = 2.0 * np.vdot(v, dU @ psi_before).real
            
            # Update v
            Ry = np.array([[np.cos(t/2), -np.sin(t/2)],
                           [np.sin(t/2), np.cos(t/2)]], dtype=complex)
            ops = [I2] * n
            ops[i] = Ry
            U = full_op(ops)
            v = U.conj().T @ v
            idx -= 1
    
    return energy, grad


def _energy_value(params, n, coupling, field, layers):
    theta = params.reshape(layers, n)
    dim = 2 ** n
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0
    
    I2 = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    def full_op(op_list):
        result = op_list[0]
        for i in range(1, n):
            result = np.kron(op_list[i], result)
        return result
    
    for l in range(layers):
        for i in range(n):
            t = theta[l, i]
            Ry = np.array([[np.cos(t/2), -np.sin(t/2)],
                           [np.sin(t/2), np.cos(t/2)]], dtype=complex)
            ops = [I2] * n
            ops[i] = Ry
            state = full_op(ops) @ state
        if l < layers - 1:
            for i in range(n - 1):
                H_cz = np.zeros((dim, dim), dtype=complex)
                ops = [I2] * n
                H_cz += 0.5 * full_op(ops)
                ops = [I2] * n
                ops[i+1] = Z
                H_cz += 0.5 * full_op(ops)
                ops = [I2] * n
                ops[i] = Z
                H_cz += 0.5 * full_op(ops)
                ops = [I2] * n
                ops[i] = Z
                ops[i+1] = Z
                H_cz -= 0.5 * full_op(ops)
                state = H_cz @ state
    
    # Energy
    H = np.zeros((dim, dim), dtype=complex)
    for i in range(n - 1):
        ops = [I2] * n
        ops[i] = Z
        ops[i+1] = Z
        H -= coupling * full_op(ops)
    for i in range(n):
        ops = [I2] * n
        ops[i] = X
        H -= field * full_op(ops)
    
    return np.vdot(state, H @ state).real


def angles(n, coupling, field, layers):
    """Optimize circuit parameters to minimize mean energy gap per spin."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    e_exact = _exact_ground_energy(n, coupling, field)
    
    def objective(params):
        e_circuit = _energy_value(params, n, coupling, field, layers)
        return (e_circuit - e_exact) / n
    
    def objective_grad(params):
        e_circuit, grad_e = _energy_and_grad(params, n, coupling, field, layers)
        return (e_circuit - e_exact) / n, grad_e / n
    
    best_theta = None
    best_val = np.inf
    
    # Multi-start
    n_restarts = 30
    rng = np.random.RandomState(42)
    
    for restart in range(n_restarts):
        if time.time() > deadline - 15:
            break
        
        if restart == 0:
            init = np.zeros(layers * n)
            init[0:n] = 0.2
            init[n:2*n] = 0.6
            init[2*n:3*n] = 1.2
        elif restart == 1:
            init = np.zeros(layers * n)
            init[2*n:3*n] = np.pi / 2
        elif restart == 2:
            init = np.full(layers * n, np.pi / 4)
        else:
            init = rng.uniform(-np.pi, np.pi, layers * n)
        
        try:
            result = minimize(
                objective,
                init,
                jac=objective_grad,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_theta = result.x.reshape(layers, n)
        except Exception:
            pass
    
    if best_theta is None:
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        return theta
    
    return best_theta

# EVOLVE-BLOCK-END
