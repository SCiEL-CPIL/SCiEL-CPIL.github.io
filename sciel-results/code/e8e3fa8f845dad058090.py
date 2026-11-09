import numpy as np
import time
from scipy.optimize import minimize

def _build_hamiltonian(n, coupling, field):
    """
    Build the Hamiltonian H = -coupling * sum Z_i Z_{i+1} - field * sum X_i
    for an open chain of n qubits.
    Qubit i corresponds to bit i (little-endian).
    """
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=complex)
    
    # ZZ terms: -coupling * Z_i Z_{i+1}
    # Z is diagonal with entries 1, -1. Z_i Z_{i+1} is diagonal.
    for i in range(n - 1):
        for b in range(dim):
            bit_i = (b >> i) & 1
            bit_ip1 = (b >> (i + 1)) & 1
            eigenval = (-1) ** (bit_i + bit_ip1)
            H[b, b] -= coupling * eigenval
            
    # XX terms: -field * sum X_i
    # X flips the bit.
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

def _simulate_state(params, n, layers):
    """
    Simulate the circuit and return the final state vector.
    params: array of shape (layers, n)
    Returns: state vector of size 2**n
    """
    dim = 1 << n
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0
    
    for l in range(layers):
        # Apply Ry to all qubits
        for q in range(n):
            theta = params[l, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            
            # Efficient Ry application using reshape and moveaxis
            state_tensor = state.reshape([2] * n)
            # Move axis q to the front for easier manipulation
            state_tensor = np.moveaxis(state_tensor, q, 0)
            
            new_tensor = np.empty_like(state_tensor)
            new_tensor[0] = c * state_tensor[0] - s * state_tensor[1]
            new_tensor[1] = s * state_tensor[0] + c * state_tensor[1]
            
            state_tensor = np.moveaxis(new_tensor, 0, q)
            state = state_tensor.reshape(dim)
            
        # Apply CZ between layers (except after last layer)
        if l < layers - 1:
            for q in range(n - 1):
                mask1 = 1 << q
                mask2 = 1 << (q + 1)
                indices = np.arange(dim)
                mask = (indices & mask1) != 0
                mask2_bool = (indices & mask2) != 0
                both = mask & mask2_bool
                state[both] *= -1
                
    return state

def _energy_and_grad_analytic(params, n, layers, H, E0):
    """
    Compute energy and analytic gradient using backpropagation.
    """
    dim = 1 << n
    
    # 1. Forward pass: store states after each Ry layer and CZ layer
    # We need states after every Ry application to compute gradients.
    # Structure:
    # state_0 (initial)
    # Ry_0_0, Ry_0_1, ... Ry_0_{n-1}
    # CZ_0
    # Ry_1_0, ...
    # CZ_1
    # Ry_2_0, ...
    # Final State
    
    # To save memory/time, we can store states after each qubit rotation.
    # Total states to store: layers * n + 1 (initial)
    # But we only need the state *before* each Ry to compute the gradient contribution.
    
    states = []
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0
    states.append(state.copy())
    
    for l in range(layers):
        for q in range(n):
            theta = params[l, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            
            state_tensor = state.reshape([2] * n)
            state_tensor = np.moveaxis(state_tensor, q, 0)
            
            new_tensor = np.empty_like(state_tensor)
            new_tensor[0] = c * state_tensor[0] - s * state_tensor[1]
            new_tensor[1] = s * state_tensor[0] + c * state_tensor[1]
            
            state_tensor = np.moveaxis(new_tensor, 0, q)
            state = state_tensor.reshape(dim)
            states.append(state.copy())
            
        if l < layers - 1:
            for q in range(n - 1):
                mask1 = 1 << q
                mask2 = 1 << (q + 1)
                indices = np.arange(dim)
                mask = (indices & mask1) != 0
                mask2_bool = (indices & mask2) != 0
                both = mask & mask2_bool
                state[both] *= -1
            states.append(state.copy())
            
    # Final state is states[-1]
    psi = states[-1]
    energy = np.vdot(psi, H @ psi).real - E0
    
    # 2. Backward pass
    # We want dE/dtheta_{l,q}.
    # E = <psi|H|psi>
    # dE = <dpsi|H|psi> + <psi|H|dpsi> = 2 Re[<dpsi|H|psi>]
    # Let |chi> = H|psi>. Then dE = 2 Re[<dpsi|chi>]
    
    chi = H @ psi
    
    grad = np.zeros((layers, n), dtype=float)
    
    # We propagate the gradient backwards.
    # Let A_{l,q} be the state *after* applying Ry_{l,q}.
    # Let B_{l,q} be the state *before* applying Ry_{l,q}.
    # |A> = R_y(theta) |B>
    # d|A> = (d/dtheta R_y) |B>
    # The contribution to the gradient from this gate is 2 Re[ <dA|chi_current> ]
    # where chi_current is the adjoint state at that point.
    
    # Initialize adjoint state at the end
    adj_state = chi.copy()
    
    # We need to reverse the operations.
    # The list of states we stored:
    # idx 0: |0>
    # idx 1: after Ry[0,0]
    # idx 2: after Ry[0,1]
    # ...
    # idx n: after Ry[0,n-1]
    # idx n+1: after CZ[0]
    # idx n+2: after Ry[1,0]
    # ...
    
    # It's easier to iterate backwards through the layers and qubits.
    # We need to know the state BEFORE each gate to compute the derivative.
    # Let's reconstruct the indices or just keep a list of "pre-states".
    
    # Let's re-do the forward pass to collect pre-states in a list corresponding to each gate.
    pre_states = []
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0
    
    for l in range(layers):
        for q in range(n):
            pre_states.append(state.copy())
            theta = params[l, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            
            state_tensor = state.reshape([2] * n)
            state_tensor = np.moveaxis(state_tensor, q, 0)
            
            new_tensor = np.empty_like(state_tensor)
            new_tensor[0] = c * state_tensor[0] - s * state_tensor[1]
            new_tensor[1] = s * state_tensor[0] + c * state_tensor[1]
            
            state_tensor = np.moveaxis(new_tensor, 0, q)
            state = state_tensor.reshape(dim)
            
        if l < layers - 1:
            for q in range(n - 1):
                mask1 = 1 << q
                mask2 = 1 << (q + 1)
                indices = np.arange(dim)
                mask = (indices & mask1) != 0
                mask2_bool = (indices & mask2) != 0
                both = mask & mask2_bool
                state[both] *= -1
                
    # Now, backpropagate.
    # adj_state is currently H|psi_final>
    # We need to apply the adjoint of the gates in reverse order.
    
    # Reverse loop over layers
    for l in range(layers - 1, -1, -1):
        # Reverse loop over qubits
        for q in range(n - 1, -1, -1):
            # Current adj_state is the adjoint vector at the point AFTER Ry[l,q]
            # We need to compute the gradient for Ry[l,q]
            # d/dtheta Ry(theta) = -0.5 * (sin(theta/2) * X + cos(theta/2) * Y) ? 
            # Ry(theta) = cos(t/2) I - i sin(t/2) Y
            # d/dtheta Ry = -0.5 sin(t/2) I - i 0.5 cos(t/2) Y
            # Wait, standard Ry = [[cos, -sin], [sin, cos]] for real? No, complex.
            # Ry(theta) = exp(-i theta Y / 2) = [[cos(t/2), -sin(t/2)], [sin(t/2), cos(t/2)]]
            # d/dtheta Ry = 0.5 * [[-sin(t/2), -cos(t/2)], [cos(t/2), -sin(t/2)]]
            
            theta = params[l, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            
            # The state before this gate is pre_states[index]
            # We need <dpsi|chi>.
            # |psi_after> = Ry |psi_before>
            # d|psi_after>/dtheta = (dRy/dtheta) |psi_before>
            # grad = 2 Re[ < (dRy/dtheta)|psi_before> | chi_after> ]
            # Let |v> = (dRy/dtheta)^dagger |chi_after>
            # Then <dpsi|chi> = <psi_before| v>
            
            # dRy/dtheta = -0.5 * sin(t/2) * I - 0.5 * i * cos(t/2) * Y
            # (dRy/dtheta)^dagger = -0.5 * sin(t/2) * I + 0.5 * i * cos(t/2) * Y
            # Since Y is Hermitian.
            
            # Let's compute |v> = (dRy/dtheta)^dagger |chi_after>
            # |chi_after> is adj_state.
            # Move qubit q to front.
            chi_tensor = adj_state.reshape([2] * n)
            chi_tensor = np.moveaxis(chi_tensor, q, 0)
            
            # Apply operator to the qubit index 0 of the tensor.
            # v_0 = ( -0.5 s * I + 0.5 i c * Y ) |chi>_0
            # Y = [[0, -i], [i, 0]]
            # i Y = [[0, 1], [-1, 0]]
            # So 0.5 i c Y = 0.5 c [[0, 1], [-1, 0]]
            # Operator M = [[-0.5 s, 0.5 c], [-0.5 c, -0.5 s]]
            
            v_tensor = np.empty_like(chi_tensor)
            v_tensor[0] = -0.5 * s * chi_tensor[0] + 0.5 * c * chi_tensor[1]
            v_tensor[1] = -0.5 * c * chi_tensor[0] - 0.5 * s * chi_tensor[1]
            
            # Move axis back
            v_tensor = np.moveaxis(v_tensor, 0, q)
            v_state = v_tensor.reshape(dim)
            
            # grad = 2 Re[ <psi_before| v_state> ]
            psi_before = pre_states[l * n + q]
            inner = np.vdot(psi_before, v_state)
            grad[l, q] = 2 * inner.real
            
            # Now update adj_state to propagate to the previous gate.
            # adj_state_before = Ry^dagger adj_state_after
            # Ry is real orthogonal? No, Ry(theta) is real if we use the standard definition?
            # Ry(theta) = [[cos, -sin], [sin, cos]]. This is real and orthogonal.
            # So Ry^dagger = Ry^T = Ry(-theta).
            
            adj_tensor = adj_state.reshape([2] * n)
            adj_tensor = np.moveaxis(adj_tensor, q, 0)
            
            new_adj_tensor = np.empty_like(adj_tensor)
            new_adj_tensor[0] = c * adj_tensor[0] + s * adj_tensor[1]
            new_adj_tensor[1] = -s * adj_tensor[0] + c * adj_tensor[1]
            
            adj_tensor = np.moveaxis(new_adj_tensor, 0, q)
            adj_state = adj_tensor.reshape(dim)
            
        # Apply CZ adjoint (CZ is self-adjoint and real diagonal)
        if l < layers - 1:
            for q in range(n - 1):
                mask1 = 1 << q
                mask2 = 1 << (q + 1)
                indices = np.arange(dim)
                mask = (indices & mask1) != 0
                mask2_bool = (indices & mask2) != 0
                both = mask & mask2_bool
                adj_state[both] *= -1
                
    return energy, grad

def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 145.0
    
    E0 = _ground_energy(n, coupling, field)
    H = _build_hamiltonian(n, coupling, field)
    
    best_result = None
    best_energy = np.inf
    
    # Since analytic gradients are fast, we can do more restarts.
    # Let's try 20 restarts.
    n_restarts = 20
    rng = np.random.default_rng(42)
    
    # Pre-calculate bounds
    bounds = [(-np.pi, np.pi)] * (layers * n)
    
    for i in range(n_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Random initial point
        init_params = rng.uniform(-np.pi, np.pi, (layers, n))
        
        def objective(flat_params):
            params = flat_params.reshape((layers, n))
            e, _ = _energy_and_grad_analytic(params, n, layers, H, E0)
            return e
        
        def grad_func(flat_params):
            params = flat_params.reshape((layers, n))
            _, g = _energy_and_grad_analytic(params, n, layers, H, E0)
            return g.flatten()
        
        try:
            res = minimize(
                objective,
                init_params.flatten(),
                jac=grad_func,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if res.fun < best_energy:
                best_energy = res.fun
                best_result = res.x
                
        except Exception as e:
            pass
            
    if best_result is None:
        return np.zeros((layers, n))
        
    return best_result.reshape((layers, n))
