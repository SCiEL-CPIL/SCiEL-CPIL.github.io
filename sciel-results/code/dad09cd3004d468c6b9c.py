import numpy as np
import scipy.optimize
import time

def _get_hamiltonian(n, coupling, field):
    """Construct the Hamiltonian matrix H = -coupling * sum Z_i Z_{i+1} - field * sum X_i."""
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Z = np.array([[1, 0], [0, -1]])
    
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)
    
    def op_on_qubit(op, i, n):
        ops = [I] * n
        ops[n-1-i] = op
        result = ops[0]
        for k in range(1, n):
            result = np.kron(result, ops[k])
        return result
    
    # Interaction terms: -coupling * sum Z_i Z_{i+1}
    for i in range(n - 1):
        Zi = op_on_qubit(Z, i, n)
        Zj = op_on_qubit(Z, i+1, n)
        H -= coupling * (Zi @ Zj)
        
    # Field terms: -field * sum X_i
    for i in range(n):
        Xi = op_on_qubit(X, i, n)
        H -= field * Xi
        
    return H

def _apply_circuit(theta, n, layers):
    """Apply the circuit to |0>^n and return the state vector."""
    state = np.zeros(2 ** n, dtype=complex)
    state[0] = 1.0
    
    # Precompute masks for CZ
    indices = np.arange(2 ** n)
    cz_masks = []
    for i in range(n - 1):
        mask_i = 1 << i
        mask_j = 1 << (i + 1)
        both = ((indices & mask_i) != 0) & ((indices & mask_j) != 0)
        cz_masks.append(both)
    
    def apply_ry(state, theta_val, qubit, n):
        c = np.cos(theta_val / 2)
        s = np.sin(theta_val / 2)
        # Ry(theta) = [[c, -s], [s, c]]
        Ry = np.array([[c, -s], [s, c]], dtype=complex)
        
        # Apply to qubit (0 is LSB)
        # Reshape state to tensor of shape (2, 2, ..., 2)
        # Qubit 0 is the last axis in standard C-order if we view index as b_{n-1}...b_0
        # Wait, index k = b_{n-1} * 2^{n-1} + ... + b_0 * 2^0.
        # If we reshape to (2, 2, ..., 2), the first axis corresponds to the most significant bit (b_{n-1}), 
        # and the last axis corresponds to the least significant bit (b_0).
        # So qubit i corresponds to axis (n - 1 - i).
        
        # Tensor reshape
        state_tensor = state.reshape([2] * n)
        
        # Get the axes for the other qubits to move them to the front? 
        # Or just use np.tensordot? 
        # Simpler: use the kron matrix multiplication but optimized?
        # For n=6, dim=64. Building the 64x64 matrix every time is slow.
        # Let's stick to the tensor method.
        
        # Move the target axis to the end
        target_axis = n - 1 - qubit
        # Move all other axes to the front, keeping relative order
        other_axes = [ax for ax in range(n) if ax != target_axis]
        
        # Permute so that target axis is last
        perm = other_axes + [target_axis]
        state_perm = np.transpose(state_tensor, perm)
        
        # Reshape to (2^{n-1}, 2)
        state_mat = state_perm.reshape(-1, 2)
        
        # Apply Ry
        new_state_mat = state_mat @ Ry.T  # Wait, state is a column vector? 
        # If state is a row vector in tensor form, we multiply by matrix on the right?
        # Let's check.
        # State vector v. v_i = <i|psi>.
        # New state v' = U v.
        # In tensor form, we are effectively doing matrix multiplication.
        # If we treat the last axis as the vector index, then v' = v @ U^T.
        # Yes, because (v @ U^T)_j = sum_i v_i U_{ij} = sum_i <i|psi> U_{ij} = <j| U |psi>. Correct.
        
        # Reshape back
        new_state_perm = new_state_mat.reshape(state_perm.shape)
        
        # Permute back
        inv_perm = np.argsort(perm)
        new_state_tensor = np.transpose(new_state_perm, inv_perm)
        
        return new_state_tensor.reshape(2 ** n)

    def apply_cz(state, mask):
        state[mask] *= -1
        return state

    for l in range(layers):
        # Apply Ry to all qubits
        for i in range(n):
            state = apply_ry(state, theta[l, i], i, n)
        
        # Apply CZ between layers (not after last)
        if l < layers - 1:
            for i in range(n - 1):
                state = apply_cz(state, cz_masks[i])
                
    return state

def _energy(state, H):
    psi = state
    E = np.vdot(psi, H @ psi).real
    return E

def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin for 150s limit
    
    # 1. Compute Exact Ground Energy
    H = _get_hamiltonian(n, coupling, field)
    E_vals = np.linalg.eigvalsh(H)
    E_exact = E_vals[0]
    
    # 2. Define Cost Function
    def cost(theta_flat):
        theta = theta_flat.reshape(layers, n)
        state = _apply_circuit(theta, n, layers)
        E_circuit = _energy(state, H)
        return (E_circuit - E_exact) / n

    # 3. Optimization
    # Initial guess:
    # The seed program suggests that for small fields, the state is close to |+>^n or |0>^n?
    # Inspiration 1: theta[-1] = pi/2. This creates |+> on the last layer?
    # If theta=0, state is |0>.
    # If theta=pi/2, Ry(pi/2)|0> = (|0>+|1>)/sqrt(2) = |+>.
    # So the seed creates a product state of |+> on the last layer, |0> on previous layers (if theta=0).
    # Wait, the seed sets theta[-1] = pi/2 and others 0.
    # Layer 0: Ry(0) -> |0>. CZ.
    # Layer 1: Ry(0) -> |0>. CZ.
    # Layer 2: Ry(pi/2) -> |+>.
    # So final state is |+>^n.
    
    # For small field (0.3), ground state is close to |0>^n? 
    # H = -J ZZ - h X.
    # If h=0, ground state is ferromagnetic |0000> or |1111> (degenerate).
    # If h is small, it's a superposition.
    # Let's try multiple starts.
    
    # Start 1: Zeros (|0>^n)
    # Start 2: Seed-like (|+>^n on last layer)
    # Start 3: Random
    
    candidates = []
    
    # Start 1: Zeros
    x0_1 = np.zeros(layers * n)
    candidates.append(x0_1)
    
    # Start 2: Seed-like
    x0_2 = np.zeros(layers * n)
    # Last layer all pi/2
    x0_2[(layers-1)*n : layers*n] = np.pi / 2
    candidates.append(x0_2)
    
    # Start 3: Small random
    np.random.seed(42)
    x0_3 = np.random.randn(layers * n) * 0.1
    candidates.append(x0_3)
    
    best_theta = None
    best_cost = np.inf
    
    for x0 in candidates:
        if time.time() > deadline - 5:
            break
            
        # Use Nelder-Mead
        try:
            res = scipy.optimize.minimize(
                cost, 
                x0, 
                method='Nelder-Mead', 
                options={'maxiter': 2000, 'xatol': 1e-7, 'fatol': 1e-9, 'disp': False}
            )
            if res.fun < best_cost:
                best_cost = res.fun
                best_theta = res.x
        except Exception:
            pass
            
    # If we have time, try BFGS or L-BFGS-B with gradients? 
    # Gradient of energy w.r.t theta can be computed via parameter shift rule.
    # But Nelder-Mead is robust.
    
    # If we have lots of time, do a random restart
    if time.time() < deadline - 10 and best_theta is not None:
        for _ in range(2):
            if time.time() > deadline - 5:
                break
            x0_rand = np.random.randn(layers * n) * 0.5
            try:
                res = scipy.optimize.minimize(
                    cost, 
                    x0_rand, 
                    method='Nelder-Mead', 
                    options={'maxiter': 1000, 'xatol': 1e-7, 'fatol': 1e-9, 'disp': False}
                )
                if res.fun < best_cost:
                    best_cost = res.fun
                    best_theta = res.x
            except Exception:
                pass

    if best_theta is None:
        # Fallback to seed
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        return theta

    return best_theta.reshape(layers, n)

# EVOLVE-BLOCK-END
