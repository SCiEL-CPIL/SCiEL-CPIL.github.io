import numpy as np
import scipy.optimize as opt

def angles(n, coupling, field, layers):
    """
    Optimize parameters of a fixed three-layer Ry/CZ circuit to approximate 
    ground states of open transverse-field Ising chains.
    """
    # Dimensions
    dim = 2 ** n
    
    # Construct Pauli matrices
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    # Construct Hamiltonian H = -coupling * sum Z_i Z_{i+1} - field * sum X_i
    H = np.zeros((dim, dim), dtype=complex)
    
    # Helper to create operator on specific qubits
    def op_on_qubits(ops, qubits):
        # ops is a list of 2x2 matrices, qubits is a list of indices (0-based, least significant bit is qubit 0)
        # We need to map qubit indices to tensor product positions.
        # Standard convention: qubit 0 is the rightmost (least significant) in the binary string, 
        # but in Kronecker products, the order depends on implementation.
        # Let's stick to a consistent convention: 
        # State |b_{n-1} ... b_0>. Qubit i corresponds to bit b_i.
        # In numpy kron, A ⊗ B applies A to the "left" (more significant) and B to the "right" (less significant).
        # So if we want to apply Z to qubit i, we need to place it in the correct position.
        
        # Let's build the operator by iterating from most significant to least significant.
        # Qubit n-1 is the most significant. Qubit 0 is the least significant.
        full_op = np.array([[1]], dtype=complex)
        for q in range(n-1, -1, -1):
            if q in qubits:
                idx = qubits.index(q)
                op = ops[idx]
            else:
                op = I
            full_op = np.kron(full_op, op)
        return full_op

    # ZZ terms
    for i in range(n - 1):
        # Qubits i and i+1
        # We need to apply Z to qubit i and Z to qubit i+1
        # Using the helper:
        # Note: The helper above expects a list of ops and corresponding qubits.
        # It's easier to just construct the matrix directly or use a simpler kron logic.
        
        # Let's use a simpler approach for small n: iterate through all basis states
        # Or just use the kron helper correctly.
        pass
    
    # Let's rewrite the Hamiltonian construction more simply for small n.
    H = np.zeros((dim, dim), dtype=complex)
    
    # Basis states
    basis = np.arange(dim)
    
    # ZZ terms
    for i in range(n - 1):
        # Operator Z_i Z_{i+1}
        # We can compute the expectation value or just build the matrix.
        # Matrix element <b| Z_i Z_{i+1} |b'> is non-zero only if b' = b.
        # Value is product of Z eigenvalues.
        for b in range(dim):
            # Get bits
            bit_i = (b >> i) & 1
            bit_ip1 = (b >> (i + 1)) & 1
            # Z eigenvalue: +1 for 0, -1 for 1
            z_i = 1 if bit_i == 0 else -1
            z_ip1 = 1 if bit_ip1 == 0 else -1
            val = z_i * z_ip1
            H[b, b] += -coupling * val
            
    # X terms
    for i in range(n):
        for b in range(dim):
            # X flips bit i
            b_prime = b ^ (1 << i)
            H[b, b_prime] += -field
            
    # Exact ground energy
    eigvals = np.linalg.eigvalsh(H)
    E_exact = eigvals[0].real
    
    # Define the circuit application
    def apply_circuit(theta, state):
        # theta shape: (layers, n)
        # state: vector of size dim
        for l in range(layers):
            # Apply Ry(theta[l, i]) to qubit i
            # Ry(theta) = [[cos(theta/2), -sin(theta/2)], [sin(theta/2), cos(theta/2)]]
            # We apply Ry to each qubit.
            # We need to apply the operator to the state.
            # For each qubit i, we apply Ry_i.
            # Since they are on different qubits, we can apply them sequentially.
            for i in range(n):
                theta_i = theta[l, i]
                c = np.cos(theta_i / 2)
                s = np.sin(theta_i / 2)
                Ry = np.array([[c, -s], [s, c]], dtype=complex)
                
                # Apply Ry to qubit i of the state
                # We reshape state to (2, 2, ..., 2) with n dimensions.
                # Qubit i is the (n-1-i)-th axis if we use standard kron order (left to right = high to low).
                # Let's define the reshape order explicitly.
                # State index b = sum b_k 2^k.
                # Numpy reshape: if we reshape to (2,)*n, the last axis varies fastest.
                # So axis 0 is qubit n-1, axis n-1 is qubit 0.
                # Axis for qubit i is n-1-i.
                
                shape = (2,) * n
                state_tensor = state.reshape(shape)
                
                # Swap the axis for qubit i to the front, apply, swap back.
                axis = n - 1 - i
                state_tensor = np.swapaxes(state_tensor, axis, 0)
                # Apply Ry: new_state = Ry @ old_state (for each slice)
                # state_tensor is now (2, 2, ..., 2). We want to multiply Ry (2x2) with the first axis.
                state_tensor = np.tensordot(Ry, state_tensor, axes=([1], [0]))
                # tensordot(Ry, state_tensor, axes=([1],[0])) sums over the second index of Ry and first of state.
                # Result shape: (2, 2, ..., 2). The first axis is the new qubit state.
                state_tensor = np.swapaxes(state_tensor, 0, axis)
                state = state_tensor.reshape(dim)
                
            # Apply CZ on every adjacent pair in ascending order
            # "between layers apply CZ on every adjacent pair in ascending order"
            # This implies: for i in range(n-1): apply CZ(i, i+1)
            # CZ matrix: diag(1, 1, 1, -1) in basis |00>, |01>, |10>, |11>
            # It applies a phase of -1 if both qubits are |1>.
            
            if l < layers - 1: # No entangler after final layer
                for i in range(n - 1):
                    # Apply CZ to qubits i and i+1
                    # Iterate over all basis states
                    # Or use tensor operations
                    state_tensor = state.reshape((2,) * n)
                    axis1 = n - 1 - i
                    axis2 = n - 1 - (i + 1)
                    
                    # Move axes to front
                    # We need to apply a 4x4 operator on two qubits.
                    # Easier: iterate over the 2^2 combinations of these two qubits?
                    # No, just apply the phase.
                    # If bit i is 1 and bit i+1 is 1, multiply by -1.
                    
                    # Create a mask or just iterate
                    # For small n, iterating is fine.
                    for b in range(dim):
                        bit_i = (b >> i) & 1
                        bit_ip1 = (b >> (i + 1)) & 1
                        if bit_i == 1 and bit_ip1 == 1:
                            state[b] *= -1
                            
    # Objective function
    def objective(theta_flat):
        theta = theta_flat.reshape(layers, n)
        # Start state |0>^n
        state = np.zeros(dim, dtype=complex)
        state[0] = 1.0
        
        state = apply_circuit(theta, state)
        
        # Energy <psi|H|psi>
        energy = np.vdot(state, H @ state).real
        return energy
    
    # Initial guess: Seed program angles
    # Seed: theta = zeros, theta[-1] = pi/2
    theta0 = np.zeros((layers, n))
    theta0[-1] = np.pi / 2
    
    # Bounds
    bounds = [(0, 2 * np.pi) for _ in range(layers * n)]
    
    # Optimize
    try:
        res = opt.minimize(objective, theta0.ravel(), method='L-BFGS-B', bounds=bounds, options={'maxiter': 200, 'ftol': 1e-8})
        theta_opt = res.x.reshape(layers, n)
    except Exception:
        # Fallback to seed if optimization fails
        theta_opt = theta0
        
    return theta_opt
