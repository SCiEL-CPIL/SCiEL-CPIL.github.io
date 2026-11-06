import numpy as np
import scipy.optimize as opt
import time

# Global cache for results
_cache = {}

def _build_hamiltonian(n, coupling, field):
    """
    Build the Hamiltonian matrix for the open transverse-field Ising chain.
    H = -coupling * sum Z_i Z_{i+1} - field * sum X_i
    """
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)
    
    # Precompute bit values for efficiency
    idx = np.arange(dim)
    
    # ZZ terms (Diagonal)
    for i in range(n - 1):
        mask_i = 1 << i
        mask_ip1 = 1 << (i + 1)
        bit_i = (idx >> i) & 1
        bit_ip1 = (idx >> (i + 1)) & 1
        z_i = 1 - 2 * bit_i
        z_ip1 = 1 - 2 * bit_ip1
        zz_val = -coupling * z_i * z_ip1
        H[idx, idx] += zz_val
        
    # X terms (Off-diagonal)
    for i in range(n):
        mask = 1 << i
        k_flip = idx ^ mask
        H[idx, k_flip] += -field
        H[k_flip, idx] += -field
        
    return H

def _get_exact_ground_state(n, coupling, field):
    """
    Compute the exact ground state and energy.
    """
    H = _build_hamiltonian(n, coupling, field)
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    ground_state = eigenvectors[:, 0]
    ground_energy = eigenvalues[0]
    return ground_state, ground_energy

def _compute_local_magnetizations(state, n):
    """
    Compute <Z_i> for each qubit i in the given state.
    """
    dim = 2 ** n
    probs = np.abs(state) ** 2
    idx = np.arange(dim)
    
    mags = np.zeros(n)
    for i in range(n):
        mask = 1 << i
        bit_vals = (idx >> i) & 1
        z_vals = 1 - 2 * bit_vals
        mags[i] = np.sum(probs * z_vals)
    return mags

class _CircuitSimulator:
    """
    Optimized circuit simulator with analytic gradient support.
    State is stored as a complex vector of length 2^n.
    """
    def __init__(self, n, coupling, field, layers):
        self.n = n
        self.coupling = coupling
        self.field = field
        self.layers = layers
        self.dim = 2 ** n
        
        # Precompute Hamiltonian for energy calculation
        self.H = _build_hamiltonian(n, coupling, field)
        
        # Precompute Ry matrices for each layer? No, they depend on params.
        # But we can precompute the CZ gates structure.
        # CZ on (i, i+1) flips sign if both bits are 1.
        # We will apply CZ by multiplying the state vector by a diagonal matrix.
        # diag[i] = -1 if bit_i and bit_{i+1} are both 1, else 1.
        self.cz_diags = []
        idx = np.arange(self.dim)
        for i in range(n - 1):
            mask_i = 1 << i
            mask_ip1 = 1 << (i + 1)
            cond = ((idx & mask_i) != 0) & ((idx & mask_ip1) != 0)
            diag = np.ones(self.dim, dtype=complex)
            diag[cond] = -1.0
            self.cz_diags.append(diag)

    def _apply_ry(self, state, i, theta):
        """Apply Ry(theta) to qubit i (bit i) in place? No, returns new state.
        Qubit i corresponds to axis i in the tensor, but in the flat vector,
        it's the i-th bit from the right.
        We can use a permutation-free approach or just construct the matrix.
        For n=4,6, dim is 16, 64. Matrix construction is cheap.
        """
        n = self.n
        dim = self.dim
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        
        # Construct the Ry matrix for qubit i
        # Identity on other qubits, Ry on qubit i
        # Ry = [[c, -s], [s, c]]
        
        # Efficient application:
        # Split the state into pairs based on bit i.
        # If bit i is 0, it's in the first half of the block, if 1, second half.
        # Actually, for bit i, the stride is 2^i.
        # Let's just use the matrix multiplication for simplicity and speed with n<=6.
        # M = I_{2^{n-1}} \otimes Ry \otimes I_{2^{n-1-i}} ... wait, ordering.
        # Qubit 0 is LSB.
        # So M = (I_{2^{n-i-1}} \otimes Ry \otimes I_{2^i})
        
        # Let's construct the full matrix M of size dim x dim.
        M = np.eye(dim, dtype=complex)
        # This is slow. Let's use the tensor product logic or just precompute the action.
        
        # Alternative: Use the fact that Ry acts on bit i.
        # We can reshape state to (2, 2, ..., 2).
        # Axis for qubit i in numpy tensor (where state[b_{n-1}...b_0]) is n-1-i.
        # Let's stick to the flat vector and use a precomputed permutation or just matrix mult.
        # For n=6, dim=64. 64x64 matrix mult is fast.
        
        # Let's build M once per call? No, theta changes.
        # We can build the sparse structure.
        # Actually, let's just use the tensor approach, it's cleaner.
        
        state_tensor = state.reshape([2] * n)
        axis = n - 1 - i
        
        state_tensor = np.moveaxis(state_tensor, axis, -1)
        
        old_0 = state_tensor[..., 0]
        old_1 = state_tensor[..., 1]
        
        new_0 = c * old_0 - s * old_1
        new_1 = s * old_0 + c * old_1
        
        new_state_slice = np.empty_like(state_tensor)
        new_state_slice[..., 0] = new_0
        new_state_slice[..., 1] = new_1
        
        new_state = np.moveaxis(new_state_slice, -1, axis)
        return new_state.reshape(-1)

    def _apply_cz_sequence(self, state):
        """Apply CZ on all adjacent pairs in ascending order."""
        for i in range(self.n - 1):
            state = self.cz_diags[i] * state
        return state

    def forward(self, params):
        """
        params: array of shape (layers, n)
        Returns: final_state
        """
        state = np.zeros(self.dim, dtype=complex)
        state[0] = 1.0  # |0...0>
        
        thetas = params.reshape(self.layers, self.n)
        
        for l in range(self.layers):
            # Apply Ry to all qubits
            for i in range(self.n):
                state = self._apply_ry(state, i, thetas[l, i])
            
            # Apply CZ between layers
            if l < self.layers - 1:
                state = self._apply_cz_sequence(state)
                
        return state

    def energy(self, state):
        """Compute <state|H|state>"""
        return np.real(np.vdot(state, self.H @ state))

    def objective_and_grad(self, params):
        """
        Compute energy and gradient of energy w.r.t. params.
        Uses backpropagation through the circuit.
        """
        # Forward pass, storing states
        states = []
        state = np.zeros(self.dim, dtype=complex)
        state[0] = 1.0
        states.append(state)
        
        thetas = params.reshape(self.layers, self.n)
        
        for l in range(self.layers):
            for i in range(self.n):
                state = self._apply_ry(state, i, thetas[l, i])
                states.append(state)
            
            if l < self.layers - 1:
                state = self._apply_cz_sequence(state)
                states.append(state)
                
        final_state = states[-1]
        energy = self.energy(final_state)
        
        # Backward pass
        # grad = dE/dparams
        # dE/dstate = H @ state (for real energy, it's 2*Re(H|state>)? No, E = <psi|H|psi>. dE/dpsi* = H|psi>. dE/dpsi = <psi|H.)
        # Chain rule: dE/dtheta = Re( <dpsi/dtheta>^dagger (H|psi>) ) ... wait.
        # E = psi^H H psi.
        # dE = (dpsi)^H H psi + psi^H H dpsi.
        # Since E is real, dE = 2 Re( (dpsi)^H H psi ).
        # Let r = H @ final_state.
        # We need to propagate the gradient of the loss w.r.t. the state backwards.
        # Let g_k be the gradient of E w.r.t. state at step k.
        # g_final = 2 * Re(H @ final_state)? No, the complex gradient.
        # Let's use the standard trick: 
        # L = <psi|H|psi>. 
        # dL/dpsi* = H psi.
        # dL/dpsi = (H psi)^* (if H is Hermitian, (H psi)^* = psi^H H^T = psi^H H).
        
        # Backprop:
        # grad_state_final = H @ final_state  (this is dE/dpsi*)
        # Actually, we want dE/dtheta.
        # dE = sum (dE/dpsi*_k * dpsi*_k/dtheta + dE/dpsi_k * dpsi_k/dtheta)
        # It's easier to propagate the "adjoint" vector v_k such that dE = v_k^H dpsi_k.
        # v_final = H @ final_state.
        
        v = self.H @ final_state
        
        grads = np.zeros_like(params, dtype=float)
        
        # Reverse steps
        # We need to undo the operations.
        # Steps in forward:
        # 1. Ry(i, theta)
        # 2. CZ (diagonal multiplication)
        
        # We stored states. We can re-derive the gradients.
        # Let's iterate backwards through the layers.
        
        # State indexing in 'states':
        # states[0] = initial
        # For l in 0..L-1:
        #   For i in 0..n-1:
        #      states.append(Ry_i(state))
        #   If l < L-1:
        #      states.append(CZ(state))
        
        # Let's just re-simulate backwards using the stored states to get the input to each gate.
        
        idx = len(states) - 1
        # v is currently associated with states[-1]
        
        for l in range(self.layers - 1, -1, -1):
            # Undo CZ if present
            if l < self.layers - 1:
                # CZ is diagonal. v_new = diag * v_old
                # Since diag is real and symmetric (involutory), it's just multiplication.
                for i in range(self.n - 1):
                    v = self.cz_diags[i] * v
                idx -= 1 # skip the CZ state in the list? 
                # Wait, my forward loop appended state after CZ.
                # So states[idx] is the state after CZ.
                # v corresponds to states[idx].
                # We need to propagate v to the state before CZ.
                # v_before_cz = CZ^H v_after_cz = CZ v_after_cz (since CZ is Hermitian).
                # This is what I did above.
                # Now v corresponds to the state BEFORE the CZ, which is states[idx-1] in my list?
                # Let's check the list structure.
                
                # Forward:
                # state = Ry...
                # state = CZ(state)
                # states.append(state)
                
                # So if l < L-1, the last appended state for this layer is post-CZ.
                # The state before CZ is the one before that append.
                # My loop above updated v to be the gradient w.r.t. the state BEFORE CZ.
                # Now we need to propagate through the Ry gates of layer l.
                
            # Undo Ry gates for layer l
            for i in range(self.n - 1, -1, -1):
                # The gate was Ry(i, theta) applied to state_in to get state_out.
                # state_out = M_Ry state_in.
                # v_in = M_Ry^H v_out.
                # M_Ry is unitary, so M_Ry^H = M_Ry^(-1) = Ry(-theta).
                
                theta = thetas[l, i]
                
                # We need v_in.
                # v is currently the gradient w.r.t. state_out.
                # We compute v_in.
                
                # Then we compute dE/dtheta.
                # dE = v_in^H (dM_Ry/dtheta) state_in.
                # state_in is the state BEFORE this Ry gate.
                # We can retrieve state_in from the 'states' list.
                
                # Let's find the index of state_in.
                # This is getting complicated with indices. 
                # Alternative: Store the inputs to each gate.
                
                pass # Placeholder, need better structure

        # Let's rewrite the backprop more cleanly by storing inputs.
        # This is a bit slow to code but necessary for correctness.
        
        return energy, grads

    def objective_and_grad_v2(self, params):
        """
        Cleaner backprop implementation.
        """
        n = self.n
        layers = self.layers
        dim = self.dim
        
        # Forward pass
        thetas = params.reshape(layers, n)
        
        # Store:
        # inputs[l, i] = state before Ry(l, i)
        # outputs[l, i] = state after Ry(l, i)
        # cz_inputs[l] = state before CZ after layer l (if l < layers-1)
        
        inputs = np.zeros((layers, n, dim), dtype=complex)
        outputs = np.zeros((layers, n, dim), dtype=complex)
        cz_inputs = np.zeros((layers - 1, dim), dtype=complex) if layers > 1 else None
        
        state = np.zeros(dim, dtype=complex)
        state[0] = 1.0
        
        for l in range(layers):
            for i in range(n):
                inputs[l, i] = state
                state = self._apply_ry(state, i, thetas[l, i])
                outputs[l, i] = state
            
            if l < layers - 1:
                cz_inputs[l] = state
                state = self._apply_cz_sequence(state)
                
        final_state = state
        energy = self.energy(final_state)
        
        # Backward pass
        # v = dE/dfinal_state = H @ final_state
        v = self.H @ final_state
        
        grads = np.zeros((layers, n), dtype=float)
        
        for l in range(layers - 1, -1, -1):
            # Undo CZ
            if l < layers - 1:
                # v is grad w.r.t. state after CZ
                # state after CZ = CZ * cz_inputs[l]
                # v_before_cz = CZ^H v = CZ v
                for i in range(n - 1):
                    v = self.cz_diags[i] * v
                # Now v is grad w.r.t. cz_inputs[l] (which is outputs[l, n-1])
            
            # Undo Ry gates
            for i in range(n - 1, -1, -1):
                # v is grad w.r.t. outputs[l, i]
                # outputs[l, i] = M_Ry(theta) inputs[l, i]
                # v_in = M_Ry(-theta) v
                
                theta = thetas[l, i]
                state_in = inputs[l, i]
                
                # Compute v_in
                v_in = self._apply_ry(v, i, -theta)
                
                # Compute grad
                # dE/dtheta = Re( v_in^H (dM/dtheta) state_in )
                # dM/dtheta = 0.5 * [[-s, -c], [c, -s]] (in the subspace of qubit i)
                # Let's compute (dM/dtheta) state_in.
                # This is the derivative of the state w.r.t. theta.
                
                # state_out = M(theta) state_in
                # dstate_out/dtheta = M'(theta) state_in
                
                # We can compute M'(theta) state_in by:
                # M'(theta) = 0.5 * d/dtheta M(theta)
                # M(theta) = c I - i s X (in local basis, but Ry is real)
                # Ry(theta) = [[cos(t/2), -sin(t/2)], [sin(t/2), cos(t/2)]]
                # dRy/dt = 0.5 * [[-sin(t/2), -cos(t/2)], [cos(t/2), -sin(t/2)]]
                
                c = np.cos(theta / 2)
                s = np.sin(theta / 2)
                
                # We need to apply dRy/dt to state_in on qubit i.
                # Let's construct the operator D = dRy/dt for qubit i.
                # D = 0.5 * ( -s Y - c X )? No.
                # Ry = exp(-i theta Y / 2).
                # dRy/dtheta = -i/2 Y Ry = -i/2 Y exp(-i theta Y / 2).
                # In the basis where Y is Pauli Y, this is tricky.
                # Let's just use the matrix form.
                
                # Apply dRy/dt to state_in.
                # state_in_tensor = state_in.reshape([2]*n)
                # axis = n-1-i
                # Move axis to end
                # Apply matrix [[-s/2, -c/2], [c/2, -s/2]]
                
                st = state_in.reshape([2] * n)
                axis = n - 1 - i
                st = np.moveaxis(st, axis, -1)
                
                old_0 = st[..., 0]
                old_1 = st[..., 1]
                
                new_0 = -0.5 * s * old_0 - 0.5 * c * old_1
                new_1 = 0.5 * c * old_0 - 0.5 * s * old_1
                
                d_st = np.empty_like(st)
                d_st[..., 0] = new_0
                d_st[..., 1] = new_1
                d_st = np.moveaxis(d_st, -1, axis)
                d_state = d_st.reshape(-1)
                
                # grad = Re( v_in^H d_state )
                grad_val = np.real(np.vdot(v_in, d_state))
                grads[l, i] = grad_val
                
                # Update v to v_in for the next gate
                v = v_in
                
        return energy, grads.reshape(-1)

def _optimize(n, coupling, field, layers, timeout=115.0):
    key = (n, coupling, field, layers)
    if key in _cache:
        return _cache[key]
    
    start_time = time.time()
    n_params = n * layers
    
    try:
        ground_state, ground_energy = _get_exact_ground_state(n, coupling, field)
        mags = _compute_local_magnetizations(ground_state, n)
        # Theta for Ry to produce magnetization m: <Z> = cos(theta)
        # So theta = arccos(m)
        theta_last = np.arccos(np.clip(mags, -1, 1))
        
        # Initialize
        # Best guess: Last layer matches ground state magnetizations, others zero?
        init = np.zeros(n_params)
        init[-n:] = theta_last
        
    except Exception:
        init = np.random.uniform(0, 2*np.pi, n_params)
        
    sim = _CircuitSimulator(n, coupling, field, layers)
    
    best_params = init
    best_energy = sim.objective_and_grad_v2(init)[0]
    
    # Use L-BFGS-B with analytic gradient
    def obj_grad(params):
        e, g = sim.objective_and_grad_v2(params)
        return e, g
        
    try:
        res = opt.minimize(
            obj_grad, 
            init, 
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 20}
        )
        if res.fun < best_energy:
            best_energy = res.fun
            best_params = res.x
    except Exception:
        pass
    
    # If time remains, try a second start or refine further
    current_time = time.time()
    if current_time - start_time < timeout - 5:
        try:
            # Try a different initialization: random
            init2 = np.random.uniform(0, 2*np.pi, n_params)
            res2 = opt.minimize(
                obj_grad, 
                init2, 
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 20}
            )
            if res2.fun < best_energy:
                best_energy = res2.fun
                best_params = res2.x
        except Exception:
            pass

    result_angles = best_params.reshape(layers, n)
    _cache[key] = result_angles
    return result_angles

def angles(n, coupling, field, layers):
    n = int(n)
    coupling = float(coupling)
    field = float(field)
    layers = int(layers)
    
    return _optimize(n, coupling, field, layers)
