"""Optimize Ry/CZ circuit parameters for Ising ground states."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _build_hamiltonian(n, coupling, field):
    """Build H = -coupling * sum Z_i Z_{i+1} - field * sum X_i."""
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.float64)
    
    # Pre-compute indices for efficiency
    indices = np.arange(dim)
    
    for i in range(n - 1):
        bi = (indices >> i) & 1
        bj = (indices >> (i + 1)) & 1
        zval = (1 - 2 * bi) * (1 - 2 * bj)
        H[indices, indices] += -coupling * zval
    
    for i in range(n):
        mask = 1 << i
        # We need to add -field to H[b, b^mask] and H[b^mask, b]
        # Since H is symmetric and real, we can just add to the off-diagonals
        # Iterate over blocks where bit i is 0
        # Vectorized approach:
        # b is even with respect to bit i?
        # Let's just loop over the upper triangle part for X terms to avoid double counting or just do full matrix update
        # H += -field * X_i
        # X_i flips bit i.
        # H[b, b^mask] += -field
        # H[b^mask, b] += -field
        
        # Efficient way:
        # For all b, H[b, b^mask] += -field
        # This is a permutation matrix scaled.
        perm = indices ^ mask
        H[indices, perm] += -field
        # Since X is Hermitian, H is symmetric. Adding -field to H[b, b^mask] covers both directions if we iterate all b?
        # No, if we do H[indices, perm] += -field, we add -field to H[0, 1] and H[1, 0] etc.
        # Wait, if b=0, perm=1. H[0,1] += -field.
        # If b=1, perm=0. H[1,0] += -field.
        # So this single line adds -field to both off-diagonal entries for every pair. Correct.
        
    return H


def _circuit_state(params, n, layers):
    """Compute circuit state vector using vectorized operations."""
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0
    
    # Pre-compute index masks for Ry gates
    # For qubit q, we need indices where bit q is 0 (idx0) and 1 (idx1)
    # idx1 = idx0 | (1 << q)
    # We can reshape the state to apply gates more efficiently?
    # For n=6, dim=64. Simple slicing is fast enough.
    
    for layer in range(layers):
        for q in range(n):
            theta = params[layer, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            mask = 1 << q
            
            # Create indices
            # idx0: bits where q-th bit is 0
            # idx1: bits where q-th bit is 1
            # We can use np.arange and bitwise ops
            idx = np.arange(dim)
            idx0 = idx[(idx & mask) == 0]
            idx1 = idx0 | mask
            
            # Apply Ry:
            # [c  -s]
            # [s   c]
            # new[idx0] = c*old[idx0] - s*old[idx1]
            # new[idx1] = s*old[idx0] + c*old[idx1]
            
            v0 = state[idx0]
            v1 = state[idx1]
            
            state[idx0] = c * v0 - s * v1
            state[idx1] = s * v0 + c * v1
        
        if layer < layers - 1:
            # Apply CZ on adjacent pairs
            # CZ flips sign if both qubits are 1
            # For pair (q, q+1), mask = (1<<q) | (1<<(q+1))
            # indices where both bits are 1
            for q in range(n - 1):
                m1 = 1 << q
                m2 = 1 << (q + 1)
                mask_cz = m1 | m2
                idx_cz = np.arange(dim)
                # Check if both bits are set
                # (idx & m1) != 0 AND (idx & m2) != 0
                # (idx & mask_cz) == mask_cz
                idx_both = idx_cz[(idx_cz & mask_cz) == mask_cz]
                state[idx_both] = -state[idx_both]
    
    return state


def _energy(params, n, layers, H, e_ground):
    """Compute mean energy gap per spin."""
    state = _circuit_state(params, n, layers)
    e_circuit = np.real(np.vdot(state, H @ state))
    return (e_circuit - e_ground) / n


def _gradient(params, n, layers, H, e_ground):
    """Compute gradient of mean energy gap per spin using backpropagation (adjoint method)."""
    dim = 1 << n
    grad = np.zeros((layers, n), dtype=np.float64)
    
    # 1. Forward pass: Compute state and store intermediate states?
    # Or just recompute? Backprop requires storing states after each gate.
    # For n=6, dim=64, storing states is cheap.
    
    # We need to store the state after each Ry gate to compute the gradient.
    # The gradient of E = <psi|H|psi> with respect to theta_q at layer l is:
    # dE/dtheta = -2 * Im( <psi| (d/dtheta |psi>) H |psi> )? No.
    # Let |psi> = U |0>.
    # dE/dtheta = <dpsi|H|psi> + <psi|H|dpsi> = 2 Re( <dpsi|H|psi> )
    # d|psi>/dtheta_q = U_{layers > l} ... (dU_q/dtheta) U_{q-1}... |0>
    # This is expensive to compute directly for each q.
    
    # Adjoint method:
    # Let |phi> = H |psi>.
    # dE/dtheta_q = 2 Re( <psi| (dU/dtheta_q)^dag |phi> )?
    # Actually, dE/dtheta = 2 Re( <psi| H |dpsi> ).
    # |dpsi> = (dU/dtheta) |psi_before_gate>.
    # So dE/dtheta = 2 Re( <psi_before_gate| (dU/dtheta)^dag H |psi_after_gate> )?
    # Let's use the standard parameter-shift for gradient but vectorized?
    # No, parameter shift requires 2 circuit evaluations per parameter.
    # With 18 parameters (3 layers * 6 qubits), that's 36 evaluations.
    # Each evaluation is ~18 gates. Total ~648 gate applications.
    # For n=6, dim=64. 648 * 64 operations is trivial.
    # Wait, the previous failure was likely due to Python loops in `_circuit_state`.
    # The new `_circuit_state` is vectorized.
    # Let's stick to parameter-shift but ensure it's fast.
    # Or use analytic gradient via backprop which is 1 forward + 1 backward pass.
    
    # Let's implement analytic gradient via backprop.
    # Store states:
    # states[layer][q] is the state AFTER applying Ry(q) in layer `layer`.
    # We also need the state BEFORE applying Ry(q) in layer `layer`?
    # No, we can compute the gradient contribution locally.
    
    # Let |psi_l> be the state after layer l.
    # |psi_l> = CZ_l Ry_l |psi_{l-1}>
    # Actually, the circuit is:
    # Start |0>
    # Layer 0: Ry(0), Ry(1)... Ry(n-1)
    # CZ(0,1), CZ(1,2)...
    # Layer 1: Ry(0)...
    # CZ...
    # Layer 2: Ry(0)...
    
    # Let's denote the unitary for Ry(theta) on qubit q as U_q(theta).
    # dU_q(theta)/dtheta = -1/2 * (sin(theta/2) Z + cos(theta/2) X) ?
    # U = cos(theta/2) I - i sin(theta/2) X
    # dU/dtheta = -1/2 sin(theta/2) I - i/2 cos(theta/2) X
    # = -1/2 ( sin(theta/2) I + i cos(theta/2) X )
    
    # We want dE/dtheta_{l,q} = 2 Re( <psi| H |dpsi> )
    # |dpsi> = U_{after} (dU_{l,q}/dtheta) U_{before} |0>
    # Let |chi> = H |psi>.
    # dE/dtheta = 2 Re( <psi_before| U_{before}^dag U_{after}^dag H |psi_after> ... )?
    # Simpler:
    # Let |psi> be the final state.
    # Let |phi> = H |psi>.
    # We propagate |phi> backwards through the circuit.
    # At each gate, we compute the gradient contribution.
    
    # Forward pass:
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0
    
    # Store states after each Ry gate?
    # We need <psi_after_gate| H |psi_after_gate> ? No.
    # We need dE/dtheta = 2 Re( <psi| H |dpsi> ).
    # |dpsi> = U_{total}^{(l,q)} |0> where only U_{l,q} is differentiated.
    # This is equivalent to:
    # Let |psi_{l,q}^{after}> be the state after applying Ry_{l,q}.
    # Let |psi_{l,q}^{before}> be the state before applying Ry_{l,q}.
    # |dpsi> = U_{rest} (dU_{l,q}/dtheta) |psi_{l,q}^{before}>
    # dE/dtheta = 2 Re( <psi| H U_{rest} (dU_{l,q}/dtheta) |psi_{l,q}^{before}> )
    # = 2 Re( <psi_{l,q}^{before}| (dU_{l,q}/dtheta)^dag U_{rest}^dag H |psi> )
    
    # So we need:
    # 1. |psi> (final state)
    # 2. |psi_{l,q}^{before}> (state before each Ry)
    # 3. |chi_{l,q}> = U_{rest}^dag H |psi> (backpropagated H|psi>)
    
    # This requires storing all intermediate states (n * layers of them) and backpropagating.
    # For n=6, layers=3, that's 18 states of size 64. Very small.
    
    # Forward pass
    states_before = np.zeros((layers, n, dim), dtype=np.complex128)
    states_after = np.zeros((layers, n, dim), dtype=np.complex128)
    
    current_state = state.copy()
    
    for l in range(layers):
        for q in range(n):
            states_before[l, q] = current_state
            theta = params[l, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            mask = 1 << q
            idx = np.arange(dim)
            idx0 = idx[(idx & mask) == 0]
            idx1 = idx0 | mask
            
            v0 = current_state[idx0]
            v1 = current_state[idx1]
            
            current_state[idx0] = c * v0 - s * v1
            current_state[idx1] = s * v0 + c * v1
            
            states_after[l, q] = current_state
            
        if l < layers - 1:
            for q in range(n - 1):
                m1 = 1 << q
                m2 = 1 << (q + 1)
                mask_cz = m1 | m2
                idx_cz = np.arange(dim)
                idx_both = idx_cz[(idx_cz & mask_cz) == mask_cz]
                current_state[idx_both] = -current_state[idx_both]
    
    final_state = current_state
    
    # Backward pass
    # Start with |chi> = H |final_state>
    chi = H @ final_state
    
    # We need to backpropagate through the gates in reverse order.
    # The gradient for Ry(l, q) is:
    # dE/dtheta = 2 Re( <states_before[l,q]| (dU/dtheta)^dag |chi_at_gate> )
    # Where |chi_at_gate> is the backpropagated vector arriving at this gate from the right.
    
    # We iterate layers in reverse, and qubits in reverse?
    # The circuit order is:
    # Ry(0), Ry(1), ..., Ry(n-1), CZ(0,1), ..., CZ(n-2,n-1), Ry(0), ...
    # So we must reverse this exact sequence.
    
    # Let's store the CZ operations too? Or just re-apply them in reverse?
    # CZ is self-inverse and Hermitian.
    
    # Reverse order:
    # Last layer (l = layers - 1):
    # No CZ after last layer.
    # Reverse Ry(n-1), ..., Ry(0).
    # Then CZ(n-2, n-1), ..., CZ(0, 1).
    # Then previous layer Ry(n-1), ..., Ry(0).
    # ...
    
    # We need to update `chi` as we go backwards.
    # When passing through Ry(q, theta) backwards:
    # |chi_before> = U(theta)^dag |chi_after>
    # And we compute gradient using |chi_after> (which is the chi arriving at the gate from the right).
    
    # Wait, the formula was:
    # dE/dtheta_{l,q} = 2 Re( <psi_before| (dU/dtheta)^dag |chi_after> )
    # Where |chi_after> is the vector at the output of the gate in the backward pass.
    
    # Let's trace it:
    # E = <psi|H|psi>
    # dE = <dpsi|H|psi> + <psi|H|dpsi>
    # |dpsi> = U_{post} (dU_{l,q}) |psi_{pre}>
    # dE = <psi_{pre}| (dU_{l,q})^dag U_{post}^dag H |psi> + c.c.
    # Let |chi_{l,q}> = U_{post}^dag H |psi>.
    # dE = 2 Re( <psi_{pre}| (dU_{l,q})^dag |chi_{l,q}> )
    
    # So we need |chi_{l,q}> which is the backpropagated H|psi> arriving at the output of gate (l,q).
    
    # Initialize chi = H @ final_state
    chi = H @ final_state
    
    # We need to backpropagate through the entire circuit.
    # We'll iterate through the gates in reverse.
    
    # We need to know the state of `chi` at each gate.
    # Let's do it layer by layer.
    
    # Layer l = layers - 1
    # Gates: Ry(n-1), ..., Ry(0)
    # No CZ after.
    
    for l in range(layers - 1, -1, -1):
        # Reverse Ry gates
        for q in range(n - 1, -1, -1):
            # Compute gradient for this gate
            theta = params[l, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            
            # (dU/dtheta)^dag
            # dU/dtheta = -1/2 ( s I + i c X )
            # (dU/dtheta)^dag = -1/2 ( s I - i c X )
            # We need <psi_before| (dU/dtheta)^dag |chi>
            # Let |v> = (dU/dtheta)^dag |chi>
            # v = -1/2 ( s |chi> - i c X |chi> )
            
            # X |chi> flips qubit q.
            mask = 1 << q
            idx = np.arange(dim)
            idx0 = idx[(idx & mask) == 0]
            idx1 = idx0 | mask
            
            chi_c = chi.copy()
            # X operation on chi
            chi_x = chi.copy()
            # Swap idx0 and idx1
            temp = chi_x[idx0].copy()
            chi_x[idx0] = chi_x[idx1]
            chi_x[idx1] = temp
            
            # v = -0.5 * (s * chi - 1j * c * chi_x)
            v = -0.5 * (s * chi_c - 1j * c * chi_x)
            
            # Gradient = 2 * Re( <psi_before| v> )
            psi_before = states_before[l, q]
            grad[l, q] = 2.0 * np.real(np.vdot(psi_before, v)) / n # Divide by n for mean energy
            
            # Backpropagate chi through this gate:
            # chi_before = U(theta)^dag chi_after
            # U = c I - i s X
            # U^dag = c I + i s X
            # chi_before = c chi + i s X chi
            
            chi_new = c * chi + 1j * s * chi_x
            chi = chi_new
        
        # If not the first layer, backpropagate through CZ gates
        if l > 0:
            # Reverse CZ gates: CZ(n-2, n-1), ..., CZ(0, 1)
            for q in range(n - 2, -1, -1):
                m1 = 1 << q
                m2 = 1 << (q + 1)
                mask_cz = m1 | m2
                idx_cz = np.arange(dim)
                idx_both = idx_cz[(idx_cz & mask_cz) == mask_cz]
                
                # CZ is diagonal: diag(1, 1, 1, -1) in local basis
                # Backpropagation: chi_before = CZ^dag chi_after = CZ chi_after
                # So we just flip sign of components where both bits are 1.
                chi[idx_both] = -chi[idx_both]
    
    return grad


def angles(n, coupling, field, layers):
    """Optimize circuit parameters to minimize mean energy gap per spin."""
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    H = _build_hamiltonian(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    e_ground = eigvals[0]
    
    def obj(params):
        return _energy(params, n, layers, H, e_ground)
    
    def grad(params):
        return _gradient(params, n, layers, H, e_ground)
    
    ratio = field / coupling
    
    np.random.seed(42)
    initial_guesses = []
    
    if ratio < 0.5:
        base = 0.3 * ratio
    elif ratio < 1.0:
        base = 0.6 * np.arctan(ratio)
    elif ratio < 2.0:
        base = np.pi / 4
    else:
        base = 0.4 * np.pi
    
    initial_guesses.append(np.full((layers, n), base))
    
    alt = np.zeros((layers, n))
    for l in range(layers):
        for q in range(n):
            alt[l, q] = base if q % 2 == 0 else -base
    initial_guesses.append(alt)
    
    initial_guesses.append(np.zeros((layers, n)))
    initial_guesses.append(np.full((layers, n), np.pi / 2))
    
    for i in range(5):
        g = np.full((layers, n), base) + np.random.normal(0, 0.5, (layers, n))
        initial_guesses.append(g)
    
    best_params = initial_guesses[0].copy()
    best_val = obj(best_params)
    
    try:
        from scipy.optimize import minimize
        
        for guess in initial_guesses:
            if time.time() > deadline - 20:
                break
            
            x_flat = guess.flatten()
            
            def obj_flat(xf):
                return obj(xf.reshape((layers, n)))
            
            def grad_flat(xf):
                p = xf.reshape((layers, n))
                return grad(p).flatten()
            
            try:
                res = minimize(obj_flat, x_flat, jac=grad_flat, method='L-BFGS-B',
                              options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
                if res.fun < best_val:
                    best_val = res.fun
                    best_params = res.x.reshape((layers, n))
            except Exception:
                pass
            
            if time.time() > deadline - 10:
                continue
            
            try:
                res2 = minimize(obj_flat, best_params.flatten(), method='Nelder-Mead',
                               options={'maxiter': 50, 'xatol': 1e-8, 'fatol': 1e-10})
                if res2.fun < best_val:
                    best_val = res2.fun
                    best_params = res2.x.reshape((layers, n))
            except Exception:
                pass
    
    except ImportError:
        pass
    
    return best_params
# EVOLVE-BLOCK-END
