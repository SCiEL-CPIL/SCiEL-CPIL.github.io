import numpy as np
import time

def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    dim = 1 << n
    
    # 1. Build Hamiltonian H
    H = np.zeros((dim, dim), dtype=float)
    # ZZ terms: H_ii += -coupling * Z_i * Z_{i+1}
    for i in range(dim):
        zz_val = 1.0
        for a in range(n - 1):
            bit_a = (i >> a) & 1
            bit_b = (i >> (a + 1)) & 1
            z_a = 1.0 if bit_a == 0 else -1.0
            z_b = 1.0 if bit_b == 0 else -1.0
            zz_val *= z_a * z_b
        H[i, i] += -coupling * zz_val
    
    # X terms: H_{i, i^2^q} += -field
    for i in range(dim):
        for q in range(n):
            j = i ^ (1 << q)
            H[i, j] += -field
    
    # 2. Ground state energy
    eigvals = np.linalg.eigvalsh(H)
    e0 = eigvals[0]
    
    # 3. Precompute phase mask for CZ gates
    # The circuit applies CZ(0,1), CZ(1,2), ..., CZ(n-2, n-1) in sequence between layers.
    # Since they are all diagonal, they commute. We can combine them into a single phase vector.
    # Phase is -1 if ANY adjacent pair (a, a+1) are both 1.
    # Note: The task says "CZ on every adjacent pair in ascending order".
    # Since CZ is diagonal, the order doesn't matter for the final state vector, only the product of phases.
    # Phase(i) = (-1)^(k) where k is number of adjacent pairs (a, a+1) such that bit_a=1 and bit_{a+1}=1.
    # Wait, CZ applies phase -1 if BOTH are 1.
    # So if multiple CZs apply to the same state component, the phases multiply.
    # Example: n=3, state |111>. CZ(0,1) gives -1. CZ(1,2) gives -1. Total phase (+1).
    # So we need to count the number of "11" blocks mod 2.
    
    phase_mask = np.ones(dim, dtype=float)
    for i in range(dim):
        count = 0
        for a in range(n - 1):
            if ((i >> a) & 1) and ((i >> (a + 1)) & 1):
                count += 1
        if count % 2 == 1:
            phase_mask[i] = -1.0
    
    # 4. Helper: Apply Ry(theta) to qubit q on state vector (vectorized)
    # State is reshaped to (2, 2, ..., 2) with qubit q as the first axis?
    # No, standard bit ordering: i = sum b_k 2^k.
    # Qubit 0 is LSB.
    # Let's use a helper that takes the flat state and applies Ry to qubit q.
    
    def apply_ry(state, q, theta):
        """
        Apply Ry(theta) to qubit q.
        state: flat array of size dim.
        Returns new flat array.
        """
        # Reshape to tensor of shape (2, 2, ..., 2)
        # Axis order: qubit 0 is axis 0?
        # i = b_0 + 2*b_1 + ... + 2^(n-1)*b_{n-1}
        # In numpy reshape, the last index changes fastest.
        # So if we reshape to (2,)*n, the last axis corresponds to b_{n-1} (MSB).
        # Axis 0 corresponds to b_0 (LSB) if we reverse?
        # Let's check: i=1 (01...0). b_0=1.
        # Reshape (2,)*n: index [1, 0, ..., 0] -> 1 * 2^0? No.
        # Numpy C-order: index [i0, i1, ..., in-1] maps to i0 * 2^(n-1) + ... + in-1.
        # So axis 0 is MSB (qubit n-1), axis n-1 is LSB (qubit 0).
        # We want to operate on qubit q.
        # Axis corresponding to qubit q is n-1-q.
        
        tensor = state.reshape((2,) * n)
        
        # Move target qubit to front for easy slicing?
        # Or just use transpose.
        # Let's move axis (n-1-q) to 0.
        target_axis = n - 1 - q
        tensor = np.moveaxis(tensor, target_axis, 0)
        
        # Now tensor[0, ...] is state with qubit q = 0
        # tensor[1, ...] is state with qubit q = 1
        
        c = np.cos(theta / 2.0)
        s = np.sin(theta / 2.0)
        
        # Ry(theta) = [[c, -s], [s, c]]
        # new_0 = c * old_0 - s * old_1
        # new_1 = s * old_0 + c * old_1
        
        new_tensor = np.empty_like(tensor)
        new_tensor[0] = c * tensor[0] - s * tensor[1]
        new_tensor[1] = s * tensor[0] + c * tensor[1]
        
        # Move axis back
        new_tensor = np.moveaxis(new_tensor, 0, target_axis)
        
        return new_tensor.reshape(dim)

    # 5. Objective and Gradient
    # Energy E = <psi|H|psi>
    # dE/dtheta_q = 2 * Re( <psi| (d|psi>/dtheta_q) H + h.c. > )?
    # Actually, dE/dtheta = 2 Re( <dpsi/dtheta| H |psi> )
    # Since H is Hermitian and real (in this basis), and state is real?
    # Ry is real. CZ is real. Start state is real. So state is always real.
    # H is real symmetric.
    # So E = psi^T H psi.
    # dE/dtheta = 2 psi^T H (dpsi/dtheta)
    
    def compute_grad(params):
        """
        params: shape (layers, n)
        Returns gradient of E w.r.t params (same shape)
        """
        # Forward pass to get state
        psi = np.zeros(dim, dtype=float)
        psi[0] = 1.0
        
        # We need dpsi/dtheta for each layer/qubit.
        # Storing all dpsi/dtheta is expensive (layers*n*dim).
        # dim is 16 or 64. layers=3, n=4 or 6.
        # 3*6*64 = 1152 floats. Very small.
        # So we can just store the gradients of the state for each parameter.
        
        grads = np.zeros((layers, n), dtype=float)
        
        # Store states after each layer?
        # Let's do forward pass and store the state after each Ry application?
        # No, just store the state before the Ry for each (layer, q) and the state after?
        # Actually, we can just recompute or store.
        # Given small dim, storing `layers * n` state vectors is fine.
        
        psi_before_ry = np.zeros((layers, n, dim), dtype=float)
        
        # Forward pass
        for l in range(layers):
            for q in range(n):
                psi_before_ry[l, q] = psi.copy()
                theta = params[l, q]
                psi = apply_ry(psi, q, theta)
            
            if l < layers - 1:
                psi = psi * phase_mask
        
        # Backward pass (or just compute gradients directly using stored states)
        # dE/dtheta_{l,q} = 2 * (H psi)_dot (dpsi/dtheta_{l,q})
        # dpsi/dtheta_{l,q} is the derivative of the state at step (l,q) propagated to the end.
        # This is tricky because later gates affect the final state.
        # We need the adjoint method or just finite differences?
        # Finite differences are slow but dim is small.
        # However, analytic gradient is requested in inspiration 1.
        
        # Let's use the "tape" method.
        # We need the gradient of the final state w.r.t each parameter.
        # This is equivalent to running the circuit backwards with the "adjoint" state.
        
        # Let's try a simpler approach for small dim:
        # Since dim <= 64, we can just compute the Jacobian of the state w.r.t params?
        # Or use central differences for the gradient?
        # 3*n params. n=6 -> 18 params. 2*18=36 evaluations.
        # Each evaluation is O(dim^2) roughly? No, O(dim * n * layers).
        # 36 * 64 * 6 * 3 ~ 40k ops. Very fast.
        # Let's use central differences for robustness and simplicity, given the small size.
        
        eps = 1e-5
        grad = np.zeros_like(params)
        
        def energy_of(p):
            psi = np.zeros(dim, dtype=float)
            psi[0] = 1.0
            for l in range(layers):
                for q in range(n):
                    psi = apply_ry(psi, q, p[l, q])
                if l < layers - 1:
                    psi = psi * phase_mask
            return psi @ H @ psi
        
        # Central difference gradient
        for l in range(layers):
            for q in range(n):
                p_plus = params.copy()
                p_minus = params.copy()
                p_plus[l, q] += eps
                p_minus[l, q] -= eps
                grad[l, q] = (energy_of(p_plus) - energy_of(p_minus)) / (2 * eps)
        
        return grad

    # 6. Optimization Loop
    # Multi-restart gradient descent with adaptive step size (Adam-like or simple backtracking)
    
    def optimize(params_init):
        params = params_init.copy()
        lr = 0.1
        best_e = energy_of(params)
        best_params = params.copy()
        
        # Adam optimizer
        m = np.zeros_like(params)
        v = np.zeros_like(params)
        beta1 = 0.9
        beta2 = 0.999
        t = 0
        
        for step in range(500):
            if time.time() > deadline - 5.0:
                break
                
            t += 1
            grad = compute_grad(params)
            
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)
            
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            
            params = params - lr * m_hat / (np.sqrt(v_hat) + 1e-8)
            
            e = energy_of(params)
            if e < best_e:
                best_e = e
                best_params = params.copy()
                
            # Early stopping if converged
            if np.max(np.abs(grad)) < 1e-6:
                break
                
        return best_params, best_e

    # Initialize multiple restarts
    restarts = 5
    best_global_params = None
    best_global_val = float('inf')
    
    # Restart 0: All zeros (or small)
    inits = [np.zeros((layers, n))]
    
    # Restart 1: Pi/2 on last layer (seed-like)
    p1 = np.zeros((layers, n))
    p1[-1] = np.pi / 2
    inits.append(p1)
    
    # Rest: Random
    np.random.seed(42)
    for i in range(restarts - 2):
        inits.append(np.random.uniform(-np.pi, np.pi, size=(layers, n)))
    
    for i, init in enumerate(inits):
        if time.time() > deadline - 10.0:
            break
            
        res_params, res_val = optimize(init)
        
        if res_val < best_global_val:
            best_global_val = res_val
            best_global_params = res_params.copy()
            
    return best_global_params
