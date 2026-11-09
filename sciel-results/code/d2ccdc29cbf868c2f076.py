"""Direct structure-derived QAOA angle design with analytic gradient optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 145.0

def _time_left():
    return _DEADLINE - time.time()

def _precompute_cut_sizes(n, edges):
    """Precompute cut size for all 2^n states vectorized."""
    dim = 1 << n
    cut_sizes = np.zeros(dim, dtype=np.float64)
    for (i, j) in edges:
        states = np.arange(dim, dtype=np.int64)
        bi = (states >> i) & 1
        bj = (states >> j) & 1
        cut_sizes += (bi != bj).astype(np.float64)
    return cut_sizes

def _apply_x_layer(state, beta, n):
    """Apply exp(-i beta sum_i X_i) via tensor reshape per qubit."""
    c = np.cos(beta)
    s = np.sin(beta)
    tensor = state.reshape([2] * n)
    for i in range(n):
        moved = np.moveaxis(tensor, i, 0)
        flat = moved.reshape(2, -1)
        new_flat = np.empty_like(flat)
        new_flat[0] = c * flat[0] - 1j * s * flat[1]
        new_flat[1] = -1j * s * flat[0] + c * flat[1]
        tensor = np.moveaxis(new_flat.reshape([2] * n), 0, i)
    return tensor.flatten()

def _qaoa_state_and_grads(n, cut_sizes, angles):
    """
    Simulate p=2 QAOA circuit and compute analytic gradients.
    angles: array of shape (2, 2) -> [[gamma1, beta1], [gamma2, beta2]]
    Returns: expected_cut, gradient_vector (shape 4)
    """
    dim = 1 << n
    gamma1, beta1 = angles[0]
    gamma2, beta2 = angles[1]
    
    # Initial state |+>^n
    psi0 = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Layer 1: U_C(gamma1) then U_B(beta1)
    # phi1 = U_C(gamma1) psi0
    phi1 = psi0 * np.exp(-1j * gamma1 * cut_sizes)
    
    # d_phi1/d_gamma1 = -i * cut_sizes * phi1
    d_phi1_d_g1 = -1j * cut_sizes * phi1
    
    # psi1 = U_B(beta1) phi1
    psi1 = _apply_x_layer(phi1, beta1, n)
    
    # d_psi1/d_gamma1 = U_B(beta1) d_phi1/d_gamma1
    d_psi1_d_g1 = _apply_x_layer(d_phi1_d_g1, beta1, n)
    
    # d_psi1/d_beta1 = -i * B * psi1
    # Note: B = sum X_i. Applying -i B is equivalent to applying the derivative of U_B.
    # U_B(beta) = exp(-i beta B). d/d_beta U_B = -i B U_B.
    # So d_psi1/d_beta1 = -i B psi1.
    # We can compute this by applying the X layer logic but with derivative.
    # Actually, simpler: 
    # psi1 = U_B(beta1) phi1.
    # d_psi1/d_beta1 = -i B psi1.
    # We can compute B psi1 by applying the X layer with a specific trick or just using the fact that 
    # if we define _apply_x_layer as U_B, then we need the operator B.
    # However, we can just compute the gradient of the expectation value using the state derivatives.
    
    # Let's compute d_psi1_d_b1 properly.
    # psi1 = U_B(beta1) phi1.
    # d_psi1/d_beta1 = -i B psi1.
    # We can compute B psi1 by noting that B = sum X_i.
    # Alternatively, we can just use the chain rule in the gradient calculation step.
    # Let's stick to propagating the state derivatives.
    
    # To get d_psi1_d_b1, we can use the fact that:
    # d/d_beta exp(-i beta B) |psi> = -i B exp(-i beta B) |psi>
    # So we need to apply B to psi1.
    # Applying B is equivalent to applying the X layer with a "derivative" mode?
    # No, B is the Hamiltonian. 
    # We can compute B psi1 by applying the same tensor logic as _apply_x_layer but 
    # mapping |0> -> |1> and |1> -> |0> (which is X) and summing? No, B = sum X_i.
    # X_i flips bit i.
    # So (B psi)_k = sum_i (psi with bit i flipped).
    
    def _apply_B(state, n):
        """Apply B = sum_i X_i to the state."""
        dim = 1 << n
        result = np.zeros_like(state)
        for i in range(n):
            # Flip bit i
            # Create a permutation or just use reshape
            # Easier: iterate over states? No, vectorize.
            # For each state index, flip bit i and add to result.
            # This is O(n * 2^n). For n=10, 10*1024 = 10k ops. Very fast.
            pass
            
        # Vectorized approach for B:
        # B = sum_i X_i.
        # X_i can be applied via reshape.
        # Let's just implement _apply_x_layer for B? No, B is not a unitary.
        # But we can compute B psi by applying X_i for each i and summing.
        
        b_psi = np.zeros_like(state)
        for i in range(n):
            # Apply X_i
            tensor = state.reshape([2] * n)
            moved = np.moveaxis(tensor, i, 0)
            flat = moved.reshape(2, -1)
            # X = [[0, 1], [1, 0]]
            new_flat = np.empty_like(flat)
            new_flat[0] = flat[1]
            new_flat[1] = flat[0]
            x_i_psi = np.moveaxis(new_flat.reshape([2] * n), 0, i).flatten()
            b_psi += x_i_psi
        return b_psi

    d_psi1_d_b1 = -1j * _apply_B(psi1, n)
    
    # Layer 2: U_C(gamma2) then U_B(beta2)
    # phi2 = U_C(gamma2) psi1
    phi2 = psi1 * np.exp(-1j * gamma2 * cut_sizes)
    
    # d_phi2/d_gamma2 = -i * cut_sizes * phi2
    d_phi2_d_g2 = -1j * cut_sizes * phi2
    
    # d_phi2/d_gamma1 = U_C(gamma2) d_psi1/d_gamma1
    d_phi2_d_g1 = d_psi1_d_g1 * np.exp(-1j * gamma2 * cut_sizes)
    
    # d_phi2/d_beta1 = U_C(gamma2) d_psi1/d_beta1
    d_phi2_d_b1 = d_psi1_d_b1 * np.exp(-1j * gamma2 * cut_sizes)
    
    # psi2 = U_B(beta2) phi2
    psi2 = _apply_x_layer(phi2, beta2, n)
    
    # d_psi2/d_gamma2 = U_B(beta2) d_phi2/d_gamma2
    d_psi2_d_g2 = _apply_x_layer(d_phi2_d_g2, beta2, n)
    
    # d_psi2/d_gamma1 = U_B(beta2) d_phi2/d_gamma1
    d_psi2_d_g1 = _apply_x_layer(d_phi2_d_g1, beta2, n)
    
    # d_psi2/d_beta1 = U_B(beta2) d_phi2/d_beta1
    d_psi2_d_b1 = _apply_x_layer(d_phi2_d_b1, beta2, n)
    
    # d_psi2/d_beta2 = -i B psi2
    d_psi2_d_b2 = -1j * _apply_B(psi2, n)
    
    # Expected cut E = <psi2| C |psi2> = sum P(x) C(x)
    probs = np.abs(psi2) ** 2
    E = float(np.sum(probs * cut_sizes))
    
    # Gradients: dE/dtheta = 2 Re(<d_psi/dtheta | C | psi>)
    # Note: <d_psi| C | psi> = conj(d_psi) dot (C * psi)
    c_psi = cut_sizes * psi2
    
    grad_g1 = 2.0 * np.real(np.vdot(d_psi2_d_g1, c_psi))
    grad_b1 = 2.0 * np.real(np.vdot(d_psi2_d_b1, c_psi))
    grad_g2 = 2.0 * np.real(np.vdot(d_psi2_d_g2, c_psi))
    grad_b2 = 2.0 * np.real(np.vdot(d_psi2_d_b2, c_psi))
    
    grad = np.array([grad_g1, grad_b1, grad_g2, grad_b2])
    
    return E, grad

def _objective_and_grad(params, n, cut_sizes):
    """Negative expected cut and its gradient for minimization."""
    angles = params.reshape(2, 2)
    E, grad = _qaoa_state_and_grads(n, cut_sizes, angles)
    return -E, -grad

def _graph_invariant_init(n, edges, p):
    """
    Direct initialization from graph structure.
    """
    m = len(edges)
    max_edges = n * (n - 1) // 2
    density = m / max_edges if max_edges > 0 else 0.0
    
    # Average degree
    avg_deg = 2.0 * m / n if n > 0 else 0.0
    
    # Degree variance
    deg = np.zeros(n)
    for (i, j) in edges:
        deg[i] += 1
        deg[j] += 1
    deg_var = np.var(deg)
    
    gamma1 = np.pi * 0.35 + 0.3 * density
    gamma2 = np.pi * 0.15 + 0.2 * density
    
    beta1 = np.pi * 0.25 - 0.1 * density
    beta2 = np.pi * 0.15
    
    scale = 1.0 + 0.1 * (deg_var / (avg_deg + 1e-9)) if avg_deg > 0 else 1.0
    gamma1 *= scale
    gamma2 *= scale
    
    angles = np.array([
        [gamma1 % (2 * np.pi), beta1 % (2 * np.pi)],
        [gamma2 % (2 * np.pi), beta2 % (2 * np.pi)]
    ], dtype=np.float64)
    return angles

def angles(n, edges, p):
    """Design graph-dependent two-layer QAOA angles."""
    global _DEADLINE
    _DEADLINE = time.time() + 145.0
    
    cut_sizes = _precompute_cut_sizes(n, edges)
    
    best_angles = None
    best_score = -np.inf
    
    # Build diverse initial guesses
    initial_guesses = []
    
    # Structure-derived direct init
    init_struct = _graph_invariant_init(n, edges, p)
    initial_guesses.append(init_struct.flatten())
    
    # Perturbed structure-derived inits
    rng = np.random.default_rng(42)
    for _ in range(10):
        noise_g = rng.normal(0, 0.2, p)
        noise_b = rng.normal(0, 0.2, p)
        perturbed = init_struct.copy()
        perturbed[:, 0] = (perturbed[:, 0] + noise_g) % (2 * np.pi)
        perturbed[:, 1] = (perturbed[:, 1] + noise_b) % (2 * np.pi)
        initial_guesses.append(perturbed.flatten())
        
    # Standard literature-based anchors
    initial_guesses.append(np.array([0.35, 0.25, 0.25, 0.15]))
    initial_guesses.append(np.array([0.5, 0.3, 0.3, 0.2]))
    initial_guesses.append(np.array([0.7, 0.2, 0.4, 0.15]))
    initial_guesses.append(np.array([np.pi/4, np.pi/8, np.pi/8, np.pi/16]))
    
    # Random uniform inits
    for _ in range(15):
        g = rng.uniform(0, np.pi, p)
        b = rng.uniform(0, np.pi, p)
        initial_guesses.append(np.column_stack((g, b)).flatten())
    
    # Multi-restart L-BFGS-B with analytic gradients
    for idx, x0 in enumerate(initial_guesses):
        if _time_left() < 5.0:
            break
        
        try:
            res = minimize(
                _objective_and_grad, x0, args=(n, cut_sizes),
                method='L-BFGS-B',
                jac=True, # Use the gradient returned by the objective
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            score = -res.fun
            if score > best_score:
                best_score = score
                best_angles = res.x.reshape(p, 2)
        except Exception:
            # Fallback to Nelder-Mead if gradient method fails
            try:
                res_nm = minimize(
                    lambda x: _qaoa_state_and_grads(n, cut_sizes, x.reshape(2,2))[0] * -1,
                    x0, method='Nelder-Mead', options={'maxiter': 50}
                )
                score = -res_nm.fun
                if score > best_score:
                    best_score = score
                    best_angles = res_nm.x.reshape(p, 2)
            except Exception:
                pass
    
    # Final high-precision polish
    if best_angles is not None and _time_left() > 10.0:
        try:
            res = minimize(
                _objective_and_grad, best_angles.flatten(), args=(n, cut_sizes),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-10, 'maxls': 100}
            )
            if -res.fun > best_score:
                best_score = -res.fun
                best_angles = res.x.reshape(p, 2)
        except Exception:
            pass
    
    # Fallback
    if best_angles is None:
        best_angles = np.tile([0.35, 0.25], (p, 1))
    
    return best_angles
# EVOLVE-BLOCK-END
