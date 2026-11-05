"""QAOA parameter design with vectorized simulation + analytic gradient + multi-start optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize


def _precompute_cut_values(n, edges):
    """
    Precompute the cut value for every possible state mask.
    Returns an array of shape (2^n,).
    """
    dim = 1 << n
    cut_vals = np.zeros(dim, dtype=np.float64)
    for mask in range(dim):
        val = 0
        for u, v in edges:
            if ((mask >> u) ^ (mask >> v)) & 1:
                val += 1
        cut_vals[mask] = val
    return cut_vals


def _apply_mixer_fast(psi, n, beta):
    """
    Apply exp(-i * beta * sum(X_i)) to the state vector psi.
    Uses reshape and per-axis rotation with moveaxis for efficiency.
    """
    cb = np.cos(beta)
    sb = -1j * np.sin(beta)
    
    shape = (2,) * n
    psi_t = psi.reshape(shape)
    
    for i in range(n):
        psi_moved = np.moveaxis(psi_t, i, 0)
        half_0 = psi_moved[0].copy()
        half_1 = psi_moved[1].copy()
        psi_moved[0] = cb * half_0 + sb * half_1
        psi_moved[1] = sb * half_0 + cb * half_1
        psi_t = np.moveaxis(psi_moved, 0, i)
    
    return psi_t.reshape(-1)


def _apply_mixer_dagger_fast(psi, n, beta):
    """Apply exp(+i * beta * sum(X_i)) to the state vector psi."""
    return _apply_mixer_fast(psi, n, -beta)


def _apply_mixer_h(psi, n):
    """
    Apply H_mixer = sum(X_i) to psi.
    X_i flips bit i.
    """
    shape = (2,) * n
    psi_tensor = psi.reshape(shape)
    result = np.zeros_like(psi_tensor)
    
    for i in range(n):
        slice_0 = [slice(None)] * n
        slice_1 = [slice(None)] * n
        slice_0[i] = 0
        slice_1[i] = 1
        
        result[tuple(slice_0)] += psi_tensor[tuple(slice_1)]
        result[tuple(slice_1)] += psi_tensor[tuple(slice_0)]
        
    return result.reshape(-1)


def _qaoa_expectation_and_grad(n, gammas, betas, cut_vals):
    """
    Compute expectation and gradient using backpropagation through the QAOA circuit.
    gammas, betas: arrays of length p.
    """
    p = len(gammas)
    dim = 1 << n
    psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Store states
    states = [psi]
    
    for k in range(p):
        # Cost
        psi = psi * np.exp(-1j * gammas[k] * cut_vals)
        states.append(psi)
        # Mixer
        psi = _apply_mixer_fast(psi, n, betas[k])
        states.append(psi)
        
    psi_final = states[-1]
    exp_cut = float(np.vdot(psi_final, cut_vals * psi_final).real)
    
    # Backpropagation
    chi = cut_vals * psi_final
    grads = np.zeros(2 * p)
    
    for k in range(p - 1, -1, -1):
        # Mixer gradient (beta_k)
        psi_after_mixer = states[2 * k + 2]
        h_psi_after = _apply_mixer_h(psi_after_mixer, n)
        
        chi_before_mixer = _apply_mixer_dagger_fast(chi, n, betas[k])
        
        inner = np.vdot(chi_before_mixer, h_psi_after)
        grads[2 * k + 1] = 2.0 * inner.imag
        
        # Cost gradient (gamma_k)
        psi_after_cost = states[2 * k + 1]
        
        chi_before_cost = chi_before_mixer * np.exp(1j * gammas[k] * cut_vals)
        
        c_psi_after = cut_vals * psi_after_cost
        inner_g = np.vdot(chi_before_cost, c_psi_after)
        grads[2 * k] = 2.0 * inner_g.imag
        
        chi = chi_before_cost
        
    return exp_cut, grads


def _qaoa_expectation(n, gammas, betas, cut_vals):
    """Compute expected cut size."""
    dim = 1 << n
    psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    for k in range(len(gammas)):
        psi *= np.exp(-1j * gammas[k] * cut_vals)
        psi = _apply_mixer_fast(psi, n, betas[k])
        
    probs = np.abs(psi) ** 2
    return float(np.dot(probs, cut_vals))


def _optimize_angles(n, edges, p, start_time, timeout):
    """
    Optimize angles using multi-start L-BFGS-B with analytic gradients,
    followed by Nelder-Mead polish.
    """
    cut_vals = _precompute_cut_values(n, edges)
    
    def objective(params):
        try:
            gammas = np.array([params[0], params[2]])
            betas = np.array([params[1], params[3]])
            return -_qaoa_expectation(n, gammas, betas, cut_vals)
        except Exception:
            return 1e9

    def objective_grad(params):
        try:
            gammas = np.array([params[0], params[2]])
            betas = np.array([params[1], params[3]])
            exp_cut, grads = _qaoa_expectation_and_grad(n, gammas, betas, cut_vals)
            return -exp_cut, -grads
        except Exception:
            return 1e9, np.zeros(4)

    best_score = -np.inf
    best_params = None
    
    # Estimate evaluation time
    t0 = time.time()
    _qaoa_expectation(n, np.array([0.5, 0.5]), np.array([0.5, 0.5]), cut_vals)
    t1 = time.time()
    eval_time = max(t1 - t0, 0.0001)
    
    # Each gradient call: 1 func + 1 grad (which is ~1 func + 1 backward pass)
    # L-BFGS-B iteration: ~1 func eval + 1 grad eval
    # 80 iterations per restart: ~160 evals
    time_per_restart = 160 * eval_time
    available_time = max(timeout - (time.time() - start_time) - 15, 1.0)
    num_restarts = max(5, min(200, int(available_time / max(time_per_restart, 0.001))))
    
    # Cap based on eval speed
    if eval_time < 0.001:
        num_restarts = min(num_restarts, 150)
    elif eval_time < 0.005:
        num_restarts = min(num_restarts, 80)
    else:
        num_restarts = min(num_restarts, 40)
    
    np.random.seed(42)
    
    # Good initial points
    base_inits = [
        np.array([0.5, 0.5, 0.3, 0.3]),
        np.array([0.7, 0.7, 0.4, 0.4]),
        np.array([0.3, 0.4, 0.2, 0.3]),
        np.array([0.5, 0.3, 0.5, 0.3]),
        np.array([0.4, 0.6, 0.4, 0.6]),
        np.array([0.6, 0.2, 0.6, 0.2]),
        np.array([0.55, 0.45, 0.35, 0.35]),
        np.array([0.65, 0.55, 0.45, 0.45]),
        np.array([0.45, 0.55, 0.25, 0.45]),
        np.array([0.5, 0.5, 0.5, 0.5]),
        np.array([0.35, 0.25, 0.35, 0.25]),
        np.array([0.5, 0.3, 0.4, 0.25]),
        np.array([0.6, 0.4, 0.5, 0.3]),
    ]
    
    # Add random initializations
    for _ in range(num_restarts - len(base_inits)):
        base_inits.append(np.random.uniform(0.1, 1.5, size=4))
    
    bounds = [(0.0, np.pi)] * 4
    
    for r in range(num_restarts):
        if time.time() - start_time > timeout - 15:
            break
            
        init = base_inits[r % len(base_inits)]
        
        try:
            res = scipy.optimize.minimize(
                objective_grad, init, jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 80, 'ftol': 1e-9, 'gtol': 1e-6, 'maxls': 30, 'disp': False}
            )
            score = -res.fun
            if score > best_score:
                best_score = score
                best_params = res.x.copy()
        except Exception:
            continue
        
        # Nelder-Mead polish on the best found so far
        if r % 5 == 0 and best_params is not None and time.time() - start_time < timeout - 20:
            try:
                nm_res = scipy.optimize.minimize(
                    objective, best_params,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'disp': False}
                )
                nm_score = -nm_res.fun
                if nm_score > best_score:
                    best_score = nm_score
                    best_params = nm_res.x.copy()
            except Exception:
                pass
    
    if best_params is not None:
        # Clamp to valid range
        best_params = np.clip(best_params, 0.0, np.pi)
        return np.array([[best_params[0], best_params[1]],
                         [best_params[2], best_params[3]]])
    else:
        return np.array([[0.35, 0.25], [0.35, 0.25]])


def angles(n, edges, p):
    start_time = time.time()
    timeout = 135  # Safety margin before 150s kill
    
    optimal_angles = _optimize_angles(n, edges, p, start_time, timeout)
    
    elapsed = time.time() - start_time
    if elapsed > timeout:
        return np.array([[0.35, 0.25]] * p)
        
    return optimal_angles
# EVOLVE-BLOCK-END
