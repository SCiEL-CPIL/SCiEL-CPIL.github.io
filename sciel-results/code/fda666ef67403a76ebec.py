import time
import numpy as np

_DEADLINE = 140.0  # Safety margin before 150s kill


def _build_operators(n_sites=6, anisotropy=0.8, field=0.3):
    """
    Build A (odd bonds + half field) and B (even bonds + half field) for a 6-site
    open XXZ Heisenberg chain.
    """
    dim = 2 ** n_sites

    sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    identity = np.eye(2, dtype=complex)

    def kron_all(ops_list):
        result = np.array([[1.0]], dtype=complex)
        for op in ops_list:
            result = np.kron(result, op)
        return result

    def bond_operator(j, gamma, h_half):
        ops = [identity] * n_sites
        ops[j] = sx
        ops[j + 1] = sx
        term_xx = kron_all(ops)

        ops = [identity] * n_sites
        ops[j] = sy
        ops[j + 1] = sy
        term_yy = kron_all(ops)

        ops = [identity] * n_sites
        ops[j] = sz
        ops[j + 1] = sz
        term_zz = gamma * kron_all(ops)

        ops = [identity] * n_sites
        ops[j] = sz
        term_field_j = (-h_half) * kron_all(ops)

        ops = [identity] * n_sites
        ops[j + 1] = sz
        term_field_j1 = (-h_half) * kron_all(ops)

        return term_xx + term_yy + term_zz + term_field_j + term_field_j1

    A = np.zeros((dim, dim), dtype=complex)
    for j in [0, 2, 4]:
        A += bond_operator(j, anisotropy, field / 2)

    B = np.zeros((dim, dim), dtype=complex)
    for j in [1, 3]:
        B += bond_operator(j, anisotropy, field / 2)

    return A, B


def _compute_exact_propagators(A, B, t_values):
    """Compute exact propagators exp(-i t (A+B)) for given t values."""
    H = A + B
    eigvals, eigvecs = np.linalg.eigh(H)
    exacts = []
    for t in t_values:
        exact = eigvecs @ np.diag(np.exp(-1j * t * eigvals)) @ eigvecs.conj().T
        exacts.append(exact)
    return exacts


def _symmetric_to_full(half_a, half_b):
    """
    Convert 6 half-coefficients to 12 full coefficients using palindromic symmetry.
    a_k = a_{13-k}, b_k = b_{13-k}
    half_a = [a1, a2, a3, a4, a5, a6] -> a = [a1, a2, a3, a4, a5, a6, a6, a5, a4, a3, a2, a1]
    """
    a_full = np.array([half_a[0], half_a[1], half_a[2], half_a[3], half_a[4], half_a[5],
                       half_a[5], half_a[4], half_a[3], half_a[2], half_a[1], half_a[0]])
    b_full = np.array([half_b[0], half_b[1], half_b[2], half_b[3], half_b[4], half_b[5],
                       half_b[5], half_b[4], half_b[3], half_b[2], half_b[1], half_b[0]])
    return a_full, b_full


def _get_suzuki_4th_order():
    """
    Return initial symmetric parameters based on 4th-order Suzuki-Trotter splitting.
    For 4th order: c1 = 1/(2-2^(1/3)), c0 = -c1/2
    """
    c1 = 1.0 / (2.0 - 2.0 ** (1.0/3.0))
    c0 = -c1 / 2.0
    
    # Use a reasonable initialization with some structure
    # Start with uniform and let optimization refine
    half_a = np.array([1.0/12] * 6)
    half_b = np.array([1.0/12] * 6)
    
    return np.concatenate([half_a, half_b])


def _eval_error_symmetric(params, eig_data_A, eig_data_B, exacts, t_values):
    """
    Evaluate error for symmetric parameters.
    params: array of 12 floats = [a1..a6, b1..b6]
    """
    half_a = params[:6]
    half_b = params[6:]
    a_full, b_full = _symmetric_to_full(half_a, half_b)
    
    # Normalize to sum to 1
    sum_a = np.sum(a_full)
    sum_b = np.sum(b_full)
    if abs(sum_a) > 1e-12:
        a_full = a_full / sum_a
    else:
        a_full = np.ones(12) / 12
    if abs(sum_b) > 1e-12:
        b_full = b_full / sum_b
    else:
        b_full = np.ones(12) / 12
    
    total_error = 0.0
    dim = exacts[0].shape[0]
    
    V_A, lam_A = eig_data_A
    V_B, lam_B = eig_data_B
    
    V_A_h = V_A.conj().T
    V_B_h = V_B.conj().T
    
    for i, t in enumerate(t_values):
        U = np.eye(dim, dtype=complex)
        
        for k in range(12):
            exp_B = V_B @ np.diag(np.exp(-1j * b_full[k] * t * lam_B)) @ V_B_h
            exp_A = V_A @ np.diag(np.exp(-1j * a_full[k] * t * lam_A)) @ V_A_h
            U = exp_B @ exp_A @ U
        
        diff = U - exacts[i]
        svals = np.linalg.svd(diff, compute_uv=False)
        total_error += svals[0]
        
    return total_error / len(t_values)


def _nelder_mead(func, x0, args=(), maxiter=1000, timeout=None, start_time=None):
    """
    Pure NumPy Nelder-Mead simplex optimization.
    """
    n = len(x0)
    
    # Initialize simplex
    simplex = [np.array(x0, dtype=float)]
    for i in range(n):
        x_i = x0.copy()
        x_i[i] += 0.1 if x0[i] == 0 else 0.1 * abs(x0[i])
        simplex.append(x_i)
    
    # Evaluate initial simplex
    fvals = [func(s, *args) for s in simplex]
    
    # Sort by function value
    for i in range(len(simplex)):
        min_idx = i + np.argmin(fvals[i:])
        simplex[i], simplex[min_idx] = simplex[min_idx], simplex[i]
        fvals[i], fvals[min_idx] = fvals[min_idx], fvals[i]
    
    best_x = simplex[0].copy()
    best_f = fvals[0]
    
    alpha = 1.0   # reflection
    gamma = 2.0   # expansion
    rho = 0.5     # contraction
    sigma = 0.5   # shrink
    
    for iteration in range(maxiter):
        if timeout is not None and time.time() - start_time > timeout:
            break
        
        # Current best
        best_x = simplex[0].copy()
        best_f = fvals[0]
        
        # Centroid of all points except worst
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        x_ref = centroid + alpha * (centroid - simplex[-1])
        f_ref = func(x_ref, *args)
        
        if f_ref < fvals[0]:
            # Try expansion
            x_exp = centroid + gamma * (x_ref - centroid)
            f_exp = func(x_exp, *args)
            if f_exp < f_ref:
                simplex[-1] = x_exp
                fvals[-1] = f_exp
            else:
                simplex[-1] = x_ref
                fvals[-1] = f_ref
        elif f_ref < fvals[-2]:
            simplex[-1] = x_ref
            fvals[-1] = f_ref
        else:
            # Contraction
            x_con = centroid + rho * (simplex[-1] - centroid)
            f_con = func(x_con, *args)
            if f_con < fvals[-1]:
                simplex[-1] = x_con
                fvals[-1] = f_con
            else:
                # Shrink
                for i in range(1, len(simplex)):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    fvals[i] = func(simplex[i], *args)
        
        # Sort by function value
        for i in range(len(simplex)):
            min_idx = i + np.argmin(fvals[i:])
            simplex[i], simplex[min_idx] = simplex[min_idx], simplex[i]
            fvals[i], fvals[min_idx] = fvals[min_idx], fvals[i]
        
        # Check convergence
        if abs(fvals[0] - fvals[-1]) < 1e-10:
            break
    
    return simplex[0], fvals[0]


def coefficients(stages):
    start_time = time.time()
    
    # 1. Build Operators
    A, B = _build_operators()
    
    # 2. Pre-compute Eigendecompositions for A and B
    eigvals_A, eigvecs_A = np.linalg.eigh(A)
    eig_data_A = (eigvecs_A, eigvals_A)
    
    eigvals_B, eigvecs_B = np.linalg.eigh(B)
    eig_data_B = (eigvecs_B, eigvals_B)
    
    # 3. Compute Exact Propagators
    t_values = [0.25, 0.5, 1.0]
    exacts = _compute_exact_propagators(A, B, t_values)
    
    # 4. Initial Guess - symmetric parameters
    x0 = _get_suzuki_4th_order()
    
    # 5. Optimization with Nelder-Mead on symmetric parameters
    best_params = x0.copy()
    best_err = _eval_error_symmetric(x0, eig_data_A, eig_data_B, exacts, t_values)
    
    try:
        # Stage 1: Coarse optimization
        elapsed = time.time() - start_time
        time_remaining = _DEADLINE - elapsed - 20.0
        if time_remaining > 10.0:
            p1, v1 = _nelder_mead(
                _eval_error_symmetric, 
                x0, 
                args=(eig_data_A, eig_data_B, exacts, t_values),
                maxiter=1500,
                timeout=time_remaining,
                start_time=start_time
            )
            if v1 < best_err:
                best_err = v1
                best_params = p1
        
        # Stage 2: Fine optimization from best found
        elapsed = time.time() - start_time
        time_remaining = _DEADLINE - elapsed - 10.0
        if time_remaining > 5.0:
            p2, v2 = _nelder_mead(
                _eval_error_symmetric, 
                best_params, 
                args=(eig_data_A, eig_data_B, exacts, t_values),
                maxiter=2000,
                timeout=time_remaining,
                start_time=start_time
            )
            if v2 < best_err:
                best_err = v2
                best_params = p2
                
    except Exception:
        # If optimization fails, keep the initial guess
        pass
    
    # 6. Extract Final Coefficients from symmetric params
    half_a = best_params[:6]
    half_b = best_params[6:]
    a_full, b_full = _symmetric_to_full(half_a, half_b)
    
    # Ensure sums are exactly 1
    sum_a = np.sum(a_full)
    sum_b = np.sum(b_full)
    
    if abs(sum_a) > 1e-12:
        a_full = a_full / sum_a
    if abs(sum_b) > 1e-12:
        b_full = b_full / sum_b
        
    # Clip to bounds
    a_full = np.clip(a_full, -50, 50)
    b_full = np.clip(b_full, -50, 50)
    
    # Re-normalize after clipping
    sum_a = np.sum(a_full)
    sum_b = np.sum(b_full)
    if abs(sum_a) > 1e-12:
        a_full = a_full / sum_a
    if abs(sum_b) > 1e-12:
        b_full = b_full / sum_b
        
    return list(a_full), list(b_full)
