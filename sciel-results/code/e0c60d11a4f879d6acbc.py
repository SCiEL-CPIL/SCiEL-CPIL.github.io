import time
import numpy as np

_DEADLINE = 135.0  # Safety margin before 150s kill

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

def _expand_symmetric(params):
    """
    Expand 12 parameters (6 for A, 6 for B) into 24 coefficients using symmetry.
    params = [a1..a6, b1..b6]
    a_full = [a1, a2, a3, a4, a5, a6, a6, a5, a4, a3, a2, a1]
    b_full = [b1, b2, b3, b4, b5, b6, b6, b5, b4, b3, b2, b1]
    """
    a_half = params[:6]
    b_half = params[6:12]
    
    a_full = np.concatenate([a_half, a_half[::-1]])
    b_full = np.concatenate([b_half, b_half[::-1]])
    
    return a_full, b_full

def _eval_error_sym(params, A, B, eig_data_A, eig_data_B, exacts, t_values):
    """
    params: array of 12 floats (6 for A, 6 for B).
    Enforces symmetry.
    """
    a_full, b_full = _expand_symmetric(params)
    
    # Normalize to ensure sum is exactly 1
    sum_a = np.sum(a_full)
    sum_b = np.sum(b_full)
    
    if np.abs(sum_a) < 1e-12:
        sum_a = 1.0
    if np.abs(sum_b) < 1e-12:
        sum_b = 1.0
        
    a_full = a_full / sum_a
    b_full = b_full / sum_b
    
    # Clip to valid range
    a_full = np.clip(a_full, -50, 50)
    b_full = np.clip(b_full, -50, 50)
    
    # Re-normalize after clipping if necessary
    sum_a_clip = np.sum(a_full)
    sum_b_clip = np.sum(b_full)
    if np.abs(sum_a_clip) > 1e-12:
        a_full = a_full / sum_a_clip
    if np.abs(sum_b_clip) > 1e-12:
        b_full = b_full / sum_b_clip
    
    total_error = 0.0
    dim = A.shape[0]
    
    V_A, lam_A = eig_data_A
    V_B, lam_B = eig_data_B
    
    V_A_h = V_A.conj().T
    V_B_h = V_B.conj().T
    
    for i, t in enumerate(t_values):
        U = np.eye(dim, dtype=complex)
        
        for k in range(12):
            # Pre-compute diagonal exponentials
            exp_B_diag = np.exp(-1j * b_full[k] * t * lam_B)
            exp_A_diag = np.exp(-1j * a_full[k] * t * lam_A)
            
            exp_B = (V_B * exp_B_diag) @ V_B_h
            exp_A = (V_A * exp_A_diag) @ V_A_h
            U = exp_B @ exp_A @ U
        
        diff = U - exacts[i]
        svals = np.linalg.svd(diff, compute_uv=False)
        total_error += svals[0]
        
    return total_error / len(t_values)

def _nelder_mead(func, x0, args=(), maxiter=1000, timeout=None, start_time=None):
    """
    Pure NumPy Nelder-Mead simplex optimization with adaptive step sizes.
    """
    n = len(x0)
    
    # Initialize simplex with adaptive step size
    simplex = [np.array(x0, dtype=float)]
    step = 0.1
    for i in range(n):
        x_i = x0.copy()
        x_i[i] += step * (1.0 if x0[i] == 0 else np.sign(x0[i]))
        simplex.append(x_i)
    
    # Evaluate initial simplex
    fvals = [func(s, *args) for s in simplex]
    
    # Sort by function value
    idx = np.argsort(fvals)
    simplex = [simplex[i] for i in idx]
    fvals = [fvals[i] for i in idx]
    
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
        idx = np.argsort(fvals)
        simplex = [simplex[i] for i in idx]
        fvals = [fvals[i] for i in idx]
        
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
    
    # 4. Initial Guess - Symmetric Suzuki-inspired coefficients
    # We optimize 12 parameters: [a1..a6, b1..b6]
    # Start with uniform 1/12 for the full 12, so 1/12 for each of the 6 unique ones?
    # No, sum(a_full)=1. If a_full is [a1..a6, a6..a1], sum = 2*sum(a1..a6) = 1 => sum(a1..a6) = 0.5.
    # Uniform a1..a6 = 0.5/6 = 1/12.
    # So x0 = [1/12]*12 is a valid start for the half-parameters.
    
    x0 = np.ones(12) / 12.0
    
    # 5. Optimization with multiple restarts
    best_params = x0.copy()
    best_err = _eval_error_sym(x0, A, B, eig_data_A, eig_data_B, exacts, t_values)
    
    try:
        # Multiple restarts with different initial conditions
        rng = np.random.RandomState(42)
        n_restarts = 8 # Can afford more restarts due to lower dimensionality
        
        for restart in range(n_restarts):
            elapsed = time.time() - start_time
            time_remaining = _DEADLINE - elapsed - 15.0
            if time_remaining < 10.0:
                break
            
            # Generate initial point: uniform + random perturbation
            if restart == 0:
                x_start = x0.copy()
            else:
                # Random perturbation
                scale = 0.05 * (restart + 1)
                x_start = x0 + rng.randn(12) * scale
                # Ensure sums are close to 1 (for the expanded version)
                # We don't strictly need to normalize x_start here as the eval function does it,
                # but keeping it close helps NM.
                a_half = x_start[:6]
                b_half = x_start[6:]
                sum_a = np.sum(a_half)
                sum_b = np.sum(b_half)
                if abs(sum_a) > 1e-6:
                    x_start[:6] = a_half / sum_a * 0.5
                if abs(sum_b) > 1e-6:
                    x_start[6:] = b_half / sum_b * 0.5
            
            p, v = _nelder_mead(
                _eval_error_sym, 
                x_start, 
                args=(A, B, eig_data_A, eig_data_B, exacts, t_values),
                maxiter=400, # More iterations allowed for smaller dim
                timeout=time_remaining,
                start_time=start_time
            )
            if v < best_err:
                best_err = v
                best_params = p.copy()
        
        # Fine-tune from best found
        elapsed = time.time() - start_time
        time_remaining = _DEADLINE - elapsed - 5.0
        if time_remaining > 5.0:
            p2, v2 = _nelder_mead(
                _eval_error_sym, 
                best_params, 
                args=(A, B, eig_data_A, eig_data_B, exacts, t_values),
                maxiter=400,
                timeout=time_remaining,
                start_time=start_time
            )
            if v2 < best_err:
                best_err = v2
                best_params = p2.copy()
                
    except Exception:
        # If optimization fails, keep the initial guess
        pass
    
    # 6. Extract Final Coefficients
    a_full, b_full = _expand_symmetric(best_params)
    
    # Ensure sums are exactly 1 and within bounds
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
