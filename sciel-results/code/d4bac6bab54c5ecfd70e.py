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


def _eval_error(params, A, B, eig_A, eig_B, exacts, t_values):
    """
    params: array of 12 floats representing coefficients directly.
    a = params[:6] (first 6 of the 12, symmetric: a_full = [a0..a5, a5..a0])
    b = params[6:12] (first 6 of the 12, symmetric: b_full = [b0..b5, b5..b0])
    We normalize so that sum(a_full) = 1 and sum(b_full) = 1.
    """
    a_half = params[:6]
    b_half = params[6:12]

    # Construct symmetric full lists
    a_full_raw = np.concatenate([a_half, a_half[::-1]])
    b_full_raw = np.concatenate([b_half, b_half[::-1]])

    # Normalize to ensure sum is exactly 1
    sum_a = np.sum(a_full_raw)
    sum_b = np.sum(b_full_raw)

    if np.abs(sum_a) < 1e-12:
        sum_a = 1e-12
    if np.abs(sum_b) < 1e-12:
        sum_b = 1e-12

    a_full = a_full_raw / sum_a
    b_full = b_full_raw / sum_b

    # Clip to valid range [-50, 50]
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

    V_A, lam_A = eig_A
    V_B, lam_B = eig_B

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


def _nelder_mead(func, x0, args=(), max_iter=500, timeout=None, start_time=None):
    """
    Pure NumPy implementation of Nelder-Mead simplex optimization.
    """
    if start_time is None:
        start_time = time.time()

    n = len(x0)
    # Initial simplex
    simplex = [np.array(x0, dtype=float)]
    for i in range(n):
        point = np.array(x0, dtype=float)
        step = 0.1 if abs(x0[i]) < 1e-10 else 0.1 * abs(x0[i])
        point[i] += step
        simplex.append(point)

    values = [func(p, *args) for p in simplex]
    best_val = min(values)
    best_point = simplex[values.index(best_val)].copy()

    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5

    for iteration in range(max_iter):
        if timeout is not None and time.time() - start_time > timeout:
            break

        # Sort simplex by function value
        order = np.argsort(values)
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]

        # Centroid of all but worst point
        centroid = np.mean(simplex[:-1], axis=0)

        # Reflection
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = func(xr, *args)

        if fr < values[0]:
            # Expansion
            xe = centroid + gamma * (centroid - simplex[-1])
            fe = func(xe, *args)
            if fe < fr:
                simplex[-1] = xe
                values[-1] = fe
            else:
                simplex[-1] = xr
                values[-1] = fr
        elif fr < values[-2]:
            simplex[-1] = xr
            values[-1] = fr
        else:
            # Contraction
            if fr < values[-1]:
                xc = centroid + 0.5 * (simplex[-1] - centroid)
            else:
                xc = centroid - 0.5 * (simplex[-1] - centroid)
            fc = func(xc, *args)
            if fc < values[-1]:
                simplex[-1] = xc
                values[-1] = fc
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    values[i] = func(simplex[i], *args)

        # Track best
        min_val = min(values)
        if min_val < best_val:
            best_val = min_val
            best_point = simplex[values.index(min_val)].copy()

    return best_point, best_val


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

    # 4. Initial Guess
    # Use a symmetric 4th-order-like initialization
    # a_half = [0.1, 0.2, 0.2, 0.2, 0.2, 0.1] sum=1
    # b_half = [0.1, 0.2, 0.2, 0.2, 0.2, 0.1] sum=1
    init_a = np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1])
    init_b = np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1])
    x0 = np.concatenate([init_a, init_b])

    # 5. Multi-stage optimization
    best_params = x0.copy()
    best_err = _eval_error(x0, A, B, eig_data_A, eig_data_B, exacts, t_values)

    # Multiple restarts with different initial points
    np.random.seed(42)

    for restart in range(5):
        elapsed = time.time() - start_time
        time_remaining = _DEADLINE - elapsed - 10.0
        if time_remaining < 5.0:
            break

        if restart == 0:
            x_start = x0
        else:
            # Perturb the best found so far
            x_start = best_params + np.random.normal(0, 0.05, size=12)
            x_start = np.clip(x_start, -1.0, 1.0)

        try:
            p, v = _nelder_mead(
                _eval_error,
                x_start,
                args=(A, B, eig_data_A, eig_data_B, exacts, t_values),
                max_iter=300,
                timeout=time_remaining,
                start_time=start_time
            )
            if v < best_err:
                best_err = v
                best_params = p
        except Exception:
            pass

    # 6. Extract Final Coefficients
    a_half = best_params[:6]
    b_half = best_params[6:12]

    a_full = np.concatenate([a_half, a_half[::-1]])
    b_full = np.concatenate([b_half, b_half[::-1]])

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
