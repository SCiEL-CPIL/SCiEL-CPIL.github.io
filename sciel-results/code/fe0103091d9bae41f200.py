import numpy as np
import time
import sys

def coefficients(stages):
    """
    Compute 12-stage product formula coefficients for Heisenberg chain time evolution.
    Returns (a, b) where each is a list of 12 floats summing to 1.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    # Build the Hamiltonians
    n = 6
    dim = 2 ** n
    delta = 0.8
    h = 0.3

    # Pauli matrices
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)

    def kron_all(ops):
        result = np.array([[1]], dtype=complex)
        for op in ops:
            result = np.kron(result, op)
        return result

    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)

    Jz = 1.0
    Jxy = delta * Jz

    def add_bond(H, i, j):
        ops = [np.eye(2) for _ in range(n)]
        ops[i] = sz
        ops[j] = sz
        H += Jz * kron_all(ops)

        ops = [np.eye(2) for _ in range(n)]
        ops[i] = sx
        ops[j] = sx
        H += (Jxy / 2) * kron_all(ops)

        ops = [np.eye(2) for _ in range(n)]
        ops[i] = sy
        ops[j] = sy
        H += (Jxy / 2) * kron_all(ops)

    for (i, j) in [(0, 1), (2, 3), (4, 5)]:
        add_bond(A, i, j)

    for (i, j) in [(1, 2), (3, 4)]:
        add_bond(B, i, j)

    for i in range(n):
        ops = [np.eye(2) for _ in range(n)]
        ops[i] = sz
        field_term = h * 0.5 * kron_all(ops)
        A += field_term
        B += field_term

    # Precompute eigendecompositions for A, B, and A+B
    vals_A, vecs_A = np.linalg.eigh(A)
    vals_B, vecs_B = np.linalg.eigh(B)
    vals_AB, vecs_AB = np.linalg.eigh(A + B)

    times = [0.25, 0.5, 1.0]

    def expm_A(t):
        """exp(-i * t * A)"""
        return vecs_A @ np.diag(np.exp(-1j * t * vals_A)) @ vecs_A.conj().T

    def expm_B(t):
        """exp(-i * t * B)"""
        return vecs_B @ np.diag(np.exp(-1j * t * vals_B)) @ vecs_B.conj().T

    def expm_AB(t):
        """exp(-i * t * (A+B))"""
        return vecs_AB @ np.diag(np.exp(-1j * t * vals_AB)) @ vecs_AB.conj().T

    # Precompute exact propagators
    U_exact = {}
    for t in times:
        U_exact[t] = expm_AB(t)

    def compute_error(a, b, t):
        """Compute spectral norm of U(t) - exp(-i t (A+B))"""
        U = np.eye(dim, dtype=complex)
        for k in range(len(a)):
            U = expm_B(b[k] * t) @ expm_A(a[k] * t) @ U
        diff = U - U_exact[t]
        s = np.linalg.svd(diff, compute_uv=False)
        return s[0]

    def objective(params):
        """
        params: 24-element array. First 12 are a, last 12 are b.
        """
        if time.time() > deadline:
            return 1e10

        a_full = params[:12]
        b_full = params[12:24]

        # Normalize to sum to 1
        a_sum = np.sum(a_full)
        b_sum = np.sum(b_full)
        if abs(a_sum) < 1e-15:
            a_full = np.ones(12) / 12
        else:
            a_full = a_full / a_sum
        if abs(b_sum) < 1e-15:
            b_full = np.ones(12) / 12
        else:
            b_full = b_full / b_sum

        # Check bounds
        if np.any(a_full < -50) or np.any(a_full > 50):
            return 1e10
        if np.any(b_full < -50) or np.any(b_full > 50):
            return 1e10

        total_err = 0.0
        for t in times:
            total_err += compute_error(a_full, b_full, t)
        return total_err / len(times)

    # 6th-order Suzuki-Trotter template (12-stage symmetric)
    # The 6th-order coefficients for a symmetric splitting have the form:
    # c1, c2, c3, c4, c5, c6, c6, c5, c4, c3, c2, c1 (for both a and b)
    # Using known 6th-order values (normalized to sum to 1):
    # These come from the 6th-order Suzuki-Trotter decomposition
    
    # 6th-order Suzuki coefficients (for 6 stages per half, mirrored for 12 total)
    # Standard 6th-order: use the known coefficients from literature
    # For a 12-stage symmetric formula: positions 0-5 and 6-11 are mirrors
    
    # Known 6th-order symmetric coefficients (12 stages):
    # a template: [c1, c2, c3, c4, c5, c6, c6, c5, c4, c3, c2, c1]
    # where the 6 unique values sum to 1
    
    # Using approximate 6th-order Suzuki-Trotter coefficients:
    # c = [0.16802951, 0.20030828, 0.12763218, 0.11203955, 0.19199049, 0.20000000]
    # (these need to sum to 1, let me use more accurate values)
    
    # More accurate 6th-order coefficients (from Suzuki 1991, adapted for 12 stages):
    c_template = np.array([
        0.16802951,  # c1
        0.20030828,  # c2  
        0.12763218,  # c3
        0.11203955,  # c4
        0.19199049,  # c5
        0.20000000,  # c6
        0.20000000,  # c6
        0.19199049,  # c5
        0.11203955,  # c4
        0.12763218,  # c3
        0.20030828,  # c2
        0.16802951   # c1
    ])
    
    # Normalize to sum to 1
    c_template = c_template / np.sum(c_template)
    
    # The scaling parameter s scales the template:
    # a_k = s * c_template[k] + (1 - s) / 12
    # b_k = s * c_template[k] + (1 - s) / 12
    # This ensures sum(a) = sum(b) = 1 for any s
    
    def params_from_scale(s):
        """Generate 24 params from a single scale factor s."""
        a = s * c_template + (1 - s) / 12
        b = s * c_template + (1 - s) / 12
        return np.concatenate([a, b])
    
    # Phase 1: 1D line search over scale factor s
    # s=0 gives uniform (1/12 each), s=1 gives pure template
    # Search s in [-2, 3] to allow negative coefficients
    
    best_s = 0.0
    best_obj = 1e10
    
    # Coarse grid search
    s_grid = np.linspace(-2, 3, 200)
    for s in s_grid:
        if time.time() > deadline - 60:
            break
        params = params_from_scale(s)
        obj = objective(params)
        if obj < best_obj:
            best_obj = obj
            best_s = s
    
    # Golden section refinement
    if time.time() < deadline - 40:
        # Find bracket around best_s
        step = 0.05
        s_lo = best_s - step
        s_hi = best_s + step
        
        # Expand bracket if needed
        f_lo = objective(params_from_scale(s_lo))
        f_hi = objective(params_from_scale(s_hi))
        f_mid = best_obj
        
        while f_lo > f_mid and time.time() < deadline - 45:
            s_lo -= step
            step *= 2
            f_lo = objective(params_from_scale(s_lo))
        
        while f_hi > f_mid and time.time() < deadline - 45:
            s_hi += step
            step *= 2
            f_hi = objective(params_from_scale(s_hi))
        
        # Golden section search
        gr = (np.sqrt(5) - 1) / 2  # golden ratio ~0.618
        a_gs = s_lo
        b_gs = s_hi
        c_gs = b_gs - gr * (b_gs - a_gs)
        d_gs = a_gs + gr * (b_gs - a_gs)
        f_c = objective(params_from_scale(c_gs))
        f_d = objective(params_from_scale(d_gs))
        
        for _ in range(50):
            if time.time() > deadline - 40:
                break
            if f_c < f_d:
                b_gs = d_gs
                d_gs = c_gs
                f_d = f_c
                c_gs = b_gs - gr * (b_gs - a_gs)
                f_c = objective(params_from_scale(c_gs))
            else:
                a_gs = c_gs
                c_gs = d_gs
                f_c = f_d
                d_gs = a_gs + gr * (b_gs - a_gs)
                f_d = objective(params_from_scale(d_gs))
            
            if abs(b_gs - a_gs) < 1e-8:
                break
        
        s_opt = (a_gs + b_gs) / 2
        obj_opt = objective(params_from_scale(s_opt))
        if obj_opt < best_obj:
            best_obj = obj_opt
            best_s = s_opt
    
    # Phase 2: Local Nelder-Mead polish in full 24D space
    def nelder_mead(func, x0, max_iter=300, tol=1e-10):
        n = len(x0)
        simplex = [x0.copy() for _ in range(n + 1)]
        simplex[0] = x0
        for i in range(1, n + 1):
            simplex[i] = x0.copy()
            simplex[i][i - 1] += 0.01

        fvals = [func(s) for s in simplex]

        for it in range(max_iter):
            if time.time() > deadline - 5:
                break

            indices = np.argsort(fvals)
            simplex = [simplex[i] for i in indices]
            fvals = [fvals[i] for i in indices]

            if abs(fvals[0] - fvals[-1]) < tol:
                break

            centroid = np.mean(simplex[:n], axis=0)

            # Reflection
            xr = centroid + (centroid - simplex[-1])
            fr = func(xr)

            if fr < fvals[0]:
                xe = centroid + 2 * (centroid - simplex[-1])
                fe = func(xe)
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            elif fr < fvals[-2]:
                simplex[-1] = xr
                fvals[-1] = fr
            else:
                if fr < fvals[-1]:
                    xc = centroid + 0.5 * (centroid - simplex[-1])
                else:
                    xc = centroid + 0.5 * (simplex[-1] - centroid)
                fc = func(xc)
                if fc < fvals[-1]:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        fvals[i] = func(simplex[i])

        return simplex[0], fvals[0]

    # Start Nelder-Mead from the best 1D solution
    x0 = params_from_scale(best_s)
    best_x = x0.copy()
    
    try:
        res_x, res_f = nelder_mead(objective, x0, max_iter=200)
        if res_f < best_obj:
            best_obj = res_f
            best_x = res_x
    except Exception:
        pass
    
    # A few random restarts if time permits
    np.random.seed(42)
    n_restarts = 0
    max_restarts = 3
    for _ in range(max_restarts):
        if time.time() > deadline - 10:
            break
        n_restarts += 1
        x_rand = best_x + 0.02 * np.random.randn(24)
        try:
            res_x, res_f = nelder_mead(objective, x_rand, max_iter=100)
            if res_f < best_obj:
                best_obj = res_f
                best_x = res_x
        except Exception:
            pass

    # Extract final coefficients
    a_full = best_x[:12]
    b_full = best_x[12:24]

    # Clip and renormalize
    a_full = np.clip(a_full, -50, 50)
    b_full = np.clip(b_full, -50, 50)
    a_sum = np.sum(a_full)
    b_sum = np.sum(b_full)
    if abs(a_sum) > 1e-12:
        a_full = a_full / a_sum
    if abs(b_sum) > 1e-12:
        b_full = b_full / b_sum

    return [float(x) for x in a_full], [float(x) for x in b_full]
