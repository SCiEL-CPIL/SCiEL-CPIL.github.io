"""
High-order product formula coefficients for Heisenberg chain time evolution.
Combines structured Suzuki-Trotter initializations with Nelder-Mead optimization
and multi-restart refinement.
"""
import numpy as np
import time

def coefficients(stages):
    """
    Return (a, b) where each is a list of 12 floats summing to 1.
    Uses Nelder-Mead optimization starting from structured high-order templates.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    # Build the Hamiltonians
    n = 6
    dim = 2 ** n
    delta = 0.8
    h = 0.3

    # Pauli matrices
    sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    eye = np.eye(2, dtype=complex)

    def kron_all(ops):
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result

    def bond_interaction(i, j):
        ops = [eye] * n
        ops[i] = sx
        ops[j] = sx
        term1 = kron_all(ops)
        
        ops = [eye] * n
        ops[i] = sy
        ops[j] = sy
        term2 = kron_all(ops)
        
        ops = [eye] * n
        ops[i] = sz
        ops[j] = sz
        term3 = kron_all(ops)
        
        return term1 + term2 + delta * term3

    A = np.zeros((dim, dim), dtype=complex)
    for i, j in [(0,1), (2,3), (4,5)]:
        A += bond_interaction(i, j)
    for i in range(n):
        ops = [eye] * n
        ops[i] = sz
        A += (h/2) * kron_all(ops)

    B = np.zeros((dim, dim), dtype=complex)
    for i, j in [(1,2), (3,4)]:
        B += bond_interaction(i, j)
    for i in range(n):
        ops = [eye] * n
        ops[i] = sz
        B += (h/2) * kron_all(ops)

    # Precompute eigendecompositions
    vals_A, vecs_A = np.linalg.eigh(A)
    vals_B, vecs_B = np.linalg.eigh(B)
    H = A + B
    vals_H, vecs_H = np.linalg.eigh(H)

    times = [0.25, 0.5, 1.0]

    # Precompute exact propagators
    U_exact = {}
    for t in times:
        U_exact[t] = vecs_H @ np.diag(np.exp(-1j * t * vals_H)) @ vecs_H.conj().T

    identity = np.eye(dim, dtype=complex)

    # Cache for exponential matrices
    expA_cache = {}
    expB_cache = {}

    def expm_A(t_val):
        key = round(t_val, 12)
        if key not in expA_cache:
            expA_cache[key] = vecs_A @ np.diag(np.exp(-1j * t_val * vals_A)) @ vecs_A.conj().T
        return expA_cache[key]

    def expm_B(t_val):
        key = round(t_val, 12)
        if key not in expB_cache:
            expB_cache[key] = vecs_B @ np.diag(np.exp(-1j * t_val * vals_B)) @ vecs_B.conj().T
        return expB_cache[key]

    def compute_error(a, b):
        """Compute mean spectral-norm error."""
        total_err = 0.0
        for t in times:
            U = identity.copy()
            for k in range(12):
                bk_t = b[k] * t
                if abs(bk_t) > 1e-15:
                    expB = expm_B(bk_t)
                else:
                    expB = identity
                U = expB @ U
                
                ak_t = a[k] * t
                if abs(ak_t) > 1e-15:
                    expA = expm_A(ak_t)
                else:
                    expA = identity
                U = expA @ U
            
            diff = U - U_exact[t]
            # Use SVD for spectral norm (more robust than eigvalsh for non-Hermitian)
            s = np.linalg.svd(diff, compute_uv=False)
            total_err += s[0]
        return total_err / len(times)

    def normalize(a, b):
        """Normalize a and b to sum to 1, clip to bounds."""
        a = np.clip(a, -50, 50)
        b = np.clip(b, -50, 50)
        a_sum = np.sum(a)
        b_sum = np.sum(b)
        if abs(a_sum) > 1e-12:
            a = a / a_sum
        else:
            a = np.ones(12) / 12
        if abs(b_sum) > 1e-12:
            b = b / b_sum
        else:
            b = np.ones(12) / 12
        return a, b

    def objective(params):
        """
        params: 24-element array. First 12 are a, last 12 are b.
        """
        if time.time() > deadline:
            return 1e10

        a_full = params[:12].copy()
        b_full = params[12:24].copy()

        a_full, b_full = normalize(a_full, b_full)

        if np.any(a_full < -50) or np.any(a_full > 50):
            return 1e10
        if np.any(b_full < -50) or np.any(b_full > 50):
            return 1e10

        return compute_error(a_full, b_full)

    # Nelder-Mead implementation
    def nelder_mead(func, x0, max_iter=200, tol=1e-10):
        n_params = len(x0)
        simplex = [x0.copy() for _ in range(n_params + 1)]
        simplex[0] = x0
        for i in range(1, n_params + 1):
            simplex[i] = x0.copy()
            simplex[i][i - 1] += 0.02

        fvals = [func(s) for s in simplex]

        for it in range(max_iter):
            if time.time() > deadline - 5:
                break

            indices = np.argsort(fvals)
            simplex = [simplex[i] for i in indices]
            fvals = [fvals[i] for i in indices]

            if abs(fvals[0] - fvals[-1]) < tol:
                break

            centroid = np.mean(simplex[:n_params], axis=0)

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
                    for i in range(1, n_params + 1):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        fvals[i] = func(simplex[i])

        return simplex[0], fvals[0]

    # Generate structured initial templates
    def get_suzuki_template():
        """4th-order Suzuki-Trotter coefficients for 12 stages."""
        x = 1.0 / (2.0 - 2.0**(1.0/3.0))
        y = (x - 1.0) / 2.0
        
        # Pattern for 12 stages: B A B A B A B A B A B A
        # Using symmetric 4th-order structure
        b_template = np.array([y, y, x, 1.0-2*x, x, y, y, x, 1.0-2*x, x, y, y])
        a_template = np.array([x, 1.0-2*x, x, y, y, x, x, 1.0-2*x, x, y, y, x])
        
        b_template = b_template / np.sum(b_template)
        a_template = a_template / np.sum(a_template)
        
        return a_template, b_template

    def get_uniform_template():
        a = np.ones(12) / 12
        b = np.ones(12) / 12
        return a, b

    def get_middle_emphasized():
        weights = np.array([1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 1], dtype=float)
        a = weights / np.sum(weights)
        b = weights / np.sum(weights)
        return a, b

    # Collect initial guesses
    np.random.seed(42)
    initial_guesses = []
    
    a_s, b_s = get_suzuki_template()
    initial_guesses.append(np.concatenate([a_s, b_s]))
    
    a_u, b_u = get_uniform_template()
    initial_guesses.append(np.concatenate([a_u, b_u]))
    
    a_m, b_m = get_middle_emphasized()
    initial_guesses.append(np.concatenate([a_m, b_m]))
    
    # Add some random perturbations of the Suzuki template
    for _ in range(3):
        pert = 0.01 * np.random.randn(24)
        initial_guesses.append(np.concatenate([a_s, b_s]) + pert)

    best_x = None
    best_obj = float('inf')

    # Phase 1: Evaluate all initial guesses
    for x0 in initial_guesses:
        if time.time() > deadline - 10:
            break
        try:
            f = objective(x0)
            if f < best_obj:
                best_obj = f
                best_x = x0.copy()
        except Exception:
            pass

    # Phase 2: Nelder-Mead from best initial guess
    if best_x is not None and time.time() < deadline - 20:
        try:
            res_x, res_f = nelder_mead(objective, best_x, max_iter=300)
            if res_f < best_obj:
                best_obj = res_f
                best_x = res_x
        except Exception:
            pass

    # Phase 3: Random restarts with varied perturbation scales
    max_restarts = 8
    for r in range(max_restarts):
        if time.time() > deadline - 15:
            break
        
        scale = 0.005 + 0.01 * (r / max_restarts)
        x_rand = best_x + scale * np.random.randn(24)
        
        try:
            res_x, res_f = nelder_mead(objective, x_rand, max_iter=150)
            if res_f < best_obj:
                best_obj = res_f
                best_x = res_x
        except Exception:
            pass

    # Phase 4: Fine-tune with smaller steps around best
    if time.time() < deadline - 10:
        for r in range(5):
            if time.time() > deadline - 5:
                break
            x_rand = best_x + 0.002 * np.random.randn(24)
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
    a_full, b_full = normalize(a_full, b_full)

    return [float(x) for x in a_full], [float(x) for x in b_full]
