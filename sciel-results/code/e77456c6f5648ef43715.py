"""Optimized product formula for Heisenberg chain time evolution."""
import time
import numpy as np

def _build_hamiltonians():
    """Build A (odd bonds + half field) and B (even bonds + half field) for 6-site XXZ chain."""
    N = 6
    anis = 0.8
    field = 0.3
    dim = 2 ** N
    
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    
    def kron_all(ops):
        result = np.array([[1.0]], dtype=complex)
        for op in ops:
            result = np.kron(result, op)
        return result
    
    def op_on_site(site, op):
        ops = [I2] * N
        ops[site] = op
        return kron_all(ops)
    
    def bond_op(j, op1, op2):
        ops = [I2] * N
        ops[j] = op1
        ops[j+1] = op2
        return kron_all(ops)
    
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    # Odd bonds: (0,1), (2,3), (4,5) -> A
    for j in [0, 2, 4]:
        A += bond_op(j, sx, sx) + bond_op(j, sy, sy) + anis * bond_op(j, sz, sz)
    
    # Even bonds: (1,2), (3,4) -> B
    for j in [1, 3]:
        B += bond_op(j, sx, sx) + bond_op(j, sy, sy) + anis * bond_op(j, sz, sz)
    
    # Uniform field: half in A, half in B
    for j in range(N):
        A += 0.5 * field * op_on_site(j, sz)
        B += 0.5 * field * op_on_site(j, sz)
    
    return A, B


def _expm_via_eig(M, scale):
    """Compute exp(scale * M) using eigendecomposition."""
    try:
        evals, evecs = np.linalg.eig(M)
        return evecs @ np.diag(np.exp(scale * evals)) @ np.linalg.inv(evecs)
    except np.linalg.LinAlgError:
        return None


def _objective(a, b, A, B, H_eig_cache):
    """Compute mean spectral norm error over t = 0.25, 0.5, 1.0."""
    stages = len(a)
    if abs(np.sum(a) - 1.0) > 1e-8 or abs(np.sum(b) - 1.0) > 1e-8:
        return 1e10
    if np.any(a < -50) or np.any(a > 50) or np.any(b < -50) or np.any(b > 50):
        return 1e10
    
    H = A + B
    t_vals = [0.25, 0.5, 1.0]
    errors = []
    
    for t in t_vals:
        # Exact propagator
        if H_eig_cache is not None:
            evals, evecs = H_eig_cache
            exact = evecs @ np.diag(np.exp(-1j * t * evals)) @ np.linalg.inv(evecs)
        else:
            exact = _expm_via_eig(H, -1j * t)
            if exact is None:
                return 1e10
        
        # Product formula: U = prod_k exp(-i b_k t B) exp(-i a_k t A)
        dim = A.shape[0]
        U = np.eye(dim, dtype=complex)
        
        for k in range(stages):
            exp_a = _expm_via_eig(A, -1j * a[k] * t)
            exp_b = _expm_via_eig(B, -1j * b[k] * t)
            if exp_a is None or exp_b is None:
                return 1e10
            U = exp_b @ exp_a @ U
        
        diff = U - exact
        s = np.linalg.svd(diff, compute_uv=False)
        errors.append(s[0] if len(s) > 0 else 0.0)
    
    return np.mean(errors)


def coefficients(stages):
    """
    12-stage product formula using symmetric 4th-order-inspired initialization
    refined by Nelder-Mead optimization.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    # Build Hamiltonians once
    A, B = _build_hamiltonians()
    H = A + B
    
    # Precompute eigendecomposition of H
    try:
        H_eig_cache = np.linalg.eig(H)
    except np.linalg.LinAlgError:
        H_eig_cache = None
    
    # Initialize with symmetric pattern inspired by 4th-order Yoshida
    # c = 2^(1/3)
    c = 2.0 ** (1.0 / 3.0)
    x1 = 1.0 / (2.0 - c)
    x2 = -c / (2.0 - c)
    x3 = 1.0 - 2.0 * x1
    
    # Symmetric 12-stage pattern: mirror the first 6 stages
    # Use a pattern that approximates 4th-order behavior
    # a: [x3/2, x1/2, x1/2, x3/2, x3/2, x1/2, x1/2, x3/2, x3/2, x1/2, x1/2, x3/2]
    # But need sum(a) = 1, so scale appropriately
    
    # Start with a reasonable symmetric initialization
    a = np.array([x3/4, x1/4, x1/4, x3/4, x3/4, x1/4, x1/4, x3/4, x3/4, x1/4, x1/4, x3/4])
    b = np.array([x2/4, x2/4, 0, 0, x2/4, x2/4, 0, 0, x2/4, x2/4, 0, 0])
    
    # Normalize to ensure sum = 1
    if abs(np.sum(a)) > 1e-10:
        a = a / np.sum(a)
    if abs(np.sum(b)) > 1e-10:
        b = b / np.sum(b)
    else:
        b = np.ones(stages) / stages
    
    # Ensure within bounds
    a = np.clip(a, -50, 50)
    b = np.clip(b, -50, 50)
    
    # Re-normalize after clipping
    if abs(np.sum(a) - 1.0) > 1e-10:
        a = a + (1.0 - np.sum(a)) / stages
    if abs(np.sum(b) - 1.0) > 1e-10:
        b = b + (1.0 - np.sum(b)) / stages
    
    best_a = a.copy()
    best_b = b.copy()
    best_score = _objective(a, b, A, B, H_eig_cache)
    
    # Nelder-Mead optimization on the 24-dimensional parameter space
    # Use a reduced parameterization to maintain sum constraints
    # We'll optimize 22 free parameters and derive the last two to satisfy sum=1
    
    def constrained_objective(params):
        """params: 22 free values; a[11] and b[11] are derived."""
        a_free = params[:11]
        b_free = params[11:22]
        
        a_full = np.append(a_free, 1.0 - np.sum(a_free))
        b_full = np.append(b_free, 1.0 - np.sum(b_free))
        
        if np.any(a_full < -50) or np.any(a_full > 50) or np.any(b_full < -50) or np.any(b_full > 50):
            return 1e10
        
        return _objective(a_full, b_full, A, B, H_eig_cache)
    
    # Initial parameter vector
    x0 = np.concatenate([best_a[:11], best_b[:11]])
    
    # Nelder-Mead implementation
    n = len(x0)
    simplex = [x0.copy()]
    for i in range(n):
        v = x0.copy()
        v[i] += 0.1 * (abs(v[i]) + 1.0)
        simplex.append(v)
    
    fvals = []
    for v in simplex:
        if time.time() > deadline:
            break
        fvals.append(constrained_objective(v))
    
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    max_iter = 100
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        # Sort simplex by objective value
        order = np.argsort(fvals)
        simplex = [simplex[idx] for idx in order]
        fvals = [fvals[idx] for idx in order]
        
        # Check convergence
        if abs(fvals[0] - fvals[-1]) < 1e-8:
            break
        
        # Update best
        a_full = np.append(simplex[0][:11], 1.0 - np.sum(simplex[0][:11]))
        b_full = np.append(simplex[0][11:22], 1.0 - np.sum(simplex[0][11:22]))
        if fvals[0] < best_score:
            best_a = a_full.copy()
            best_b = b_full.copy()
            best_score = fvals[0]
        
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        x_r = centroid + alpha * (centroid - simplex[-1])
        f_r = constrained_objective(x_r)
        
        if f_r < fvals[0]:
            # Expansion
            x_e = centroid + gamma * (x_r - centroid)
            f_e = constrained_objective(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                fvals[-1] = f_e
            else:
                simplex[-1] = x_r
                fvals[-1] = f_r
        elif f_r < fvals[-2]:
            simplex[-1] = x_r
            fvals[-1] = f_r
        else:
            # Contraction
            x_c = centroid + rho * (simplex[-1] - centroid)
            f_c = constrained_objective(x_c)
            if f_c < fvals[-1]:
                simplex[-1] = x_c
                fvals[-1] = f_c
            else:
                # Shrink
                for j in range(1, n + 1):
                    simplex[j] = simplex[0] + sigma * (simplex[j] - simplex[0])
                    fvals[j] = constrained_objective(simplex[j])
    
    # Final check
    if time.time() < deadline:
        # Try a few more random restarts if time permits
        for _ in range(3):
            if time.time() > deadline - 20:
                break
            x_rand = x0 + np.random.normal(0, 0.3, n)
            f_rand = constrained_objective(x_rand)
            if f_rand < best_score:
                a_full = np.append(x_rand[:11], 1.0 - np.sum(x_rand[:11]))
                b_full = np.append(x_rand[11:22], 1.0 - np.sum(x_rand[11:22]))
                best_a = a_full.copy()
                best_b = b_full.copy()
                best_score = f_rand
    
    # Final normalization and bounds check
    best_a = np.clip(best_a, -50, 50)
    best_b = np.clip(best_b, -50, 50)
    
    # Ensure exact sum = 1
    best_a = best_a + (1.0 - np.sum(best_a)) / stages
    best_b = best_b + (1.0 - np.sum(best_b)) / stages
    
    # Final bounds check
    if np.any(best_a < -50) or np.any(best_a > 50) or np.any(best_b < -50) or np.any(best_b > 50):
        # Fallback to equal weights
        best_a = np.ones(stages) / stages
        best_b = np.ones(stages) / stages
    
    return list(best_a), list(best_b)
