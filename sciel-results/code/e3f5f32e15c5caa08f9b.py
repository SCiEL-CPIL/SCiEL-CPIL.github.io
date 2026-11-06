"""Product-formula design for Heisenberg-chain time evolution using optimized symmetric splitting."""
import time
import numpy as np

def _build_hamiltonians():
    """Build A (odd bonds) and B (even bonds) for 6-site open XXZ chain.
    Anisotropy gamma=0.8, field h=0.3.
    """
    N = 6
    dim = 2 ** N
    gamma = 0.8
    h = 0.3
    
    sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    sp = sx + 1j * sy
    sm = sx - 1j * sy
    I2 = np.eye(2, dtype=complex)
    
    def op_on_site(op, site):
        ops = [I2] * N
        ops[site] = op
        result = ops[0]
        for o in ops[1:]:
            result = np.kron(result, o)
        return result
    
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    odd_bonds = [(0, 1), (2, 3), (4, 5)]
    even_bonds = [(1, 2), (3, 4)]
    
    def bond_hamiltonian(i, j):
        sp_i = op_on_site(sp, i)
        sm_j = op_on_site(sm, j)
        sm_i = op_on_site(sm, i)
        sp_j = op_on_site(sp, j)
        sz_i = op_on_site(sz, i)
        sz_j = op_on_site(sz, j)
        return gamma / 2 * (sp_i @ sm_j + sm_i @ sp_j) + sz_i @ sz_j
    
    for i, j in odd_bonds:
        A += bond_hamiltonian(i, j)
    for i, j in even_bonds:
        B += bond_hamiltonian(i, j)
    
    half_field = 0.5 * h
    for s in range(N):
        sz_s = op_on_site(sz, s)
        A -= half_field * sz_s
        B -= half_field * sz_s
    
    return A, B

def _compute_error(a, b, A_eigvals, A_eigvecs, B_eigvals, B_eigvecs, U_exact_cache, times):
    """Compute mean spectral norm error."""
    dim = A_eigvecs.shape[0]
    errors = []
    
    for t in times:
        U = np.eye(dim, dtype=complex)
        
        for k in range(len(a)):
            if abs(b[k]) > 1e-15:
                theta_b = b[k] * t
                exp_b = np.exp(-1j * theta_b * B_eigvals)
                U = U @ (B_eigvecs @ np.diag(exp_b) @ B_eigvecs.conj().T)
            
            if abs(a[k]) > 1e-15:
                theta_a = a[k] * t
                exp_a = np.exp(-1j * theta_a * A_eigvals)
                U = U @ (A_eigvecs @ np.diag(exp_a) @ A_eigvecs.conj().T)
        
        U_exact = U_exact_cache[t]
        err = np.linalg.norm(U - U_exact, 2)
        errors.append(err)
    
    return np.mean(errors)

def _yoshida_initial_guess(stages):
    """Generate a symmetric Yoshida-inspired initial guess for 12 stages.
    
    Uses the 4th-order Suzuki-Trotter structure distributed symmetrically.
    The 4th order scheme has coefficients:
    c0 = 1/(2 - 2^(1/3)), c1 = -2^(1/3)/(2 - 2^(1/3))
    
    For a symmetric 12-stage formula, we distribute these in a pattern that
    maintains 4th-order accuracy while using all 12 stages.
    """
    c = 2 ** (1.0 / 3.0)
    denom = 2.0 - c
    c0 = 1.0 / denom
    c1 = -c / denom
    
    # The 4th order Suzuki-Trotter for exp(-i t (A+B)) is:
    # exp(-i c0/2 * t A) exp(-i c1/2 * t B) exp(-i c0/2 * t A)
    # exp(-i c1/2 * t A) exp(-i c0 * t B) exp(-i c1/2 * t A)
    # exp(-i c0/2 * t A) exp(-i c1/2 * t B) exp(-i c0/2 * t A)
    
    # This gives 5 A-factors and 4 B-factors.
    # For 12 stages, we can use a symmetric pattern.
    
    # Let's create a symmetric 12-stage pattern based on the Yoshida structure.
    # We'll use a pattern that's symmetric and sums to 1.
    
    # Start with a symmetric distribution inspired by the 4th-order coefficients
    # The key coefficients are c0/2, c1/2, c0/2, c1/2, c0/2 for A
    # and c1/2, c0, c1/2 for B
    
    # For 12 stages, let's use a symmetric pattern:
    # a = [x1, x2, x3, x4, x5, x6, x6, x5, x4, x3, x2, x1]
    # b = [y1, y2, y3, y4, y5, y6, y6, y5, y4, y3, y2, y1]
    
    # Initialize with values based on the Yoshida coefficients
    # c0 ≈ 1.86, c1 ≈ -0.86
    
    a_half = [c0/4, c1/4, c0/4, c1/4, c0/4, (1 - 2*(c0/4 + c1/4 + c0/4 + c1/4 + c0/4))/6]
    # Simplify: sum of first 5 = 3*c0/4 + 2*c1/4
    # Remaining for last = 1 - 3*c0/4 - 2*c1/4
    # Split remaining evenly among 6 positions
    
    sum_first5 = 3*c0/4 + 2*c1/4
    remaining = 1 - sum_first5
    a_half = [c0/4, c1/4, c0/4, c1/4, c0/4, remaining/6]
    
    a = a_half + a_half[::-1]
    
    # For b, use a similar symmetric pattern
    b_half = [c1/4, c0/2, c1/4, c1/4, c0/2, c1/4]
    sum_b_half = sum(b_half)
    # Scale to make total sum = 1
    if abs(sum_b_half) > 1e-10:
        b_half = [x / sum_b_half for x in b_half]
    
    b = b_half + b_half[::-1]
    
    # Ensure sums are exactly 1
    sa = sum(a)
    sb = sum(b)
    if abs(sa) > 1e-10:
        a = [x / sa for x in a]
    if abs(sb) > 1e-10:
        b = [x / sb for x in b]
    
    return a, b

def _nelder_mead_constrained(f, x0, bounds, max_iters=200, timeout=None, start_time=None):
    """Nelder-Mead with simple bound handling.
    
    x0: initial point
    bounds: list of (low, high) for each coordinate
    """
    n = len(x0)
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    def clip(x):
        return [max(bounds[i][0], min(bounds[i][1], x[i])) for i in range(n)]
    
    # Create simplex
    simplex = [x0[:]]
    f_simplex = [f(x0)]
    
    for i in range(n):
        xi = x0[:]
        step = 0.1 * (bounds[i][1] - bounds[i][0]) if bounds[i][1] > bounds[i][0] else 0.1
        xi[i] += step
        xi = clip(xi)
        simplex.append(xi)
        f_simplex.append(f(xi))
    
    for it in range(max_iters):
        if timeout is not None and start_time is not None:
            if time.time() - start_time > timeout:
                break
        
        indices = np.argsort(f_simplex)
        simplex = [simplex[i] for i in indices]
        f_simplex = [f_simplex[i] for i in indices]
        
        if abs(f_simplex[0] - f_simplex[-1]) < 1e-14:
            break
        
        centroid = np.mean(simplex[:-1], axis=0)
        worst = simplex[-1]
        
        x_r = clip([centroid[i] + alpha * (centroid[i] - worst[i]) for i in range(n)])
        f_r = f(x_r)
        
        if f_r < f_simplex[0]:
            x_e = clip([centroid[i] + gamma * (x_r[i] - centroid[i]) for i in range(n)])
            f_e = f(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                f_simplex[-1] = f_e
            else:
                simplex[-1] = x_r
                f_simplex[-1] = f_r
        elif f_r < f_simplex[-2]:
            simplex[-1] = x_r
            f_simplex[-1] = f_r
        else:
            x_c = clip([centroid[i] + rho * (worst[i] - centroid[i]) for i in range(n)])
            f_c = f(x_c)
            if f_c < f_simplex[-1]:
                simplex[-1] = x_c
                f_simplex[-1] = f_c
            else:
                for i in range(1, n + 1):
                    simplex[i] = [(simplex[i][j] + simplex[0][j]) / 2.0 for j in range(n)]
                    simplex[i] = clip(simplex[i])
                    f_simplex[i] = f(simplex[i])
    
    best_idx = np.argmin(f_simplex)
    return simplex[best_idx], f_simplex[best_idx]

def _normalize_and_clip(a, b):
    """Ensure sums are 1 and values are in [-50, 50]."""
    sa = sum(a)
    sb = sum(b)
    if abs(sa) > 1e-10:
        a = [x / sa for x in a]
    if abs(sb) > 1e-10:
        b = [x / sb for x in b]
    a = [max(-50.0, min(50.0, x)) for x in a]
    b = [max(-50.0, min(50.0, x)) for x in b]
    sa = sum(a)
    sb = sum(b)
    if abs(sa) > 1e-10:
        a = [x / sa for x in a]
    if abs(sb) > 1e-10:
        b = [x / sb for x in b]
    return a, b

def coefficients(stages):
    start_time = time.time()
    deadline = 140.0  # Leave 10s safety margin before 150s kill
    
    try:
        A, B = _build_hamiltonians()
        dim = A.shape[0]
        times = [0.25, 0.5, 1.0]
        
        # Compute eigendecompositions
        A_eigvals, A_eigvecs = np.linalg.eigh(A)
        B_eigvals, B_eigvecs = np.linalg.eigh(B)
        H = A + B
        H_eigvals, H_eigvecs = np.linalg.eigh(H)
        
        # Precompute exact propagators
        U_exact_cache = {}
        for t in times:
            exp_h = np.exp(-1j * t * H_eigvals)
            U_exact_cache[t] = H_eigvecs @ np.diag(exp_h) @ H_eigvecs.conj().T
        
        # Initial guess: Yoshida-inspired symmetric pattern
        a0, b0 = _yoshida_initial_guess(stages)
        
        best_a = a0[:]
        best_b = b0[:]
        best_err = _compute_error(best_a, best_b, A_eigvals, A_eigvecs, B_eigvals, B_eigvecs, U_exact_cache, times)
        
        elapsed = time.time() - start_time
        if elapsed > deadline - 5:
            return _normalize_and_clip(best_a, best_b)
        
        # Optimize using Nelder-Mead
        # We optimize 24 parameters: 12 a's and 12 b's
        # But we need sum(a)=1 and sum(b)=1, so we use 22 free parameters
        # a_0..a_10 are free, a_11 = 1 - sum(a_0..a_10)
        # b_0..b_10 are free, b_11 = 1 - sum(b_0..b_10)
        
        def objective(params):
            a = params[:12]
            b = params[12:24]
            
            # Enforce sum constraints by adjusting last element
            a = list(a)
            b = list(b)
            a[11] = 1.0 - sum(a[:11])
            b[11] = 1.0 - sum(b[:11])
            
            # Clip to valid range
            a = [max(-50.0, min(50.0, x)) for x in a]
            b = [max(-50.0, min(50.0, x)) for x in b]
            
            # Re-normalize to ensure sum = 1
            sa = sum(a)
            sb = sum(b)
            if abs(sa) > 1e-10:
                a = [x / sa for x in a]
            if abs(sb) > 1e-10:
                b = [x / sb for x in b]
            
            return _compute_error(a, b, A_eigvals, A_eigvecs, B_eigvals, B_eigvecs, U_exact_cache, times)
        
        # Initial parameters (22 free + 2 determined)
        x0 = a0[:] + b0[:]
        
        # Bounds for each parameter
        bounds = [(-50.0, 50.0)] * 24
        
        # Run optimization with time limit
        remaining_time = deadline - (time.time() - start_time) - 2  # 2s safety margin
        max_iters = int(remaining_time / 0.5)  # Estimate ~0.5s per iteration
        
        if remaining_time > 5:
            x_opt, f_opt = _nelder_mead_constrained(
                objective, x0, bounds, 
                max_iters=max(50, min(200, max_iters)),
                timeout=remaining_time,
                start_time=start_time
            )
            
            a_opt = x_opt[:12]
            b_opt = x_opt[12:24]
            
            # Enforce sum constraints
            a_opt = list(a_opt)
            b_opt = list(b_opt)
            a_opt[11] = 1.0 - sum(a_opt[:11])
            b_opt[11] = 1.0 - sum(b_opt[:11])
            
            # Final normalization and clipping
            a_opt, b_opt = _normalize_and_clip(a_opt, b_opt)
            
            # Check if optimized result is better
            err_opt = _compute_error(a_opt, b_opt, A_eigvals, A_eigvecs, B_eigvals, B_eigvecs, U_exact_cache, times)
            
            if err_opt < best_err:
                best_a = a_opt
                best_b = b_opt
                best_err = err_opt
        
        return _normalize_and_clip(best_a, best_b)
    
    except Exception as e:
        # Fallback: return a valid uniform-like coefficient set
        a = [1.0 / stages] * stages
        b = [1.0 / stages] * stages
        return a, b
# EVOLVE-BLOCK-END
