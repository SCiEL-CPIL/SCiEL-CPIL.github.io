import numpy as np
import scipy.linalg
import scipy.optimize
import time

def _build_hamiltonians():
    """
    Builds the odd (A) and even (B) bond Hamiltonians for a 6-site open XXZ Heisenberg chain.
    Anisotropy delta = 0.8, field h = 0.3.
    """
    delta = 0.8
    h_field = 0.3
    N = 6
    dim = 2 ** N
    
    # Pauli matrices
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    
    # Helper to apply operator at site i
    def op_at_site(site, op):
        ops = [np.eye(2)] * N
        ops[site] = op
        result = np.eye(2)
        for o in ops:
            result = np.kron(result, o)
        return result
    
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    # Field term: h * sum sz_i / 2 (split equally)
    field_term = 0.5 * h_field * sum(op_at_site(i, sz) for i in range(N))
    
    # Bond terms: sx*sx + sy*sy + delta*sz*sz
    for i in range(N - 1):
        bond = op_at_site(i, sx) @ op_at_site(i+1, sx) + \
               op_at_site(i, sy) @ op_at_site(i+1, sy) + \
               delta * op_at_site(i, sz) @ op_at_site(i+1, sz)
        
        if i % 2 == 0: # Even bond index (0, 2, 4) -> usually called "even" in 1-based 1-2, 3-4? 
            # Task says "even and odd bonds". 
            # 1-based: bonds (1,2), (2,3), (3,4), (4,5), (5,6).
            # Odd bonds: (1,2), (3,4), (5,6). Indices 0, 2, 4 in 0-based.
            # Even bonds: (2,3), (4,5). Indices 1, 3 in 0-based.
            A += bond
        else:
            B += bond
            
    A += field_term
    B += field_term
    
    return A, B

def _compute_error(a, b, A, B, times):
    """
    Computes the mean spectral norm error for given coefficients.
    U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A)
    """
    dim = A.shape[0]
    U = np.eye(dim, dtype=complex)
    errors = []
    for t in times:
        # Compute U(t)
        U_t = np.eye(dim, dtype=complex)
        # Order k=1..12
        for k in range(12):
            U_t = np.linalg.multi_dot([
                scipy.linalg.expm(-1j * b[k] * t * B),
                scipy.linalg.expm(-1j * a[k] * t * A),
                U_t
            ])
        
        # Compute exact propagator
        H = A + B
        U_exact = scipy.linalg.expm(-1j * t * H)
        
        # Spectral norm of difference
        diff = U_t - U_exact
        # Use svd for spectral norm
        s = np.linalg.svd(diff, compute_uv=False)
        errors.append(s[0])
        
    return np.mean(errors)

def coefficients(stages):
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    A, B = _build_hamiltonians()
    times = [0.25, 0.5, 1.0]
    
    # Initial guess: 4th order symmetric splitting
    # Solve for x_1..x_6 such that sum x=1/2, sum x^2=1/4, sum x^3=1/6
    # Use a simple optimizer to find such x
    def moment_residuals(x):
        return [
            np.sum(x) - 0.5,
            np.sum(x**2) - 0.25,
            np.sum(x**3) - 1.0/6.0
        ]
    
    def moment_objective(x):
        r = moment_residuals(x)
        return np.sum(r**2)
    
    x0 = np.full(6, 1.0/12.0)
    # Try to find a symmetric solution for moments
    try:
        res_moments = scipy.optimize.minimize(moment_objective, x0, method='Nelder-Mead', options={'maxiter': 1000})
        x_opt = res_moments.x
    except:
        x_opt = x0
        
    # Construct initial a, b
    a_init = np.array([x_opt[0], x_opt[1], x_opt[2], x_opt[3], x_opt[4], x_opt[5], 
                       x_opt[5], x_opt[4], x_opt[3], x_opt[2], x_opt[1], x_opt[0]])
    b_init = a_init.copy()
    
    # Flatten for optimizer
    x_flat_init = np.concatenate([a_init, b_init])
    
    def objective(x_flat):
        a = x_flat[:12]
        b = x_flat[12:]
        # Ensure finite
        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            return 1e10
        # Clip to bounds to help optimizer, though SLSQP handles bounds
        a_clipped = np.clip(a, -50, 50)
        b_clipped = np.clip(b, -50, 50)
        return _compute_error(a_clipped, b_clipped, A, B, times)
    
    def constraint_sum_a(x_flat):
        return np.sum(x_flat[:12]) - 1.0
        
    def constraint_sum_b(x_flat):
        return np.sum(x_flat[12:]) - 1.0
        
    constraints = [
        {'type': 'eq', 'fun': constraint_sum_a},
        {'type': 'eq', 'fun': constraint_sum_b}
    ]
    
    bounds = [(-50, 50)] * 24
    
    # Optimize
    best_x = x_flat_init
    best_score = objective(best_x)
    
    try:
        res = scipy.optimize.minimize(objective, best_x, method='SLSQP', 
                                      bounds=bounds, constraints=constraints, 
                                      options={'maxiter': 50, 'ftol': 1e-8})
        if res.fun < best_score:
            best_x = res.x
            best_score = res.fun
    except Exception as e:
        pass
        
    # Check time
    if time.time() > deadline:
        a = np.clip(best_x[:12], -50, 50)
        b = np.clip(best_x[12:], -50, 50)
        # Normalize to ensure sum=1 strictly
        if np.sum(a) != 0:
            a = a / np.sum(a)
        if np.sum(b) != 0:
            b = b / np.sum(b)
        return a.tolist(), b.tolist()
        
    # Polish with more iterations if time allows
    try:
        res2 = scipy.optimize.minimize(objective, best_x, method='SLSQP', 
                                       bounds=bounds, constraints=constraints, 
                                       options={'maxiter': 100, 'ftol': 1e-10})
        if res2.fun < best_score:
            best_x = res2.x
            best_score = res2.fun
    except:
        pass
        
    a = np.clip(best_x[:12], -50, 50)
    b = np.clip(best_x[12:], -50, 50)
    
    # Final normalization
    sum_a = np.sum(a)
    sum_b = np.sum(b)
    if sum_a != 0:
        a = a / sum_a
    if sum_b != 0:
        b = b / sum_b
        
    return a.tolist(), b.tolist()

# EVOLVE-BLOCK-START
def coefficients(stages):
    """
    Finds coefficients for a 12-stage product formula approximating Heisenberg chain evolution.
    Uses 4th order symmetric splitting as initialization and refines with SLSQP.
    """
    return _coefficients_impl(stages)

def _coefficients_impl(stages):
    # The logic from the draft above
    start_time = time.time()
    deadline = start_time + 50.0
    
    A, B = _build_hamiltonians()
    times = [0.25, 0.5, 1.0]
    
    # Initial guess: 4th order symmetric splitting moments
    def moment_objective(x):
        r0 = np.sum(x) - 0.5
        r1 = np.sum(x**2) - 0.25
        r2 = np.sum(x**3) - 1.0/6.0
        return r0**2 + r1**2 + r2**2

    x0 = np.full(6, 1.0/12.0)
    try:
        res_moments = scipy.optimize.minimize(moment_objective, x0, method='Nelder-Mead', options={'maxiter': 1000})
        x_opt = res_moments.x
    except:
        x_opt = x0
        
    a_init = np.array([x_opt[0], x_opt[1], x_opt[2], x_opt[3], x_opt[4], x_opt[5], 
                       x_opt[5], x_opt[4], x_opt[3], x_opt[2], x_opt[1], x_opt[0]])
    b_init = a_init.copy()
    x_flat_init = np.concatenate([a_init, b_init])
    
    def objective(x_flat):
        a = x_flat[:12]
        b = x_flat[12:]
        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            return 1e10
        a_clipped = np.clip(a, -50, 50)
        b_clipped = np.clip(b, -50, 50)
        return _compute_error(a_clipped, b_clipped, A, B, times)
        
    def constraint_sum_a(x_flat):
        return np.sum(x_flat[:12]) - 1.0
        
    def constraint_sum_b(x_flat):
        return np.sum(x_flat[12:]) - 1.0
        
    constraints = [
        {'type': 'eq', 'fun': constraint_sum_a},
        {'type': 'eq', 'fun': constraint_sum_b}
    ]
    bounds = [(-50, 50)] * 24
    
    best_x = x_flat_init
    best_score = objective(best_x)
    
    try:
        res = scipy.optimize.minimize(objective, best_x, method='SLSQP', 
                                      bounds=bounds, constraints=constraints, 
                                      options={'maxiter': 50, 'ftol': 1e-8})
        if res.fun < best_score:
            best_x = res.x
            best_score = res.fun
    except:
        pass
        
    if time.time() > deadline:
        a = np.clip(best_x[:12], -50, 50)
        b = np.clip(best_x[12:], -50, 50)
        if np.sum(a) != 0: a = a / np.sum(a)
        if np.sum(b) != 0: b = b / np.sum(b)
        return a.tolist(), b.tolist()
        
    try:
        res2 = scipy.optimize.minimize(objective, best_x, method='SLSQP', 
                                       bounds=bounds, constraints=constraints, 
                                       options={'maxiter': 100, 'ftol': 1e-10})
        if res2.fun < best_score:
            best_x = res2.x
            best_score = res2.fun
    except:
        pass
        
    a = np.clip(best_x[:12], -50, 50)
    b = np.clip(best_x[12:], -50, 50)
    if np.sum(a) != 0: a = a / np.sum(a)
    if np.sum(b) != 0: b = b / np.sum(b)
    
    return a.tolist(), b.tolist()
# EVOLVE-BLOCK-END
