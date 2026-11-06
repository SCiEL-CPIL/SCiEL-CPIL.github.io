"""Product-formula design for Heisenberg-chain time evolution using 4th-order Suzuki-Trotter."""
import time

# EVOLVE-BLOCK-START
def coefficients(stages):
    """
    Optimized 12-stage product formula for Heisenberg-chain time evolution.
    """
    import numpy as np
    from scipy.linalg import expm
    from scipy.optimize import minimize
    
    deadline = time.time() + 135.0
    
    # Build the 6-site open XXZ Heisenberg chain Hamiltonian
    sx = 0.5 * np.array([[0, 1], [1, 0]])
    sy = 0.5 * np.array([[0, -1j], [1j, 0]])
    sz = 0.5 * np.array([[1, 0], [0, -1]])
    
    n = 6
    dim = 2 ** n
    anisotropy = 0.8
    field = 0.3
    
    def kron_all(ops):
        result = np.asarray(ops[0])
        for op in ops[1:]:
            result = np.kron(result, np.asarray(op))
        return result
    
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    for (i, j) in [(0, 1), (2, 3), (4, 5)]:
        for a_op, b_op, coeff in [(sx, sx, 1.0), (sy, sy, 1.0), (sz, sz, anisotropy)]:
            ops = [np.eye(2)] * n
            ops[i] = a_op
            ops[j] = b_op
            A += coeff * kron_all(ops)
    
    for (i, j) in [(1, 2), (3, 4)]:
        for a_op, b_op, coeff in [(sx, sx, 1.0), (sy, sy, 1.0), (sz, sz, anisotropy)]:
            ops = [np.eye(2)] * n
            ops[i] = a_op
            ops[j] = b_op
            B += coeff * kron_all(ops)
    
    for i in range(n):
        ops = [np.eye(2)] * n
        ops[i] = sz
        A += 0.5 * field * kron_all(ops)
        B += 0.5 * field * kron_all(ops)
    
    # Precompute exact propagators
    times = [0.25, 0.5, 1.0]
    U_exacts = [expm(-1j * t * (A + B)) for t in times]
    
    def compute_error(params):
        a = params[:stages]
        b = params[stages:]
        
        if np.any(a > 50) or np.any(a < -50) or np.any(b > 50) or np.any(b < -50):
            return 1e10
        
        total_error = 0.0
        for idx, t in enumerate(times):
            U = np.eye(dim, dtype=complex)
            for k in range(stages):
                U = expm(-1j * b[k] * t * B) @ U
                U = expm(-1j * a[k] * t * A) @ U
            
            diff = U - U_exacts[idx]
            s = np.linalg.svd(diff, compute_uv=False)
            total_error += s[0] if len(s) > 0 else 0.0
        
        return total_error / 3.0
    
    # Initial guess: 4th-order Suzuki-Trotter
    c1 = 1.0 / (2.0 - 2.0**(1.0/3.0))
    c2 = -2.0**(1.0/3.0) / (2.0 - 2.0**(1.0/3.0))
    
    n_substeps = stages // 4
    if n_substeps < 1:
        n_substeps = 1
    
    a_init = []
    b_init = []
    scale = 1.0 / n_substeps
    a1 = c1 / 2.0 * scale
    a2 = (c1 + c2) / 2.0 * scale
    b1 = c1 * scale
    b2 = c2 * scale
    
    for _ in range(n_substeps):
        a_init.append(a1)
        b_init.append(0.0)
        a_init.append(a2)
        b_init.append(b1)
        a_init.append(a2)
        b_init.append(b2)
        a_init.append(a1)
        b_init.append(b1)
    
    while len(a_init) < stages:
        a_init.append(0.0)
        b_init.append(0.0)
    
    a_init = a_init[:stages]
    b_init = b_init[:stages]
    x0 = np.array(a_init + b_init)
    
    best_result = x0
    best_cost = compute_error(x0)
    
    # Quick optimization
    if time.time() < deadline - 30:
        try:
            result = minimize(
                compute_error, x0,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
            if result.fun < best_cost:
                best_result = result.x
                best_cost = result.fun
        except Exception:
            pass
    
    a = list(best_result[:stages])
    b = list(best_result[stages:])
    
    # Normalize
    sum_a = sum(a)
    sum_b = sum(b)
    
    if abs(sum_a) > 1e-15:
        a = [x / sum_a for x in a]
    if abs(sum_b) > 1e-15:
        b = [x / sum_b for x in b]
    
    # Clamp
    a = [max(-50.0, min(50.0, x)) for x in a]
    b = [max(-50.0, min(50.0, x)) for x in b]
    
    return a, b
# EVOLVE-BLOCK-END
