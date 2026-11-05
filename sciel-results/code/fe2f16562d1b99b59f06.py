import numpy as np
import time
from scipy.optimize import minimize

def _build_hamiltonians():
    """Build A (odd bonds + half field) and B (even bonds + half field) for 6-site XXZ chain."""
    N = 6
    dim = 2 ** N
    anisotropy = 0.8
    field = 0.3
    
    # Pauli matrices
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    
    # Build operators for each site
    def op_on_site(op, site):
        """Apply op on given site (0-indexed from left)."""
        ops = [I2] * N
        ops[site] = op
        result = ops[0]
        for i in range(1, N):
            result = np.kron(result, ops[i])
        return result
    
    # XXZ interaction: J*(sx*sx + sy*sy + anisotropy*sz*sz) / 2
    # Standard Heisenberg: H = J * sum (sx_i sx_{i+1} + sy_i sy_{i+1} + anisotropy * sz_i sz_{i+1})
    # Let's use J=1 for the bond terms
    J = 1.0
    
    # Odd bonds: (0,1), (2,3), (4,5)  -- sites 0-1, 2-3, 4-5
    # Even bonds: (1,2), (3,4)        -- sites 1-2, 3-4
    # Wait, for 6 sites open chain: bonds are (0,1),(1,2),(2,3),(3,4),(4,5)
    # Odd bonds (by bond index): bonds 0,2,4 -> (0,1),(2,3),(4,5)
    # Even bonds (by bond index): bonds 1,3 -> (1,2),(3,4)
    
    def bond_hamiltonian(i, j):
        """XXZ bond Hamiltonian between sites i and j."""
        sx_i = op_on_site(sx, i)
        sx_j = op_on_site(sx, j)
        sy_i = op_on_site(sy, i)
        sy_j = op_on_site(sy, j)
        sz_i = op_on_site(sz, i)
        sz_j = op_on_site(sz, j)
        return J * (sx_i @ sx_j + sy_i @ sy_j + anisotropy * sz_i @ sz_j)
    
    # Uniform field: h * sum sz_i, split half into A and half into B
    field_total = field * sum(op_on_site(sz, i) for i in range(N))
    half_field = 0.5 * field_total
    
    # A = odd bonds + half field
    A = bond_hamiltonian(0, 1) + bond_hamiltonian(2, 3) + bond_hamiltonian(4, 5) + half_field
    # B = even bonds + half field
    B = bond_hamiltonian(1, 2) + bond_hamiltonian(3, 4) + half_field
    
    return A, B

def _spectral_norm_error(a, b, A, B, t_values):
    """Compute mean spectral norm error over t_values."""
    dim = A.shape[0]
    AB = A + B
    exact_factors = [np.linalg.matrix_exp(-1j * t * AB) for t in t_values]
    
    total_err = 0.0
    for t in t_values:
        U = np.eye(dim, dtype=complex)
        for k in range(len(a)):
            U = np.linalg.matrix_exp(-1j * b[k] * t * B) @ U
            U = np.linalg.matrix_exp(-1j * a[k] * t * A) @ U
        err = np.linalg.norm(U - exact_factors[t_values.index(t)], 2)
        total_err += err
    return total_err / len(t_values)

def coefficients(stages):
    """Optimize 12-stage product formula coefficients."""
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s kill
    
    A, B = _build_hamiltonians()
    dim = A.shape[0]
    t_values = [0.25, 0.5, 1.0]
    
    # Precompute exact propagators
    AB = A + B
    exact_factors = [np.linalg.matrix_exp(-1j * t * AB) for t in t_values]
    
    # Precompute exp matrices for speed - but they depend on coefficients
    # Instead, use a smooth surrogate: minimize sum of ||U(t) - exact(t)||_F^2
    # which is differentiable and correlated with spectral norm
    
    def objective(x):
        if time.time() > deadline:
            return 1e10
        a = x[:stages]
        b = x[stages:2*stages]
        # Enforce sum constraints via penalty
        sum_a = np.sum(a)
        sum_b = np.sum(b)
        penalty = 1000.0 * (sum_a - 1.0)**2 + 1000.0 * (sum_b - 1.0)**2
        
        total_err = 0.0
        for idx, t in enumerate(t_values):
            U = np.eye(dim, dtype=complex)
            for k in range(stages):
                U = np.linalg.matrix_exp(-1j * b[k] * t * B) @ U
                U = np.linalg.matrix_exp(-1j * a[k] * t * A) @ U
            # Frobenius norm squared as smooth surrogate
            diff = U - exact_factors[idx]
            total_err += np.real(np.vdot(diff, diff))
        
        return total_err / len(t_values) + penalty
    
    # Initial guess: uniform split
    x0 = np.zeros(2 * stages)
    x0[:stages] = 1.0 / stages
    x0[stages:] = 1.0 / stages
    
    best_x = x0.copy()
    best_obj = objective(x0)
    
    # Try multiple starting points and local optimization
    start_points = [x0.copy()]
    
    # Add some structured initial guesses
    # Strang-like: alternate high/low
    for _ in range(5):
        x_try = x0.copy()
        # Random perturbations
        x_try += 0.01 * np.random.randn(2 * stages)
        start_points.append(x_try)
    
    # Known good patterns: symmetric coefficients
    # For symmetric product formulas, a_k = a_{stages+1-k}, b_k = b_{stages+1-k}
    for _ in range(10):
        x_sym = np.zeros(2 * stages)
        half = stages // 2
        # Random symmetric a
        a_half = np.random.uniform(-0.1, 0.3, half)
        x_sym[:half] = a_half
        x_sym[half:stages] = a_half[::-1]
        # Random symmetric b
        b_half = np.random.uniform(-0.1, 0.3, half)
        x_sym[stages:stages+half] = b_half
        x_sym[stages+half:2*stages] = b_half[::-1]
        # Normalize
        sa = np.sum(x_sym[:stages])
        sb = np.sum(x_sym[stages:2*stages])
        if abs(sa) > 1e-10:
            x_sym[:stages] /= sa
        if abs(sb) > 1e-10:
            x_sym[stages:2*stages] /= sb
        start_points.append(x_sym.copy())
    
    n_evals = 0
    for sp in start_points:
        if time.time() > deadline - 30.0:
            break
        n_evals += 1
        try:
            res = minimize(objective, sp, method='L-BFGS-B', 
                          options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6})
            if res.fun < best_obj:
                best_obj = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 15.0:
        try:
            res = minimize(objective, best_x, method='L-BFGS-B',
                          options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.fun < best_obj:
                best_obj = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Extract and clamp
    a = best_x[:stages].tolist()
    b = best_x[stages:2*stages].tolist()
    
    # Clamp to [-50, 50]
    a = [max(-50.0, min(50.0, v)) for v in a]
    b = [max(-50.0, min(50.0, v)) for v in b]
    
    # Normalize to ensure sum = 1
    sa = sum(a)
    sb = sum(b)
    if abs(sa) > 1e-12:
        a = [v / sa for v in a]
    else:
        a = [1.0/stages] * stages
    if abs(sb) > 1e-12:
        b = [v / sb for v in b]
    else:
        b = [1.0/stages] * stages
    
    return a, b
