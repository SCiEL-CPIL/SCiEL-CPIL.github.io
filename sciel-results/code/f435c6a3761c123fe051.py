"""Nelder-Mead optimization of 12-stage product formula coefficients for Heisenberg chain."""
import numpy as np
import time
import sys

def _build_hamiltonians():
    """
    Build A (odd bonds + half field) and B (even bonds + half field) for a 6-site
    open XXZ Heisenberg chain with anisotropy 0.8 and field 0.3.
    
    The full Hamiltonian H = A + B where:
    - A = sum over odd bonds (1-2, 3-4, 5-6) of XXZ interaction + half uniform field
    - B = sum over even bonds (2-3, 4-5) of XXZ interaction + half uniform field
    
    XXZ interaction on bond (i, i+1): Sx_i Sx_{i+1} + Sy_i Sy_{i+1} + lambda Sz_i Sz_{i+1}
    Uniform field: h * sum_i Sz_i, split as h/2 in A and h/2 in B
    """
    N = 6
    lam = 0.8
    h = 0.3
    
    # Spin-1/2 operators
    sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    identity = np.eye(2, dtype=complex)
    
    def kron_all(ops):
        """Tensor product of a list of 2x2 matrices."""
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result
    
    def bond_interaction(i, j, lam):
        """XXZ interaction between sites i and j (0-indexed)."""
        ops = [identity] * N
        ops[i] = sx
        ops[j] = sx
        term1 = kron_all(ops)
        
        ops[i] = sy
        ops[j] = sy
        term2 = kron_all(ops)
        
        ops[i] = sz
        ops[j] = sz
        term3 = lam * kron_all(ops)
        
        return term1 + term2 + term3
    
    def field_term(lam):
        """Uniform field: h * sum_i Sz_i, scaled by lam (0.5 for half field)."""
        total = np.zeros((2**N, 2**N), dtype=complex)
        for i in range(N):
            ops = [identity] * N
            ops[i] = sz
            total += lam * kron_all(ops)
        return total
    
    # Odd bonds: (0,1), (2,3), (4,5) - sites 1-2, 3-4, 5-6
    A = np.zeros((2**N, 2**N), dtype=complex)
    for i in [0, 2, 4]:
        A += bond_interaction(i, i+1, lam)
    A += field_term(h/2)
    
    # Even bonds: (1,2), (3,4) - sites 2-3, 4-5
    B = np.zeros((2**N, 2**N), dtype=complex)
    for i in [1, 3]:
        B += bond_interaction(i, i+1, lam)
    B += field_term(h/2)
    
    return A, B

def _compute_error(coeffs, A, B, times):
    """
    Compute the mean spectral norm error over the given times.
    coeffs: 24-element array [a1, b1, a2, b2, ..., a12, b12]
    """
    a = coeffs[0::2]
    b = coeffs[1::2]
    
    error_sum = 0.0
    for t in times:
        U = np.eye(2**6, dtype=complex)
        for k in range(12):
            U = U @ np.linalg.matrix_exp(-1j * b[k] * t * B)
            U = U @ np.linalg.matrix_exp(-1j * a[k] * t * A)
        
        U_exact = np.linalg.matrix_exp(-1j * t * (A + B))
        diff = U - U_exact
        # Spectral norm is the largest singular value
        error_sum += np.linalg.norm(diff, 2)
    
    return error_sum / len(times)

def _nelder_mead(f, x0, bounds, max_iter=2000, tol=1e-10, deadline=None):
    """
    Nelder-Mead optimization with bounds handling.
    f: objective function
    x0: initial point
    bounds: list of (low, high) for each dimension
    """
    n = len(x0)
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    # Create initial simplex
    simplex = [np.array(x0, dtype=float)]
    for i in range(n):
        xi = np.array(x0, dtype=float)
        if xi[i] != 0:
            xi[i] *= 1.05
        else:
            xi[i] = 1e-4
        simplex.append(xi)
    
    # Evaluate initial simplex
    fvals = [f(x) for x in simplex]
    
    # Sort simplex by function value
    order = np.argsort(fvals)
    simplex = [simplex[i] for i in order]
    fvals = [fvals[i] for i in order]
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute centroid of all but worst point
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        xr = centroid + alpha * (centroid - simplex[-1])
        # Enforce bounds
        for i in range(n):
            xr[i] = np.clip(xr[i], bounds[i][0], bounds[i][1])
        fr = f(xr)
        
        if fr < fvals[0]:
            # Expansion
            xe = centroid + gamma * (xr - centroid)
            for i in range(n):
                xe[i] = np.clip(xe[i], bounds[i][0], bounds[i][1])
            fe = f(xe)
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
        elif fr < fvals[-2]:
            # Accept reflection
            simplex[-1] = xr
            fvals[-1] = fr
        else:
            # Contraction
            if fr < fvals[-1]:
                # Outside contraction
                xc = centroid + rho * (xr - centroid)
            else:
                # Inside contraction
                xc = centroid - rho * (simplex[-1] - centroid)
            for i in range(n):
                xc[i] = np.clip(xc[i], bounds[i][0], bounds[i][1])
            fc = f(xc)
            
            if fc < min(fr, fvals[-1]):
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                # Shrink
                for i in range(1, n+1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    for j in range(n):
                        simplex[i][j] = np.clip(simplex[i][j], bounds[j][0], bounds[j][1])
                    fvals[i] = f(simplex[i])
        
        # Sort simplex
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        # Convergence check
        if abs(fvals[0] - fvals[-1]) < tol:
            break
    
    return simplex[0], fvals[0]

def coefficients(stages):
    """
    Optimize the 12-stage product formula coefficients.
    
    The product formula is: U(t) = prod_{k=1}^{12} exp(-i b_k t B) exp(-i a_k t A)
    We minimize the mean spectral norm error over t = 0.25, 0.5, 1.0.
    
    Returns (a, b) where a and b are lists of 12 floats each.
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for the 150s limit
    
    # Build Hamiltonians
    A, B = _build_hamiltonians()
    
    # Target times
    times = [0.25, 0.5, 1.0]
    
    # Initial guess: uniform splitting
    a0 = [1.0/12] * 12
    b0 = [1.0/12] * 12
    
    # Convert to parameterization that enforces sum constraints.
    # Use a_k = 1/12 + x_k, b_k = 1/12 + y_k where sum(x) = 0, sum(y) = 0.
    # To enforce sum=0, we can use a parameterization where x_12 = -sum(x_1..x_11).
    # So we optimize over 11+11=22 free parameters.
    
    def objective(params):
        x = params[:11]
        y = params[11:]
        
        a = [1.0/12 + xi for xi in x]
        a.append(1.0/12 - sum(x))
        
        b = [1.0/12 + yi for yi in y]
        b.append(1.0/12 - sum(y))
        
        # Check bounds: all coefficients in [-50, 50]
        for val in a + b:
            if val < -50 or val > 50:
                return 1e10
        
        coeffs = []
        for k in range(12):
            coeffs.append(a[k])
            coeffs.append(b[k])
        
        return _compute_error(np.array(coeffs), A, B, times)
    
    # Initial parameters: all zeros (corresponds to uniform splitting)
    x0 = np.zeros(22)
    
    # Bounds for the free parameters
    # a_k = 1/12 + x_k must be in [-50, 50], so x_k in [-50 - 1/12, 50 - 1/12]
    bound_low = -50 - 1/12
    bound_high = 50 - 1/12
    bounds = [(bound_low, bound_high)] * 22
    
    # Run Nelder-Mead
    try:
        best_params, best_val = _nelder_mead(objective, x0, bounds, max_iter=5000, tol=1e-12, deadline=deadline)
    except Exception:
        # Fallback to uniform splitting
        return a0, b0
    
    # Extract coefficients
    x = best_params[:11]
    y = best_params[11:]
    
    a = [1.0/12 + xi for xi in x]
    a.append(1.0/12 - sum(x))
    
    b = [1.0/12 + yi for yi in y]
    b.append(1.0/12 - sum(y))
    
    # Clamp to valid range
    a = [max(-50.0, min(50.0, val)) for val in a]
    b = [max(-50.0, min(50.0, val)) for val in b]
    
    # Re-normalize to ensure sum = 1 exactly
    sum_a = sum(a)
    sum_b = sum(b)
    if abs(sum_a) > 1e-15:
        a = [val / sum_a for val in a]
    if abs(sum_b) > 1e-15:
        b = [val / sum_b for val in b]
    
    return a, b
