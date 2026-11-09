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
    
    # Pauli matrices / spin-1/2 operators
    sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    
    def kron_all(ops):
        """Tensor product of a list of operators."""
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result
    
    def bond_interaction(i, j, lam):
        """XXZ interaction on bond between sites i and j (0-indexed)."""
        ops = [I2] * N
        ops[i] = sx
        ops[j] = sx
        term_x = kron_all(ops)
        
        ops[i] = sy
        ops[j] = sy
        term_y = kron_all(ops)
        
        ops[i] = sz
        ops[j] = sz
        term_z = kron_all(ops)
        
        return term_x + term_y + lam * term_z
    
    # Build A: odd bonds (0-1, 2-3, 4-5) + half field
    A = np.zeros((2**N, 2**N), dtype=complex)
    for i in range(0, N, 2):
        A += bond_interaction(i, i+1, lam)
    # Add half field to A
    for i in range(N):
        ops = [I2] * N
        ops[i] = sz
        A += (h/2) * kron_all(ops)
    
    # Build B: even bonds (1-2, 3-4) + half field
    B = np.zeros((2**N, 2**N), dtype=complex)
    for i in range(1, N, 2):
        B += bond_interaction(i, i+1, lam)
    # Add half field to B
    for i in range(N):
        ops = [I2] * N
        ops[i] = sz
        B += (h/2) * kron_all(ops)
    
    return A, B

def _compute_error(a, b, A, B, t):
    """Compute spectral norm error for given coefficients and time t."""
    N = len(a)
    # Build product U(t) = prod_{k=1}^{N} exp(-i b_k t B) exp(-i a_k t A)
    # In order k=1..N
    U = np.eye(2**6, dtype=complex)
    for k in range(N):
        # exp(-i a_k t A)
        if abs(a[k]) > 1e-15:
            # Use eigendecomposition for exp
            eigvals, eigvecs = np.linalg.eig(A)
            exp_A = eigvecs @ np.diag(np.exp(-1j * a[k] * t * eigvals)) @ np.linalg.inv(eigvecs)
            U = exp_A @ U
        if abs(b[k]) > 1e-15:
            eigvals, eigvecs = np.linalg.eig(B)
            exp_B = eigvecs @ np.diag(np.exp(-1j * b[k] * t * eigvals)) @ np.linalg.inv(eigvecs)
            U = exp_B @ U
    
    # Exact propagator
    H = A + B
    eigvals, eigvecs = np.linalg.eig(H)
    U_exact = eigvecs @ np.diag(np.exp(-1j * t * eigvals)) @ np.linalg.inv(eigvecs)
    
    # Spectral norm of difference
    diff = U - U_exact
    s = np.linalg.svd(diff, compute_uv=False)
    return s[0]

def _objective(params):
    """Objective function: mean spectral norm error at t = 0.25, 0.5, 1.0."""
    a = params[:12]
    b = params[12:]
    
    # Enforce sum constraints by projecting
    sum_a = np.sum(a)
    sum_b = np.sum(b)
    if abs(sum_a) > 1e-15:
        a = a / sum_a
    else:
        a = np.ones(12) / 12
    if abs(sum_b) > 1e-15:
        b = b / sum_b
    else:
        b = np.ones(12) / 12
    
    A, B = _build_hamiltonians()
    times = [0.25, 0.5, 1.0]
    errors = []
    for t in times:
        err = _compute_error(a, b, A, B, t)
        errors.append(err)
    return np.mean(errors)

def _nelder_mead(objective, x0, max_iter=500, tol=1e-8, deadline=None):
    """Nelder-Mead simplex optimization with deadline guard."""
    n = len(x0)
    
    # Initialize simplex
    simplex = [np.array(x0, dtype=float)]
    for i in range(n):
        xi = np.array(x0, dtype=float)
        step = 0.1 if x0[i] == 0 else 0.05 * abs(x0[i])
        xi[i] += step
        simplex.append(xi)
    
    fvals = [objective(x) for x in simplex]
    
    # Alpha, gamma, rho, sigma
    alpha = 0.5
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        # Sort simplex
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        # Check convergence
        if abs(fvals[0] - fvals[-1]) < tol:
            break
        
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = objective(xr)
        
        if fr < fvals[0]:
            # Expansion
            xe = centroid + gamma * (centroid - simplex[-1])
            fe = objective(xe)
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
            # Contraction
            if fr < fvals[-1]:
                # Outside contraction
                xc = centroid + 0.5 * (xr - centroid)
                fc = objective(xc)
                if fc <= fr:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n+1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = objective(simplex[i])
            else:
                # Inside contraction
                xc = centroid - 0.5 * (centroid - simplex[-1])
                fc = objective(xc)
                if fc < fvals[-1]:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n+1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = objective(simplex[i])
    
    # Return best point
    best_idx = np.argmin(fvals)
    return simplex[best_idx], fvals[best_idx]

def coefficients(stages):
    """Optimize 12-stage product formula coefficients using Nelder-Mead."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Build Hamiltonians once
    A, B = _build_hamiltonians()
    
    # Initial guess: symmetric splitting
    x0 = np.zeros(24)
    for k in range(12):
        x0[k] = 1.0/12      # a coefficients
        x0[12+k] = 1.0/12   # b coefficients
    
    # Run Nelder-Mead with deadline
    best_params, best_val = _nelder_mead(_objective, x0, max_iter=300, tol=1e-10, deadline=deadline)
    
    # Extract and normalize
    a = best_params[:12]
    b = best_params[12:]
    
    # Ensure sums are exactly 1
    sum_a = np.sum(a)
    sum_b = np.sum(b)
    if abs(sum_a) > 1e-15:
        a = a / sum_a
    else:
        a = np.ones(12) / 12
    if abs(sum_b) > 1e-15:
        b = b / sum_b
    else:
        b = np.ones(12) / 12
    
    # Clip to [-50, 50] and re-normalize if needed
    a = np.clip(a, -50, 50)
    b = np.clip(b, -50, 50)
    
    # Final normalization
    sum_a = np.sum(a)
    sum_b = np.sum(b)
    if abs(sum_a) > 1e-15:
        a = a / sum_a
    if abs(sum_b) > 1e-15:
        b = b / sum_b
    
    # Check deadline
    if time.time() > deadline - 2:
        # Return simple symmetric fallback
        a = [1.0/12] * 12
        b = [1.0/12] * 12
    else:
        a = list(a)
        b = list(b)
    
    return a, b
