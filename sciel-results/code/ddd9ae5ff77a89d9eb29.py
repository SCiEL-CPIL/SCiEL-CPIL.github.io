"""Combined approach: Nelder-Mead optimization with cached eigen-decompositions."""
import numpy as np
import time

def _build_hamiltonians():
    """
    Build the A and B Hamiltonians for a 6-site open XXZ Heisenberg chain.
    A: odd bonds + half field
    B: even bonds + half field
    Anisotropy lambda = 0.8, field h = 0.3
    """
    N = 6
    lambda_aniso = 0.8
    h_field = 0.3
    
    # Spin operators
    sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    
    def op_on_site(op, site, N):
        ops = [np.eye(2, dtype=complex)] * N
        ops[site] = op
        result = ops[0]
        for i in range(1, N):
            result = np.kron(result, ops[i])
        return result

    A = np.zeros((2**N, 2**N), dtype=complex)
    B = np.zeros((2**N, 2**N), dtype=complex)
    
    # Field term: h * sum Sz_i / 2, split equally between A and B
    field_term = 0.5 * h_field
    for i in range(N):
        A += field_term * op_on_site(sz, i, N)
        B += field_term * op_on_site(sz, i, N)
    
    def add_bond(H, i, j):
        sx_i = op_on_site(sx, i, N)
        sx_j = op_on_site(sx, j, N)
        sy_i = op_on_site(sy, i, N)
        sy_j = op_on_site(sy, j, N)
        sz_i = op_on_site(sz, i, N)
        sz_j = op_on_site(sz, j, N)
        
        H += sx_i @ sx_j
        H += sy_i @ sy_j
        H += lambda_aniso * (sz_i @ sz_j)
    
    # Odd bonds for A: (0,1), (2,3), (4,5)
    add_bond(A, 0, 1)
    add_bond(A, 2, 3)
    add_bond(A, 4, 5)
    
    # Even bonds for B: (1,2), (3,4)
    add_bond(B, 1, 2)
    add_bond(B, 3, 4)
    
    return A, B

def _nelder_mead(f, x0, max_iter=200, tol=1e-8, alpha=1.0, gamma=2.0, rho=0.5, sigma=2.0):
    """
    Nelder-Mead simplex optimization.
    f: objective function
    x0: initial point (array)
    Returns: best_x, best_f
    """
    n = len(x0)
    # Initialize simplex
    simplex = np.zeros((n+1, n))
    simplex[0] = x0
    for i in range(n):
        simplex[i+1] = x0.copy()
        simplex[i+1][i] += 0.05 if x0[i] == 0 else 0.05 * x0[i]
    
    # Evaluate initial simplex
    fvals = np.array([f(simplex[i]) for i in range(n+1)])
    
    best_x = simplex[np.argmin(fvals)].copy()
    best_f = np.min(fvals)
    
    for it in range(max_iter):
        # Sort simplex by function value
        order = np.argsort(fvals)
        simplex = simplex[order]
        fvals = fvals[order]
        
        # Check convergence
        if abs(fvals[0] - fvals[-1]) < tol:
            break
        
        # Centroid of all but worst point
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        x_ref = centroid + alpha * (centroid - simplex[-1])
        f_ref = f(x_ref)
        
        if f_ref < fvals[0]:
            # Expansion
            x_exp = centroid + gamma * (x_ref - centroid)
            f_exp = f(x_exp)
            if f_exp < f_ref:
                simplex[-1] = x_exp
                fvals[-1] = f_exp
            else:
                simplex[-1] = x_ref
                fvals[-1] = f_ref
        elif f_ref < fvals[-2]:
            simplex[-1] = x_ref
            fvals[-1] = f_ref
        else:
            # Contraction
            x_con = centroid + rho * (simplex[-1] - centroid)
            f_con = f(x_con)
            if f_con < fvals[-1]:
                simplex[-1] = x_con
                fvals[-1] = f_con
            else:
                # Shrink
                for i in range(1, n+1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    fvals[i] = f(simplex[i])
        
        # Update best
        idx = np.argmin(fvals)
        if fvals[idx] < best_f:
            best_f = fvals[idx]
            best_x = simplex[idx].copy()
    
    return best_x, best_f

def coefficients(stages):
    """
    Optimize 12-stage product formula coefficients using Nelder-Mead.
    """
    start_time = time.time()
    deadline = start_time + 50.0  # Leave 10s margin for 60s limit
    
    # Pre-compute Hamiltonians
    A, B = _build_hamiltonians()
    H_total = A + B
    
    # Pre-compute exact propagators for the 3 time points
    t_values = [0.25, 0.5, 1.0]
    
    # Diagonalize A and B for faster exponential calculation
    eigvals_A, eigvecs_A = np.linalg.eigh(A)
    eigvals_B, eigvecs_B = np.linalg.eigh(B)
    
    # Pre-compute exact propagators
    eigvals_H, eigvecs_H = np.linalg.eigh(H_total)
    U_exacts = []
    for t in t_values:
        exp_H = np.exp(-1j * t * eigvals_H)
        U_exact = (eigvecs_H * exp_H) @ eigvecs_H.conj().T
        U_exacts.append(U_exact)
    
    def apply_product_formula(a, b, t, eigvals_A, eigvecs_A, eigvals_B, eigvecs_B):
        """Apply the product formula using pre-computed eigen-decompositions."""
        dim = 64
        U_curr = np.eye(dim, dtype=complex)
        
        for k in range(12):
            # Step 1: exp(-i b_k t B)
            exp_B = np.exp(-1j * b[k] * t * eigvals_B)
            U_B = (eigvecs_B * exp_B) @ eigvecs_B.conj().T
            U_curr = U_B @ U_curr
            
            # Step 2: exp(-i a_k t A)
            exp_A = np.exp(-1j * a[k] * t * eigvals_A)
            U_A = (eigvecs_A * exp_A) @ eigvecs_A.conj().T
            U_curr = U_A @ U_curr
        
        return U_curr
    
    def objective(x):
        """
        Objective function: mean spectral norm error over 3 time points.
        x: 24-element array representing [a_1..a_12, b_1..b_12]
        """
        # Check time
        if time.time() > deadline:
            return float('inf')
        
        a = x[:12]
        b = x[12:]
        
        # Penalty for violating sum constraints
        sum_a = np.sum(a)
        sum_b = np.sum(b)
        penalty = 100.0 * ((sum_a - 1.0)**2 + (sum_b - 1.0)**2)
        
        # Penalty for values outside [-50, 50]
        out_of_range = np.maximum(0, np.abs(a) - 50) + np.maximum(0, np.abs(b) - 50)
        penalty += 100.0 * np.sum(out_of_range**2)
        
        if penalty > 1.0:
            return penalty
        
        error = 0.0
        for i, t in enumerate(t_values):
            U_approx = apply_product_formula(a, b, t, eigvals_A, eigvecs_A, eigvals_B, eigvecs_B)
            diff = U_approx - U_exacts[i]
            err = np.linalg.norm(diff, 2)
            error += err
        
        return error / len(t_values) + penalty
    
    # Initial guess: uniform distribution
    x0 = np.ones(24) / 12.0
    
    # Run Nelder-Mead
    best_x, best_f = _nelder_mead(objective, x0, max_iter=150, tol=1e-10)
    
    # Extract coefficients
    a = best_x[:12].tolist()
    b = best_x[12:].tolist()
    
    # Ensure sums are exactly 1 by normalizing (with small adjustment)
    sum_a = sum(a)
    sum_b = sum(b)
    if abs(sum_a - 1.0) > 1e-10:
        a = [ai / sum_a for ai in a]
    if abs(sum_b - 1.0) > 1e-10:
        b = [bi / sum_b for bi in b]
    
    # Clamp to [-50, 50]
    a = [max(-50.0, min(50.0, ai)) for ai in a]
    b = [max(-50.0, min(50.0, bi)) for bi in b]
    
    # Re-normalize after clamping
    sum_a = sum(a)
    sum_b = sum(b)
    if abs(sum_a - 1.0) > 1e-10:
        a = [ai / sum_a for ai in a]
    if abs(sum_b - 1.0) > 1e-10:
        b = [bi / sum_b for bi in b]
    
    return a, b
