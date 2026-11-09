import numpy as np
import time
import sys

def build_hamiltonians():
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
    # Sx, Sy, Sz for a single site
    sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    
    # Build operators for N sites
    def op_on_site(op, site, N):
        ops = [np.eye(2, dtype=complex)] * N
        ops[site] = op
        result = ops[0]
        for i in range(1, N):
            result = np.kron(result, ops[i])
        return result

    # Build A (odd bonds: 1-2, 3-4, 5-6) and B (even bonds: 2-3, 4-5)
    # Note: Open boundary condition, so bonds are 1-2, 2-3, 3-4, 4-5, 5-6.
    # Odd bonds: 1-2, 3-4, 5-6
    # Even bonds: 2-3, 4-5
    
    A = np.zeros((2**N, 2**N), dtype=complex)
    B = np.zeros((2**N, 2**N), dtype=complex)
    
    # Field term: h * sum Sz_i / 2
    # The problem says "each including half of the uniform field".
    # So A gets 0.5 * h * sum Sz, B gets 0.5 * h * sum Sz.
    field_term = 0.5 * h_field
    for i in range(N):
        A += field_term * op_on_site(sz, i, N)
        B += field_term * op_on_site(sz, i, N)
    
    # Bond terms
    # Helper to add bond interaction
    def add_bond(H, i, j):
        # H += Sx_i Sx_j + Sy_i Sy_j + lambda Sz_i Sz_j
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

def coefficients(stages):
    """
    Optimize 12-stage symmetric product formula coefficients.
    """
    start_time = time.time()
    deadline = start_time + 50.0  # Leave 10s margin for 60s limit
    
    # Pre-compute Hamiltonians
    A, B = build_hamiltonians()
    H_total = A + B
    
    # Pre-compute exact propagators for the 3 time points
    t_values = [0.25, 0.5, 1.0]
    U_exacts = []
    for t in t_values:
        # exp(-i t H_total)
        # Use scipy.linalg.expm
        U_exact = la.expm(-1j * t * H_total)
        U_exacts.append(U_exact)
    
    # Parameterization:
    # a = [x1, x2, x3, x4, x5, x6, x6, x5, x4, x3, x2, x1]
    # b = [x1, x2, x3, x4, x5, x6, x6, x5, x4, x3, x2, x1]
    # This ensures sum(a) = sum(b) = 2 * sum(x) = 1 => sum(x) = 0.5.
    # We optimize x1..x6.
    
    def objective(x):
        # x is array of 6 floats
        # Check time
        if time.time() > deadline:
            return float('inf')
            
        # Construct a and b
        # Symmetric sequence
        seq = [x[0], x[1], x[2], x[3], x[4], x[5], x[5], x[4], x[3], x[2], x[1], x[0]]
        a = seq
        b = seq
        
        # Check sum constraint (should be close to 1 if sum(x)=0.5, but optimizer might drift)
        # We enforce sum(x) = 0.5 by optimizing 5 variables and setting x6 = 0.5 - sum(x1..x5)?
        # No, Nelder-Mead doesn't handle constraints well.
        # Better: define x such that the constraint is met.
        # Let's stick to 6 variables and add a penalty if sum != 0.5.
        s = sum(x)
        penalty = (s - 0.5)**2 * 1000
        
        error = 0.0
        for i, t in enumerate(t_values):
            U_approx = np.eye(64, dtype=complex)
            # Product is B1 A1 B2 A2 ... B12 A12
            # U = prod_{k=1}^{12} exp(-i b_k t B) exp(-i a_k t A)
            # Note: The order in the product matters.
            # The prompt says: U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A) in order k = 1..12.
            # This means U = (...((B1 A1) B2 A2) ... B12 A12)
            # Matrix multiplication is associative.
            # We can compute this efficiently.
            
            # Pre-compute exp terms for this t?
            # exp(-i t A) and exp(-i t B) are constant for a given t.
            # But a_k and b_k vary.
            # We need exp(-i a_k t A) for each k.
            
            # Optimization: Pre-compute exp(-i t A) and exp(-i t B) is not enough because coefficients a_k differ.
            # However, we can diagonalize A and B once.
            
            # Let's just compute the product directly. 12 steps is small.
            # expm is expensive. 12 * 2 * 3 = 72 expm calls per objective evaluation.
            # 64x64 expm is ~1ms. 72ms per evaluation.
            # Nelder-Mead might take 100-200 evaluations. 72ms * 200 = 14.4s.
            # This is feasible within 50s.
            
            # To speed up, we can diagonalize A and B.
            # A = V_A D_A V_A^dag
            # exp(-i a_k t A) = V_A exp(-i a_k t D_A) V_A^dag
            # This is faster than full expm.
            
            # We'll do this outside the objective loop for efficiency.
            
            U_approx = apply_product_formula(a, b, t, A, B, eig_A, eig_B)
            diff = U_approx - U_exacts[i]
            # Spectral norm is max singular value
            # For unitary matrices, it's max |eigenvalue of diff|? No.
            # np.linalg.norm(diff, 2)
            err = np.linalg.norm(diff, 2)
            error += err
            
        return error / len(t_values) + penalty

    # Diagonalize A and B for faster exponential calculation
    eigvals_A, eigvecs_A = np.linalg.eigh(A)
    eigvals_B, eigvecs_B = np.linalg.eigh(B)
    
    # Helper to apply the product formula
    def apply_product_formula(a, b, t, A, B, eig_A, eig_B):
        # U = prod_{k=1}^{12} exp(-i b_k t B) exp(-i a_k t A)
        # We iterate k=1 to 12
        # U_curr = I
        U_curr = np.eye(64, dtype=complex)
        
        for k in range(12):
            # Step 1: exp(-i b_k t B)
            # B = V_B D_B V_B^dag
            # exp(-i b_k t B) = V_B exp(-i b_k t D_B) V_B^dag
            exp_B = np.exp(-1j * b[k] * t * eigvals_B)
            U_B = (eigvecs_B * exp_B) @ eigvecs_B.conj().T
            U_curr = U_B @ U_curr
            
            # Step 2: exp(-i a_k t A)
            exp_A = np.exp(-1j * a[k] * t * eigvals_A)
            U_A = (eigvecs_A * exp_A) @ eigvecs_A.conj().T
            U_curr = U_A @ U_curr
            
        return U_curr

    # Update objective to use the faster method
    def objective_fast(x):
        if time.time() > deadline:
            return float('inf')
            
        seq = [x[0], x[1], x[2], x[3], x[4], x[5], x[5], x[4], x[3], x[2], x[1], x[0]]
        a = seq
        b = seq
        
        s = sum(x)
        penalty = (s - 0.5)**2 * 1000
        
        error = 0.0
        for i, t in enumerate(t_values):
            U_approx = apply_product_formula(a, b, t, A, B, eigvals_A, eigvecs_A, eigvals_B, eigvecs_B)
            diff = U_approx - U_exacts[i]
            err = np.linalg.norm(diff, 2)
            error += err
            
        return error / len(t_values) + penalty

    # Initial guess: 4th order coefficients distributed?
    # 4th order: c1, c2, c3.
    # We can map them to the 6 parameters.
    # c1 = 1/(2-2^(1/3)) ~ 1.351
    # c2 = -2^(1/3)/(2-2^(1/3)) ~ -0.701
    # c3 = 1 - c1 - c2 ~ 0.349
    # Sum = 1.
    # In our symmetric 6-param scheme, sum(x) = 0.5.
    # So we scale by 0.5?
    # Let's just start with uniform distribution: x_i = 0.5/6 = 1/12.
    x0 = [1.0/12.0] * 6
    
    # Run Nelder-Mead
    try:
        res = opt.minimize(objective_fast, x0, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6})
        x_opt = res.x
    except Exception:
        x_opt = x0
    
    # Construct final a and b
    seq = [x_opt[0], x_opt[1], x_opt[2], x_opt[3], x_opt[4], x_opt[5], x_opt[5], x_opt[4], x_opt[3], x_opt[2], x_opt[1], x_opt[0]]
    a = [float(v) for v in seq]
    b = [float(v) for v in seq]
    
    # Ensure sums are exactly 1 (numerical cleanup)
    sum_a = sum(a)
    if abs(sum_a - 1.0) > 1e-9:
        scale = 1.0 / sum_a
        a = [v * scale for v in a]
        b = [v * scale for v in b]
        
    return a, b

# Note: The function `apply_product_formula` inside `coefficients` needs to access the eigenvectors.
# I will refactor the code to ensure all variables are in scope.

def coefficients(stages):
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for 150s limit
    
    # Build Hamiltonians
    A, B = build_hamiltonians()
    
    # Diagonalize
    eigvals_A, eigvecs_A = np.linalg.eigh(A)
    eigvals_B, eigvecs_B = np.linalg.eigh(B)
    
    # Pre-compute exact propagators using NumPy eigh
    H_total = A + B
    eigvals_H, eigvecs_H = np.linalg.eigh(H_total)
    
    t_values = [0.25, 0.5, 1.0]
    U_exacts = []
    for t in t_values:
        exp_H = np.exp(-1j * t * eigvals_H)
        U_exact = (eigvecs_H * exp_H) @ eigvecs_H.conj().T
        U_exacts.append(U_exact)
    
    # Pre-compute constant matrices for efficient application
    # We'll precompute eigvecs_A.conj().T and eigvecs_B.conj().T
    eigvecs_A_dag = eigvecs_A.conj().T
    eigvecs_B_dag = eigvecs_B.conj().T
    
    def apply_product_formula(a, b, t):
        U_curr = np.eye(64, dtype=complex)
        for k in range(12):
            # B step
            exp_B = np.exp(-1j * b[k] * t * eigvals_B)
            U_B = (eigvecs_B * exp_B) @ eigvecs_B_dag
            U_curr = U_B @ U_curr
            
            # A step
            exp_A = np.exp(-1j * a[k] * t * eigvals_A)
            U_A = (eigvecs_A * exp_A) @ eigvecs_A_dag
            U_curr = U_A @ U_curr
        return U_curr

    def objective(x):
        if time.time() > deadline:
            return float('inf')
            
        a = x[:12]
        b = x[12:]
        
        # Penalty for sum constraints
        sum_a = sum(a)
        sum_b = sum(b)
        penalty = (sum_a - 1.0)**2 * 100 + (sum_b - 1.0)**2 * 100
        
        error = 0.0
        for i, t in enumerate(t_values):
            U_approx = apply_product_formula(a, b, t)
            diff = U_approx - U_exacts[i]
            err = np.linalg.norm(diff, 2)
            error += err
            
        return error / len(t_values) + penalty

    # Simple Nelder-Mead implementation
    def nelder_mead(f, x0, max_iter=500, tol=1e-8):
        n = len(x0)
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        # Initialize simplex
        simplex = [np.array(x0, dtype=float)]
        for i in range(n):
            point = np.array(x0, dtype=float)
            if abs(point[i]) > 1e-10:
                point[i] *= 1 + 0.05
            else:
                point[i] = 0.05
            simplex.append(point)
        
        fvals = [f(p) for p in simplex]
        best_val = min(fvals)
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Sort simplex
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            # Check convergence
            if abs(fvals[0] - fvals[-1]) < tol:
                break
            
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            f_r = f(x_r)
            
            if f_r < fvals[0]:
                # Expansion
                x_e = centroid + gamma * (x_r - centroid)
                f_e = f(x_e)
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
                f_c = f(x_c)
                if f_c < fvals[-1]:
                    simplex[-1] = x_c
                    fvals[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, n+1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = f(simplex[i])
            
            best_val = min(best_val, fvals[0])
        
        return simplex[0]

    # Start with uniform distribution
    x0 = [1.0/12.0] * 24
    
    try:
        x_opt = nelder_mead(objective, x0)
    except Exception:
        x_opt = x0
    
    a = [float(v) for v in x_opt[:12]]
    b = [float(v) for v in x_opt[12:]]
    
    # Ensure sums are exactly 1 (numerical cleanup)
    sum_a = sum(a)
    if abs(sum_a - 1.0) > 1e-9:
        scale = 1.0 / sum_a
        a = [v * scale for v in a]
    
    sum_b = sum(b)
    if abs(sum_b - 1.0) > 1e-9:
        scale = 1.0 / sum_b
        b = [v * scale for v in b]
    
    return a, b
