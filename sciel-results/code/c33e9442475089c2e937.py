"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def coefficients(stages):
    """
    Optimize product formula coefficients using a Fourier-series parameterization.
    
    Uses a truncated Fourier representation to enforce smoothness and symmetry,
    then optimizes the few low-frequency parameters via Nelder-Mead.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    if stages != 12:
        # Fallback for non-12 stage cases
        substeps = max(1, stages // 2)
        a = [0.5 / substeps] * (2 * substeps)
        b = []
        for _ in range(substeps):
            b += [1.0 / substeps, 0.0]
        while len(a) < stages:
            a.append(0.0)
            b.append(0.0)
        return a, b
    
    # Build the Hamiltonian matrices
    anisotropy = 0.8
    field = 0.3
    N = 6
    dim = 2 ** N
    
    # Pauli matrices
    I2 = np.eye(2)
    Sx = 0.5 * np.array([[0, 1], [1, 0]])
    Sy = 0.5 * np.array([[0, -1j], [1j, 0]])
    Sz = 0.5 * np.array([[1, 0], [0, -1]])
    
    def kron_all(*ops):
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result
    
    def make_op(site, op):
        ops = [I2] * N
        ops[site] = op
        return kron_all(*ops)
    
    # Build A (even bonds + half field) and B (odd bonds + half field)
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    # Even bonds: (0,1), (2,3), (4,5)
    for i in [0, 2, 4]:
        Sx_i = make_op(i, Sx)
        Sx_ip1 = make_op(i+1, Sx)
        Sy_i = make_op(i, Sy)
        Sy_ip1 = make_op(i+1, Sy)
        Sz_i = make_op(i, Sz)
        Sz_ip1 = make_op(i+1, Sz)
        A += Sx_i @ Sx_ip1 + Sy_i @ Sy_ip1 + anisotropy * Sz_i @ Sz_ip1
    
    # Odd bonds: (1,2), (3,4)
    for i in [1, 3]:
        Sx_i = make_op(i, Sx)
        Sx_ip1 = make_op(i+1, Sx)
        Sy_i = make_op(i, Sy)
        Sy_ip1 = make_op(i+1, Sy)
        Sz_i = make_op(i, Sz)
        Sz_ip1 = make_op(i+1, Sz)
        B += Sx_i @ Sx_ip1 + Sy_i @ Sy_ip1 + anisotropy * Sz_i @ Sz_ip1
    
    # Add half field to each
    field_half = field / 2.0
    for i in range(N):
        Sz_i = make_op(i, Sz)
        A += field_half * Sz_i
        B += field_half * Sz_i
    
    H = A + B
    
    # Precompute eigen-decompositions for fast exponentiation
    eigvals_A, eigvecs_A = np.linalg.eig(A)
    eigvals_B, eigvecs_B = np.linalg.eig(B)
    eigvals_H, eigvecs_H = np.linalg.eig(H)
    
    def expm_fast(eigvals, eigvecs, theta):
        """Compute exp(theta * M) given eigendecomposition M = V diag(eigvals) V^{-1}"""
        # exp(theta * M) = V diag(exp(theta * eigvals)) V^{-1}
        exp_eigvals = np.exp(theta * eigvals)
        V_inv = np.linalg.inv(eigvecs)
        return eigvecs @ np.diag(exp_eigvals) @ V_inv
    
    def compute_U(a_coeffs, b_coeffs, t):
        """Compute product formula U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A)"""
        U = np.eye(dim, dtype=complex)
        for k in range(12):
            U = expm_fast(eigvals_B, eigvecs_B, -1j * b_coeffs[k] * t) @ U
            U = expm_fast(eigvals_A, eigvecs_A, -1j * a_coeffs[k] * t) @ U
        return U
    
    def compute_U_exact(t):
        """Compute exact exp(-i t (A+B))"""
        return expm_fast(eigvals_H, eigvecs_H, -1j * t)
    
    # Precompute exact propagators
    times = [0.25, 0.5, 1.0]
    U_exact_list = [compute_U_exact(t) for t in times]
    
    # Fourier-series parameterization
    # a_k = 1/12 + sum_{m=1}^{M} [alpha_m cos(2*pi*m*k/12) + beta_m sin(2*pi*m*k/12)]
    # b_k = 1/12 + sum_{m=1}^{M} [gamma_m cos(2*pi*m*k/12) + delta_m sin(2*pi*m*k/12)]
    # This enforces sum(a) = sum(b) = 1 automatically.
    # For symmetry a_k = a_{11-k}, we need beta_m = 0 for all m (sine terms vanish).
    # Actually, for a_k = a_{11-k} (palindromic symmetry), let's think:
    # k ranges 0..11. Symmetry: a[k] = a[11-k].
    # In Fourier: a[k] = a[11-k] means the sequence is symmetric about k=5.5.
    # This corresponds to cosine terms only (no sine terms) when indexed properly.
    
    # Let's use: a_k = 1/12 + sum_{m=1}^{M} alpha_m cos(2*pi*m*(k-5.5)/12)
    # and similarly for b_k with gamma_m.
    # This automatically gives a_k = a_{11-k} and sum = 1.
    
    M = 3  # Number of Fourier modes
    # Parameters: alpha_1, alpha_2, alpha_3, gamma_1, gamma_2, gamma_3 (6 params)
    
    def fourier_to_coeffs(params):
        """Convert Fourier params to coefficient lists"""
        alphas = params[:M]
        gammas = params[M:2*M]
        
        a = np.zeros(12)
        b = np.zeros(12)
        for k in range(12):
            a[k] = 1.0/12.0
            b[k] = 1.0/12.0
            for m in range(1, M+1):
                angle = 2 * np.pi * m * (k - 5.5) / 12.0
                a[k] += alphas[m-1] * np.cos(angle)
                b[k] += gammas[m-1] * np.cos(angle)
        
        return a, b
    
    def objective(params):
        if time.time() > deadline:
            return np.inf
        
        a_coeffs, b_coeffs = fourier_to_coeffs(params)
        
        # Check bounds
        if np.any(a_coeffs < -50) or np.any(a_coeffs > 50) or np.any(b_coeffs < -50) or np.any(b_coeffs > 50):
            return 1e10
        
        # Compute error
        errors = []
        for i, t in enumerate(times):
            U = compute_U(a_coeffs, b_coeffs, t)
            diff = U - U_exact_list[i]
            sv = np.linalg.svd(diff, compute_uv=False)
            errors.append(sv[0])
        
        return np.mean(errors)
    
    # Initial guess: small Fourier coefficients
    x0 = np.zeros(2 * M)
    
    # Optimize using Nelder-Mead
    from scipy.optimize import minimize
    
    best_result = None
    best_score = np.inf
    
    # Try multiple initial points
    initial_points = [
        np.zeros(2*M),
        np.array([0.1, 0.05, 0.02, 0.1, 0.05, 0.02]),
        np.array([-0.1, 0.05, -0.02, -0.1, 0.05, -0.02]),
        np.array([0.05, -0.03, 0.01, 0.05, -0.03, 0.01]),
    ]
    
    for x0 in initial_points:
        if time.time() > deadline - 20:
            break
        
        try:
            result = minimize(objective, x0, method='Nelder-Mead',
                            options={'maxiter': 200, 'fatol': 1e-10, 'xatol': 1e-10})
            
            if result.fun < best_score:
                best_score = result.fun
                best_result = result
        except Exception:
            continue
    
    # If optimization didn't find anything good, use a simple symmetric scheme
    if best_result is None or best_score > 0.5:
        # Use a known good symmetric splitting
        # For 12 stages, use a pattern like:
        # a = [x, y, y, y, y, x, x, y, y, y, y, x] with 2x + 4y = 1
        # b similar
        x_val = 0.05
        y_val = (1 - 2*x_val) / 4.0
        a = [x_val, y_val, y_val, y_val, y_val, x_val,
             x_val, y_val, y_val, y_val, y_val, x_val]
        b = [x_val, y_val, y_val, y_val, y_val, x_val,
             x_val, y_val, y_val, y_val, y_val, x_val]
    else:
        a_coeffs, b_coeffs = fourier_to_coeffs(best_result.x)
        
        # Clip to valid range
        a_coeffs = np.clip(a_coeffs, -50, 50)
        b_coeffs = np.clip(b_coeffs, -50, 50)
        
        # Re-normalize to ensure sum = 1
        sum_a = np.sum(a_coeffs)
        sum_b = np.sum(b_coeffs)
        if abs(sum_a) > 1e-10:
            a_coeffs = a_coeffs / sum_a
        if abs(sum_b) > 1e-10:
            b_coeffs = b_coeffs / sum_b
        
        a = list(a_coeffs)
        b = list(b_coeffs)
    
    # Final check
    if time.time() > deadline:
        return a, b
    
    return a, b
# EVOLVE-BLOCK-END
