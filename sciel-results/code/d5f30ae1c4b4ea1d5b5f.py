"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def coefficients(stages):
    """
    12-stage product formula using Strang splitting structure with optimized coefficients.
    
    Structure: B,A,B,A,B,A,B,A,B,A,B,A (12 stages)
    """
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Start with Strang splitting (inspiration 1 structure)
    substeps = stages // 2  # 6 substeps
    a = [0.5 / substeps] * stages  # each a_k = 0.5/6 = 1/12
    b = []
    for _ in range(substeps):
        b += [1.0 / substeps, 0.0]  # b = [1/6, 0, 1/6, 0, 1/6, 0, 1/6, 0, 1/6, 0, 1/6, 0]
    
    # Verify consistency
    assert abs(sum(a) - 1.0) < 1e-10
    assert abs(sum(b) - 1.0) < 1e-10
    
    # Try to build and optimize if time permits
    if time.time() < deadline - 30:
        try:
            # Build Hamiltonians
            N = 6
            dim = 2 ** N
            delta = 0.8
            field = 0.3
            
            sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
            sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
            sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
            
            def tensor_op(op, site):
                result = np.array([[1.0]], dtype=complex)
                for i in range(N):
                    if i == site:
                        result = np.kron(result, op)
                    else:
                        result = np.kron(result, np.eye(2, dtype=complex))
                return result
            
            A = np.zeros((dim, dim), dtype=complex)
            B = np.zeros((dim, dim), dtype=complex)
            
            def bond_term(j, j1):
                sz_j = tensor_op(sz, j)
                sz_j1 = tensor_op(sz, j1)
                sx_j = tensor_op(sx, j)
                sx_j1 = tensor_op(sx, j1)
                sy_j = tensor_op(sy, j)
                sy_j1 = tensor_op(sy, j1)
                return (delta/4.0) * (sz_j @ sz_j1) + (1/4.0) * (sx_j @ sx_j1 + sy_j @ sy_j1)
            
            def field_term(site, coeff):
                return coeff * tensor_op(sz, site)
            
            # Odd bonds: (0,1), (2,3), (4,5)
            for j, j1 in [(0, 1), (2, 3), (4, 5)]:
                A += bond_term(j, j1)
            
            # Even bonds: (1,2), (3,4)
            for j, j1 in [(1, 2), (3, 4)]:
                B += bond_term(j, j1)
            
            # Field: split equally
            for site in range(N):
                A += field_term(site, field / 4.0)
                B += field_term(site, field / 4.0)
            
            # Precompute exact propagator for the 3 time points
            times = [0.25, 0.5, 1.0]
            U_exacts = [np.linalg.matrix_exp(-1j * t * (A + B)) for t in times]
            
            def compute_error_full(a_coeffs, b_coeffs):
                """Compute mean spectral norm error at target times"""
                errors = []
                for t, U_exact in zip(times, U_exacts):
                    U = np.eye(dim, dtype=complex)
                    for k in range(len(a_coeffs)):
                        # exp(-i b_k t B)
                        U = np.linalg.multi_dot([np.linalg.matrix_exp(-1j * b_coeffs[k] * t * B), U])
                        # exp(-i a_k t A)
                        U = np.linalg.multi_dot([np.linalg.matrix_exp(-1j * a_coeffs[k] * t * A), U])
                    errors.append(np.linalg.norm(U - U_exact, 2))
                return np.mean(errors)

            # Objective function for optimizer
            # We optimize 22 variables: a[0:11] and b[0:11]
            # a[11] = 1 - sum(a[0:11])
            # b[11] = 1 - sum(b[0:11])
            
            def objective(params):
                a_opt = list(params[:11])
                b_opt = list(params[11:22])
                
                # Check bounds loosely to avoid overflow in exp, though 50 is safe
                # Calculate last coefficients
                sum_a = sum(a_opt)
                sum_b = sum(b_opt)
                a_full = a_opt + [1.0 - sum_a]
                b_full = b_opt + [1.0 - sum_b]
                
                # Check if any coefficient is wildly out of bounds (sanity)
                if any(abs(x) > 50 for x in a_full + b_full):
                    return 1e6
                
                return compute_error_full(a_full, b_full)
            
            # Initial guess: Strang splitting
            x0 = a[:11] + b[:11]
            
            # Run Nelder-Mead optimization
            # maxiter should be tuned to fit in time
            # Estimate: 1 eval ~ 50-100ms. 1000 evals ~ 50-100s.
            # We have ~110s available for optimization.
            # Let's try 500 iterations with Nelder-Mead.
            
            start_time = time.time()
            result = minimize(objective, x0, method='Nelder-Mead', 
                              options={'maxiter': 800, 'xatol': 1e-8, 'fatol': 1e-10, 'disp': False})
            
            elapsed = time.time() - start_time
            # print(f"Optimization took {elapsed}s")
            
            if result.success or result.fun < compute_error_full(a, b):
                # Reconstruct a and b
                best_params = result.x
                a = list(best_params[:11])
                b = list(best_params[11:22])
                
                a.append(1.0 - sum(a))
                b.append(1.0 - sum(b))
                
                # Clamp to [-50, 50]
                a = [max(-50, min(50, x)) for x in a]
                b = [max(-50, min(50, x)) for x in b]
                
                # Final consistency adjustment
                if abs(sum(a) - 1.0) > 1e-8:
                    a[-1] += 1.0 - sum(a)
                if abs(sum(b) - 1.0) > 1e-8:
                    b[-1] += 1.0 - sum(b)
            
        except Exception:
            # Fall back to Strang splitting
            pass
    
    # Final safety check
    a = [max(-50, min(50, x)) for x in a]
    b = [max(-50, min(50, x)) for x in b]
    
    # Ensure exact consistency
    if abs(sum(a) - 1.0) > 1e-10:
        a[-1] += 1.0 - sum(a)
    if abs(sum(b) - 1.0) > 1e-10:
        b[-1] += 1.0 - sum(b)
    
    return a, b
# EVOLVE-BLOCK-END
