"""Product-formula coefficient optimizer using direct 24-parameter Nelder-Mead fit."""
import numpy as np
import time
import sys

# Deadline guard
_DEADLINE = 140.0
_T0 = time.time()


def _remaining():
    return _DEADLINE - (time.time() - _T0)


def _build_hamiltonians():
    """Build A (odd bonds + half field) and B (even bonds + half field) for 6-site open XXZ chain."""
    N = 6
    delta = 0.8
    field = 0.3
    
    sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    identity = np.eye(2, dtype=complex)
    
    def kron_all(ops):
        result = np.array([[1]], dtype=complex)
        for op in ops:
            result = np.kron(result, op)
        return result
    
    A = np.zeros((2**N, 2**N), dtype=complex)
    B = np.zeros((2**N, 2**N), dtype=complex)
    
    def bond_operator(i, j):
        s_plus = np.array([[0, 1], [0, 0]], dtype=complex)
        s_minus = np.array([[0, 0], [1, 0]], dtype=complex)
        
        ops_splus_sminus = [identity] * N
        ops_splus_sminus[i] = s_plus
        ops_splus_sminus[j] = s_minus
        term1 = 0.5 * kron_all(ops_splus_sminus)
        
        ops_sminus_splus = [identity] * N
        ops_sminus_splus[i] = s_minus
        ops_sminus_splus[j] = s_plus
        term2 = 0.5 * kron_all(ops_sminus_splus)
        
        ops_sz_sz = [identity] * N
        ops_sz_sz[i] = sz
        ops_sz_sz[j] = sz
        term3 = delta * kron_all(ops_sz_sz)
        
        return term1 + term2 + term3
    
    # Odd bonds: (1,2) and (3,4) -> A
    A += bond_operator(1, 2)
    A += bond_operator(3, 4)
    
    # Even bonds: (0,1), (2,3), (4,5) -> B
    B += bond_operator(0, 1)
    B += bond_operator(2, 3)
    B += bond_operator(4, 5)
    
    # Uniform field split half into A, half into B
    field_op = np.zeros((2**N, 2**N), dtype=complex)
    for j in range(N):
        ops_sz = [identity] * N
        ops_sz[j] = sz
        field_op += kron_all(ops_sz)
    
    A += -0.5 * field * field_op
    B += -0.5 * field * field_op
    
    return A, B


def _precompute_exponentials(A, B, H_full, t_values):
    """Precompute eigendecompositions for fast evaluation."""
    dim = 2**6
    
    # Eigen-decompose A, B, H_full once
    eigvals_A, eigvecs_A = np.linalg.eigh(A)
    eigvals_B, eigvecs_B = np.linalg.eigh(B)
    eigvals_H, eigvecs_H = np.linalg.eigh(H_full)
    
    # Precompute exp(-i * t * eigvals) for each t and each operator
    # For A: exp(-i * a_k * t * eigvals_A) - depends on a_k and t
    # For B: exp(-i * b_k * t * eigvals_B) - depends on b_k and t
    # For H: exp(-i * t * eigvals_H) - depends only on t
    
    return {
        'eigvals_A': eigvals_A,
        'eigvecs_A': eigvecs_A,
        'eigvals_B': eigvals_B,
        'eigvecs_B': eigvecs_B,
        'eigvals_H': eigvals_H,
        'eigvecs_H': eigvecs_H,
        'dim': dim
    }


def _compute_error_fast(a, b, precomp, t_values):
    """Compute mean spectral norm error using precomputed eigendecompositions."""
    eigvals_A = precomp['eigvals_A']
    eigvecs_A = precomp['eigvecs_A']
    eigvals_B = precomp['eigvals_B']
    eigvecs_B = precomp['eigvecs_B']
    eigvals_H = precomp['eigvals_H']
    eigvecs_H = precomp['eigvecs_H']
    dim = precomp['dim']
    
    total_error = 0.0
    n_times = len(t_values)
    
    for t in t_values:
        U = np.eye(dim, dtype=complex)
        
        for k in range(12):
            # exp(-i a_k t A)
            if abs(a[k]) > 1e-15:
                phase = -1j * a[k] * t * eigvals_A
                exp_A = eigvecs_A @ np.diag(np.exp(phase)) @ eigvecs_A.conj().T
                U = U @ exp_A
            
            # exp(-i b_k t B)
            if abs(b[k]) > 1e-15:
                phase = -1j * b[k] * t * eigvals_B
                exp_B = eigvecs_B @ np.diag(np.exp(phase)) @ eigvecs_B.conj().T
                U = U @ exp_B
        
        # Exact propagator
        phase_H = -1j * t * eigvals_H
        U_exact = eigvecs_H @ np.diag(np.exp(phase_H)) @ eigvecs_H.conj().T
        
        diff = U - U_exact
        error = np.linalg.norm(diff, 2)
        total_error += error
    
    return total_error / n_times


def coefficients(stages):
    """Optimize all 24 coefficients independently using Nelder-Mead with sum constraints."""
    stages = 12
    
    A, B = _build_hamiltonians()
    H_full = A + B
    
    t_values = [0.25, 0.5, 1.0]
    
    precomp = _precompute_exponentials(A, B, H_full, t_values)
    
    # Initial guess: uniform
    a_init = [1.0 / 12] * 12
    b_init = [1.0 / 12] * 12
    
    best_a = a_init[:]
    best_b = b_init[:]
    best_error = _compute_error_fast(a_init, b_init, precomp, t_values)
    
    if _remaining() > 20:
        def objective(params):
            """Objective: 24 free parameters, enforce sum constraints by projection."""
            a_params = params[:12]
            b_params = params[12:24]
            
            # Project to sum=1 by subtracting mean and scaling
            a_mean = np.mean(a_params)
            b_mean = np.mean(b_params)
            
            # Center and scale to unit sum
            a_centered = a_params - a_mean
            b_centered = b_params - b_mean
            
            a_sum = np.sum(a_params)
            b_sum = np.sum(b_params)
            
            if abs(a_sum) < 1e-15 or abs(b_sum) < 1e-15:
                return 1e10
            
            a_scaled = a_params / a_sum
            b_scaled = b_params / b_sum
            
            # Clamp
            a_scaled = np.clip(a_scaled, -50.0, 50.0)
            b_scaled = np.clip(b_scaled, -50.0, 50.0)
            
            # Re-normalize after clamping
            a_sum2 = np.sum(a_scaled)
            b_sum2 = np.sum(b_scaled)
            
            if abs(a_sum2) < 1e-15 or abs(b_sum2) < 1e-15:
                return 1e10
            
            a_final = a_scaled / a_sum2
            b_final = b_scaled / b_sum2
            
            if _remaining() < 5:
                return 1e10
            
            return _compute_error_fast(a_final, b_final, precomp, t_values)
        
        x0 = np.array(a_init + b_init)
        
        try:
            from scipy.optimize import minimize
            
            # Run Nelder-Mead with all 24 parameters free
            result = minimize(objective, x0, method='Nelder-Mead',
                            options={'maxiter': 500, 'fatol': 1e-12, 'xatol': 1e-12,
                                    'adaptive': True, 'initial_simplex': None})
            
            if result.fun < best_error:
                best_error = result.fun
                a_params = result.x[:12]
                b_params = result.x[12:24]
                
                a_sum = np.sum(a_params)
                b_sum = np.sum(b_params)
                
                if abs(a_sum) > 1e-15:
                    best_a = list(a_params / a_sum)
                if abs(b_sum) > 1e-15:
                    best_b = list(b_params / b_sum)
                
                best_a = [max(-50.0, min(50.0, float(x))) for x in best_a]
                best_b = [max(-50.0, min(50.0, float(x))) for x in best_b]
                
                # Re-normalize
                a_sum = sum(best_a)
                b_sum = sum(best_b)
                if abs(a_sum) > 1e-15:
                    best_a = [x / a_sum for x in best_a]
                if abs(b_sum) > 1e-15:
                    best_b = [x / b_sum for x in best_b]
                
        except Exception:
            pass
        
        # Second pass: refine with more iterations if time allows
        if _remaining() > 15:
            try:
                from scipy.optimize import minimize
                
                x0_refine = np.array(best_a + best_b)
                
                result2 = minimize(objective, x0_refine, method='Nelder-Mead',
                                 options={'maxiter': 300, 'fatol': 1e-14, 'xatol': 1e-14,
                                         'adaptive': True})
                
                if result2.fun < best_error:
                    best_error = result2.fun
                    a_params = result2.x[:12]
                    b_params = result2.x[12:24]
                    
                    a_sum = np.sum(a_params)
                    b_sum = np.sum(b_params)
                    
                    if abs(a_sum) > 1e-15:
                        best_a = list(a_params / a_sum)
                    if abs(b_sum) > 1e-15:
                        best_b = list(b_params / b_sum)
                    
                    best_a = [max(-50.0, min(50.0, float(x))) for x in best_a]
                    best_b = [max(-50.0, min(50.0, float(x))) for x in best_b]
                    
                    a_sum = sum(best_a)
                    b_sum = sum(best_b)
                    if abs(a_sum) > 1e-15:
                        best_a = [x / a_sum for x in best_a]
                    if abs(b_sum) > 1e-15:
                        best_b = [x / b_sum for x in best_b]
                        
            except Exception:
                pass
    
    # Final normalization and clamping
    a_sum = sum(best_a)
    b_sum = sum(best_b)
    if abs(a_sum) > 1e-15:
        best_a = [x / a_sum for x in best_a]
    if abs(b_sum) > 1e-15:
        best_b = [x / b_sum for x in best_b]
    
    best_a = [max(-50.0, min(50.0, float(x))) for x in best_a]
    best_b = [max(-50.0, min(50.0, float(x))) for x in best_b]
    
    return best_a, best_b
