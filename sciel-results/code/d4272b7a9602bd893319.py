import numpy as np
import scipy.linalg
import scipy.optimize
import time

def coefficients(stages):
    start_time = time.time()
    deadline = start_time + 140
    
    # Build Hamiltonians
    Sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    Sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
    Sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    
    def kron_op(op, pos, n_sites=6):
        ops = [np.eye(2)] * n_sites
        ops[pos] = op
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result

    A = np.zeros((64, 64), dtype=complex)
    B = np.zeros((64, 64), dtype=complex)
    
    Delta = 0.8
    h = 0.3
    
    for i in range(5):
        bond = (kron_op(Sx, i) @ kron_op(Sx, i+1) + 
                kron_op(Sy, i) @ kron_op(Sy, i+1) + 
                Delta * kron_op(Sz, i) @ kron_op(Sz, i+1))
        if i % 2 == 1:
            A += bond
        else:
            B += bond
            
    field = np.zeros((64, 64), dtype=complex)
    for i in range(6):
        field += kron_op(Sz, i)
        
    A += 0.5 * h * field
    B += 0.5 * h * field
    
    H_total = A + B
    
    # Diagonalize
    eigvals_A, eigvecs_A = np.linalg.eigh(A)
    eigvals_B, eigvecs_B = np.linalg.eigh(B)
    
    # Precompute U_exact
    U_exact_025 = scipy.linalg.expm(-1j * 0.25 * H_total)
    U_exact_05 = scipy.linalg.expm(-1j * 0.5 * H_total)
    U_exact_10 = scipy.linalg.expm(-1j * 1.0 * H_total)
    
    def error_func(params):
        if time.time() > deadline:
            return 1e9
        a = params[:12]
        b = params[12:]
        
        if abs(np.sum(a) - 1) > 1e-6 or abs(np.sum(b) - 1) > 1e-6:
            return 1e9
        if np.any(a < -50) or np.any(a > 50) or np.any(b < -50) or np.any(b > 50):
            return 1e9
            
        err = 0.0
        for t, U_exact in zip([0.25, 0.5, 1.0], [U_exact_025, U_exact_05, U_exact_10]):
            U_approx = np.eye(64, dtype=complex)
            for k in range(12):
                # exp(-i b_k t B)
                if b[k] != 0:
                    exp_B = eigvecs_B @ np.diag(np.exp(-1j * b[k] * t * eigvals_B)) @ np.linalg.inv(eigvecs_B)
                    U_approx = exp_B @ U_approx
                # exp(-i a_k t A)
                if a[k] != 0:
                    exp_A = eigvecs_A @ np.diag(np.exp(-1j * a[k] * t * eigvals_A)) @ np.linalg.inv(eigvecs_A)
                    U_approx = exp_A @ U_approx
            err += np.linalg.norm(U_approx - U_exact, 2)
        return err / 3.0

    # Initial guess: Strang splitting (4 substeps)
    a_init = np.zeros(12)
    b_init = np.zeros(12)
    for j in range(4):
        base = j * 3
        a_init[base] = 0.125
        b_init[base] = 0.0
        a_init[base+1] = 0.0
        b_init[base+1] = 0.25
        a_init[base+2] = 0.125
        b_init[base+2] = 0.0
        
    params_init = np.concatenate([a_init, b_init])
    
    best_params = params_init
    best_err = error_func(params_init)
    
    try:
        res = scipy.optimize.minimize(
            error_func, 
            params_init, 
            method='SLSQP',
            options={'maxiter': 100, 'ftol': 1e-8}
        )
        if res.fun < best_err:
            best_params = res.x
            best_err = res.fun
            
    except Exception:
        pass
        
    a_best = best_params[:12].tolist()
    b_best = best_params[12:].tolist()
    
    return a_best, b_best
