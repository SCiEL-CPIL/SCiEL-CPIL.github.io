"""Product-formula design for Heisenberg-chain time evolution using numerical optimization."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.linalg import expm

# Constants
DELTA = 0.8
FIELD = 0.3
N_SITES = 6
DIM = 2 ** N_SITES
T_VALS = [0.25, 0.5, 1.0]

# Pauli matrices
Sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
Sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
Sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)

def kron_list(ops):
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result

def build_hamiltonians():
    A = np.zeros((DIM, DIM), dtype=complex)
    B = np.zeros((DIM, DIM), dtype=complex)
    
    # Field term
    H_field = 0
    for j in range(N_SITES):
        ops = [I2] * N_SITES
        ops[j] = Sz
        H_field += FIELD * kron_list(ops)
    
    A += 0.5 * H_field
    B += 0.5 * H_field
    
    # Bond terms
    def add_bond(H, j1, j2):
        ops = [I2] * N_SITES
        ops[j1] = Sx
        ops[j2] = Sx
        H += kron_list(ops)
        
        ops[j1] = Sy
        ops[j2] = Sy
        H += kron_list(ops)
        
        ops[j1] = Sz
        ops[j2] = Sz
        H += DELTA * kron_list(ops)

    # Odd bonds for A
    add_bond(A, 0, 1)
    add_bond(A, 2, 3)
    add_bond(A, 4, 5)
    
    # Even bonds for B
    add_bond(B, 1, 2)
    add_bond(B, 3, 4)
    
    return A, B

A, B = build_hamiltonians()
AB = A + B

# Pre-compute exact propagators
U_exact = [expm(-1j * t * AB) for t in T_VALS]

# Pre-compute eigendecompositions for fast expm
eigvals_A, eigvecs_A = np.linalg.eigh(A)
eigvals_B, eigvecs_B = np.linalg.eigh(B)

def fast_expm(coeff, eigvals, eigvecs):
    exp_diag = np.exp(coeff * eigvals)
    return eigvecs @ np.diag(exp_diag) @ eigvecs.conj().T

def compute_error(a, b):
    total_err = 0.0
    for i, t in enumerate(T_VALS):
        U = np.eye(DIM, dtype=complex)
        for k in range(12):
            # F_k = exp(-i b_k t B) @ exp(-i a_k t A)
            exp_a = fast_expm(-1j * a[k] * t, eigvals_A, eigvecs_A)
            exp_b = fast_expm(-1j * b[k] * t, eigvals_B, eigvecs_B)
            U = (exp_b @ exp_a) @ U
        total_err += np.linalg.norm(U - U_exact[i], 2)
    return total_err / len(T_VALS)

def coefficients(stages):
    start_time = time.time()
    deadline = start_time + 50.0
    
    # Initial guess: Strang splitting
    a0 = [0.5/6] * 12
    b0 = [1/6 if i % 2 == 0 else 0 for i in range(12)]
    
    x0 = np.array(a0 + b0)
    
    cons = (
        {'type': 'eq', 'fun': lambda x: np.sum(x[:12]) - 1},
        {'type': 'eq', 'fun': lambda x: np.sum(x[12:]) - 1},
    )
    bounds = [(-50, 50)] * 24
    
    best_params = x0
    best_val = compute_error(list(x0[:12]), list(x0[12:]))
    
    # Optimization
    try:
        res = minimize(compute_error, x0, method='L-BFGS-B', bounds=bounds, constraints=cons, 
                       options={'maxiter': 300, 'ftol': 1e-9, 'gtol': 1e-9})
        if res.fun < best_val:
            best_val = res.fun
            best_params = res.x
    except Exception:
        pass
    
    if time.time() > deadline:
        a = list(best_params[:12])
        b = list(best_params[12:])
        a = [max(-50, min(50, x)) for x in a]
        b = [max(-50, min(50, x)) for x in b]
        sum_a = sum(a)
        sum_b = sum(b)
        if sum_a != 0: a = [x / sum_a for x in a]
        if sum_b != 0: b = [x / sum_b for x in b]
        return a, b
        
    # Second pass with Nelder-Mead if time allows
    try:
        res2 = minimize(compute_error, best_params, method='Nelder-Mead', bounds=bounds, 
                        options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-9})
        if res2.fun < best_val:
            best_val = res2.fun
            best_params = res2.x
    except Exception:
        pass
        
    a = list(best_params[:12])
    b = list(best_params[12:])
    a = [max(-50, min(50, x)) for x in a]
    b = [max(-50, min(50, x)) for x in b]
    sum_a = sum(a)
    sum_b = sum(b)
    if sum_a != 0: a = [x / sum_a for x in a]
    if sum_b != 0: b = [x / sum_b for x in b]
    
    return a, b

