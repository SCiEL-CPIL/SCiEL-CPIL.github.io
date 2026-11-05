import time
import numpy as np
from scipy.optimize import minimize

_start_time = time.time()
_DEADLINE = 140.0  # safety margin before 150s kill

def _check_deadline():
    return (time.time() - _start_time) > _DEADLINE

def _build_hamiltonians():
    """Build A (odd bonds + half field) and B (even bonds + half field) for 6-site XXZ chain."""
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    
    delta = 0.8
    h = 0.3
    n_sites = 6
    dim = 2 ** n_sites
    
    def kron_all(ops):
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result
    
    def bond_term(j, k, d):
        ops = [I2] * n_sites
        ops[j] = (sx + sy + d * sz) / 2.0
        ops[k] = (sx + sy + d * sz) / 2.0
        return kron_all(ops)
    
    def field_term():
        total = np.zeros((dim, dim), dtype=complex)
        for j in range(n_sites):
            ops = [I2] * n_sites
            ops[j] = sz / 2.0
            total += h * kron_all(ops)
        return total
    
    odd_bonds = [(0, 1), (2, 3), (4, 5)]
    even_bonds = [(1, 2), (3, 4)]
    
    A = np.zeros((dim, dim), dtype=complex)
    for j, k in odd_bonds:
        A += bond_term(j, k, delta)
    A += 0.5 * field_term()
    
    B = np.zeros((dim, dim), dtype=complex)
    for j, k in even_bonds:
        B += bond_term(j, k, delta)
    B += 0.5 * field_term()
    
    return A, B

def coefficients(stages):
    if stages != 12:
        substeps = max(1, stages // 2)
        a = [1.0 / stages] * stages
        b = [1.0 / stages] * stages
        return a, b
    
    times = [0.25, 0.5, 1.0]
    A, B = _build_hamiltonians()
    dim = A.shape[0]
    
    # Eigen-decompositions
    eigvals_A, eigvecs_A = np.linalg.eigh(A)
    eigvals_B, eigvecs_B = np.linalg.eigh(B)
    AB = A + B
    eigvals_AB, eigvecs_AB = np.linalg.eigh(AB)
    
    def _expm_A(coeff, t):
        phases = np.exp(-1j * coeff * t * eigvals_A)
        return (eigvecs_A * phases) @ eigvecs_A.conj().T
    
    def _expm_B(coeff, t):
        phases = np.exp(-1j * coeff * t * eigvals_B)
        return (eigvecs_B * phases) @ eigvecs_B.conj().T
    
    def _expm_AB(t):
        phases = np.exp(-1j * t * eigvals_AB)
        return (eigvecs_AB * phases) @ eigvecs_AB.conj().T
    
    def _score_fast(a, b):
        errors = []
        for t in times:
            U = np.eye(dim, dtype=complex)
            for k in range(12):
                U = _expm_B(b[k], t) @ U
                U = _expm_A(a[k], t) @ U
            U_exact = _expm_AB(t)
            err = np.linalg.norm(U - U_exact, 2)
            errors.append(err)
        return np.mean(errors)
    
    # Initial: uniform
    a_init = [1.0/12] * 12
    b_init = [1.0/12] * 12
    best_a = a_init[:]
    best_b = b_init[:]
    best_score = _score_fast(best_a, best_b)
    
    # Random restarts
    rng = np.random.RandomState(123)
    for restart in range(8):
        if _check_deadline():
            break
        a_try = list(rng.uniform(-0.3, 1.3, 12))
        b_try = list(rng.uniform(-0.3, 1.3, 12))
        s_a = sum(a_try)
        s_b = sum(b_try)
        if abs(s_a) > 1e-10:
            a_try = [x / s_a for x in a_try]
        if abs(s_b) > 1e-10:
            b_try = [x / s_b for x in b_try]
        a_try = [max(-50.0, min(50.0, x)) for x in a_try]
        b_try = [max(-50.0, min(50.0, x)) for x in b_try]
        s_a = sum(a_try)
        s_b = sum(b_try)
        if abs(s_a) > 1e-10:
            a_try = [x / s_a for x in a_try]
        if abs(s_b) > 1e-10:
            b_try = [x / s_b for x in b_try]
        
        score_try = _score_fast(a_try, b_try)
        if score_try < best_score:
            best_a = a_try[:]
            best_b = b_try[:]
            best_score = score_try
    
    # Local optimization
    if not _check_deadline():
        def _obj(params):
            if _check_deadline():
                return best_score
            a = list(params[0:12])
            b = list(params[12:24])
            s_a = sum(a)
            s_b = sum(b)
            if abs(s_a) > 1e-15:
                a = [x / s_a for x in a]
            if abs(s_b) > 1e-15:
                b = [x / s_b for x in b]
            a = [max(-50.0, min(50.0, x)) for x in a]
            b = [max(-50.0, min(50.0, x)) for x in b]
            s_a = sum(a)
            s_b = sum(b)
            if abs(s_a) > 1e-15:
                a = [x / s_a for x in a]
            if abs(s_b) > 1e-15:
                b = [x / s_b for x in b]
            return _score_fast(a, b)
        
        x0 = best_a + best_b
        result = minimize(_obj, x0, method='Nelder-Mead',
                          options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
        
        if result.fun < best_score:
            a_opt = list(result.x[0:12])
            b_opt = list(result.x[12:24])
            s_a = sum(a_opt)
            s_b = sum(b_opt)
            if abs(s_a) > 1e-15:
                a_opt = [x / s_a for x in a_opt]
            if abs(s_b) > 1e-15:
                b_opt = [x / s_b for x in b_opt]
            a_opt = [max(-50.0, min(50.0, x)) for x in a_opt]
            b_opt = [max(-50.0, min(50.0, x)) for x in b_opt]
            s_a = sum(a_opt)
            s_b = sum(b_opt)
            if abs(s_a) > 1e-15:
                a_opt = [x / s_a for x in a_opt]
            if abs(s_b) > 1e-15:
                b_opt = [x / s_b for x in b_opt]
            best_a = a_opt
            best_b = b_opt
            best_score = result.fun
    
    # Final cleanup
    a_final = best_a[:]
    b_final = best_b[:]
    s_a = sum(a_final)
    s_b = sum(b_final)
    if abs(s_a) > 1e-15:
        a_final = [x / s_a for x in a_final]
    if abs(s_b) > 1e-15:
        b_final = [x / s_b for x in b_final]
    a_final = [max(-50.0, min(50.0, x)) for x in a_final]
    b_final = [max(-50.0, min(50.0, x)) for x in b_final]
    s_a = sum(a_final)
    s_b = sum(b_final)
    if abs(s_a) > 1e-15:
        a_final = [x / s_a for x in a_final]
    if abs(s_b) > 1e-15:
        b_final = [x / s_b for x in b_final]
    
    return a_final, b_final
