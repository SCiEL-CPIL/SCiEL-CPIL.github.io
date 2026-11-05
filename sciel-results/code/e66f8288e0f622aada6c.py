import time
import numpy as np
from scipy import optimize
from scipy.linalg import expm, eigh

_START = time.time()
_DEADLINE = _START + 48.0

def _build_hamiltonian():
    n = 6
    dim = 2 ** n
    Jz = 1.0
    Jperp = 0.8
    h = 0.3
    
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    sp = (sx + 1j*sy) / 2
    sm = (sx - 1j*sy) / 2
    
    def site_op(op, site):
        ops = [I2] * n
        ops[site] = op
        result = ops[0]
        for o in ops[1:]:
            result = np.kron(result, o)
        return result
    
    def bond_op(op1, op2, site1, site2):
        ops = [I2] * n
        ops[site1] = op1
        ops[site2] = op2
        result = ops[0]
        for o in ops[1:]:
            result = np.kron(result, o)
        return result
    
    A = np.zeros((dim, dim), dtype=complex)
    for s1, s2 in [(0,1), (2,3), (4,5)]:
        A += Jz * bond_op(sz, sz, s1, s2)
        A += Jperp/2 * (bond_op(sp, sm, s1, s2) + bond_op(sm, sp, s1, s2))
    
    B = np.zeros((dim, dim), dtype=complex)
    for s1, s2 in [(1,2), (3,4)]:
        B += Jz * bond_op(sz, sz, s1, s2)
        B += Jperp/2 * (bond_op(sp, sm, s1, s2) + bond_op(sm, sp, s1, s2))
    
    field = np.zeros((dim, dim), dtype=complex)
    for s in range(n):
        field += site_op(sz, s)
    A += (h/2) * field
    B += (h/2) * field
    
    return A, B

def coefficients(stages):
    t0 = time.time()
    
    if stages != 12:
        substeps = max(1, stages // 2)
        a = [0.5 / substeps] * (2 * substeps)
        b = []
        for _ in range(substeps):
            b += [1.0 / substeps, 0.0]
        while len(a) < stages:
            a.append(0.0)
            b.append(0.0)
        return a, b
    
    A, B = _build_hamiltonian()
    times = [0.25, 0.5, 1.0]
    dim = 2**6
    
    # Precompute eigen-decompositions (Hermitian)
    wA, vA = eigh(A)
    wB, vB = eigh(B)
    vA_inv = np.linalg.inv(vA)
    vB_inv = np.linalg.inv(vB)
    
    # Precompute exact propagators
    exact = {}
    for t in times:
        exact[t] = expm(-1j * t * (A + B))
    
    def expm_A_fast(c):
        """exp(-i*c*A)"""
        return vA @ (np.exp(-1j * c * wA)[:, None] * vA_inv)
    
    def expm_B_fast(c):
        """exp(-i*c*B)"""
        return vB @ (np.exp(-1j * c * wB)[:, None] * vB_inv)
    
    def _objective(a, b):
        error = 0.0
        for t in times:
            U = np.eye(dim, dtype=complex)
            for k in range(12):
                U = expm_B_fast(b[k] * t) @ U
                U = expm_A_fast(a[k] * t) @ U
            err = np.linalg.norm(U - exact[t], 2)
            error += err
        return error / len(times)
    
    a0 = [1.0/12] * 12
    b0 = [1.0/12] * 12
    best_a = a0[:]
    best_b = b0[:]
    best_val = _objective(best_a, best_b)
    
    def _obj_constrained(x):
        a = x[:11]
        a_full = list(a) + [1.0 - np.sum(a)]
        b = x[11:22]
        b_full = list(b) + [1.0 - np.sum(b)]
        if np.any(np.abs(a_full) > 50) or np.any(np.abs(b_full) > 50):
            return 1e10
        return _objective(a_full, b_full)
    
    x0 = np.array([1.0/12]*22)
    
    # Nelder-Mead
    try:
        res = optimize.minimize(_obj_constrained, x0, method='Nelder-Mead',
                                options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8})
        if res.fun < best_val:
            best_val = res.fun
            a = res.x[:11]
            best_a = list(a) + [1.0 - np.sum(a)]
            b = res.x[11:22]
            best_b = list(b) + [1.0 - np.sum(b)]
    except:
        pass
    
    if time.time() > _DEADLINE:
        return [max(-50, min(50, v)) for v in best_a], [max(-50, min(50, v)) for v in best_b]
    
    # Powell
    x_best = np.zeros(22)
    x_best[:11] = best_a[:11]
    x_best[11:22] = best_b[:11]
    try:
        res = optimize.minimize(_obj_constrained, x_best, method='Powell',
                                options={'maxiter': 80, 'xtol': 1e-6, 'ftol': 1e-8})
        if res.fun < best_val:
            best_val = res.fun
            a = res.x[:11]
            best_a = list(a) + [1.0 - np.sum(a)]
            b = res.x[11:22]
            best_b = list(b) + [1.0 - np.sum(b)]
    except:
        pass
    
    if time.time() > _DEADLINE:
        return [max(-50, min(50, v)) for v in best_a], [max(-50, min(50, v)) for v in best_b]
    
    return [max(-50, min(50, v)) for v in best_a], [max(-50, min(50, v)) for v in best_b]
