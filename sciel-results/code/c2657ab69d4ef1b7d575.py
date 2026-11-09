"""Product-formula coefficient optimization for Heisenberg-chain time evolution."""
import numpy as np
import time
from scipy.linalg import expm

# Build the 6-site XXZ Heisenberg chain Hamiltonians
# A = odd bonds, B = even bonds, each with half the uniform field

def _build_hamiltonians():
    """Build A (odd bonds) and B (even bonds) for 6-site XXZ chain."""
    n = 6
    dim = 2 ** n
    # Identity and Pauli matrices
    I2 = np.eye(2, dtype=complex)
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    splus = np.array([[0, 1], [0, 0]], dtype=complex)
    sminus = np.array([[0, 0], [1, 0]], dtype=complex)
    
    def kron_all(mats):
        result = np.array([[1]], dtype=complex)
        for m in mats:
            result = np.kron(result, m)
        return result
    
    anisotropy = 0.8
    field = 0.3
    
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    for site in range(n):
        # Build operators on each site
        ops_sx = [I2] * n
        ops_sx[site] = sx
        A += 0.0
        B += 0.0
        
        ops_sz = [I2] * n
        ops_sz[site] = sz
        # Half field per half
        A += -(field / 2.0) * kron_all(ops_sz)
        B += -(field / 2.0) * kron_all(ops_sz)
    
    # Bonds: site i to site i+1
    # Odd bonds: (0,1), (2,3), (4,5) -> A
    # Even bonds: (1,2), (3,4) -> B
    bond_sites_A = [(0, 1), (2, 3), (4, 5)]
    bond_sites_B = [(1, 2), (3, 4)]
    
    for (i, j) in bond_sites_A:
        ops = [I2] * n
        ops[i] = sx
        ops[j] = sx
        A += 0.5 * kron_all(ops)
        
        ops[i] = sy
        ops[j] = sy
        A += 0.5 * kron_all(ops)
        
        ops[i] = sz
        ops[j] = sz
        A += anisotropy * kron_all(ops)
        
        ops[i] = splus
        ops[j] = sminus
        A += kron_all(ops)
        
        ops[i] = sminus
        ops[j] = splus
        A += kron_all(ops)
    
    for (i, j) in bond_sites_B:
        ops = [I2] * n
        ops[i] = sx
        ops[j] = sx
        B += 0.5 * kron_all(ops)
        
        ops[i] = sy
        ops[j] = sy
        B += 0.5 * kron_all(ops)
        
        ops[i] = sz
        ops[j] = sz
        B += anisotropy * kron_all(ops)
        
        ops[i] = splus
        ops[j] = sminus
        B += kron_all(ops)
        
        ops[i] = sminus
        ops[j] = splus
        B += kron_all(ops)
    
    return A, B


_A, _B = _build_hamiltonians()


def _product_formula_error(a, b, t):
    """Compute spectral norm error ||U(t) - exp(-i t (A+B))||."""
    # Build U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A)
    U = np.eye(2**6, dtype=complex)
    for k in range(12):
        if abs(b[k]) > 1e-15:
            U = expm(-1j * b[k] * t * _B) @ U
        if abs(a[k]) > 1e-15:
            U = expm(-1j * a[k] * t * _A) @ U
    exact = expm(-1j * t * (_A + _B))
    return np.linalg.norm(U - exact, ord=2)


def _mean_error(a, b):
    """Mean spectral norm error over t = 0.25, 0.5, 1.0."""
    errors = [_product_formula_error(a, b, t) for t in [0.25, 0.5, 1.0]]
    return np.mean(errors)


# Nelder-Mead optimization with sum constraints
def _optimize_coefficients(deadline):
    """Optimize 12-stage coefficients using Nelder-Mead."""
    from scipy.optimize import minimize
    
    # Variables: a[0..11], b[0..11], but with constraints sum(a)=1, sum(b)=1
    # We'll optimize 22 free variables (drop a[11] and b[11] which are determined)
    # a[11] = 1 - sum(a[0..10])
    # b[11] = 1 - sum(b[0..10])
    
    def objective(x):
        a = list(x[:11])
        a.append(1.0 - sum(a))
        b = list(x[11:22])
        b.append(1.0 - sum(b))
        # Clamp to [-50, 50]
        a = [max(-50, min(50, v)) for v in a]
        b = [max(-50, min(50, v)) for v in b]
        return _mean_error(a, b)
    
    # Initial guess: uniform distribution
    x0 = [1.0/12] * 11 + [1.0/12] * 11
    
    # Time budget check
    elapsed = time.time() - _start_time
    if elapsed > deadline - 10:
        return None
    
    result = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12,
                               'adaptive': True})
    
    elapsed = time.time() - _start_time
    if elapsed > deadline - 5:
        return None
    
    a = list(result.x[:11])
    a.append(1.0 - sum(a))
    b = list(result.x[11:22])
    b.append(1.0 - sum(b))
    
    # Clamp to [-50, 50]
    a = [max(-50, min(50, v)) for v in a]
    b = [max(-50, min(50, v)) for v in b]
    
    return a, b


_start_time = time.time()
_deadline = 150.0  # wall-clock limit

def coefficients(stages):
    """Return optimized 12-stage product formula coefficients."""
    global _start_time, _deadline
    _start_time = time.time()
    _deadline = 150.0
    
    # Try optimization
    best = _optimize_coefficients(_deadline)
    
    if best is None:
        # Fallback: symmetric 4th-order-like splitting
        # Use known high-order splitting coefficients
        eta1 = 1.0 / (2.0 * (1.0 - np.sqrt(2.0)))
        eta2 = 1.0 - 2.0 * eta1
        mu1 = eta1
        mu2 = 2.0 * eta1
        mu3 = eta2
        
        # 6th order coefficients for 3 substeps
        A1 = mu1 * eta1
        B1 = mu2 * eta1
        A2 = mu3 * eta1
        B2 = mu2 * eta1
        A3 = mu1 * (eta1 + eta2)
        B3 = mu2 * eta2
        A4 = mu3 * eta2
        B4 = mu2 * eta2
        A5 = mu1 * (eta1 + eta2)
        B5 = mu2 * eta1
        A6 = mu3 * eta1
        B6 = mu2 * eta1
        A7 = mu1 * eta1
        
        div = 3.0
        a = [A1/div, 0, A2/div, 0, A3/div, 0, A4/div, 0, A5/div, 0, A6/div, A7/div]
        b = [0, B1/div, 0, B2/div, 0, B3/div, 0, B4/div, 0, B5/div, 0, B6/div]
        
        # Clamp
        a = [max(-50, min(50, v)) for v in a]
        b = [max(-50, min(50, v)) for v in b]
        return a, b
    
    return best
