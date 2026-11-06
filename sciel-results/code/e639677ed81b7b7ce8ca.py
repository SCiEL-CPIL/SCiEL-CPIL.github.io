import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = time.time() + 130.0  # safety margin before 150s kill

def _build_hamiltonians():
    """Build A (odd bonds 2-3,4-5 + half field) and B (even bonds 1-2,3-4,5-6 + half field)."""
    n = 6
    dim = 2 ** n
    delta = 0.8
    field = 0.3
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)

    def site_op(op, site):
        ops = [I] * n
        ops[site] = op
        result = ops[0]
        for s in range(1, n):
            result = np.kron(result, ops[s])
        return result

    def bond_op(op1, op2, s1, s2):
        ops = [I] * n
        ops[s1] = op1
        ops[s2] = op2
        result = ops[0]
        for s in range(1, n):
            result = np.kron(result, ops[s])
        return result

    def xxz_bond(s1, s2):
        return (0.25 * bond_op(sx, sx, s1, s2)
                + 0.25 * bond_op(sy, sy, s1, s2)
                + (delta / 4.0) * bond_op(sz, sz, s1, s2))

    A = np.zeros((dim, dim), dtype=complex)
    A += xxz_bond(1, 2)
    A += xxz_bond(3, 4)
    for s in range(n):
        A += (field / 2.0) * site_op(sz, s)

    B = np.zeros((dim, dim), dtype=complex)
    B += xxz_bond(0, 1)
    B += xxz_bond(2, 3)
    B += xxz_bond(4, 5)
    for s in range(n):
        B += (field / 2.0) * site_op(sz, s)

    return A, B

_A = None
_B = None
_A_eigvals = None
_A_eigvecs = None
_B_eigvals = None
_B_eigvecs = None
_AB_eigvals = None
_AB_eigvecs = None

def _get_AB():
    global _A, _B, _A_eigvals, _A_eigvecs, _B_eigvals, _B_eigvecs, _AB_eigvals, _AB_eigvecs
    if _A is None:
        _A, _B = _build_hamiltonians()
        _A_eigvals, _A_eigvecs = np.linalg.eigh(_A)
        _B_eigvals, _B_eigvecs = np.linalg.eigh(_B)
        _AB_eigvals, _AB_eigvecs = np.linalg.eigh(_A + _B)
    return _A, _B

def _expm_hermitian(t, eigvals, eigvecs):
    """Compute exp(-i H t) for Hermitian H using precomputed eigendecomposition."""
    phase = np.exp(-1j * eigvals * t)
    return (eigvecs * phase) @ eigvecs.conj().T

def _clamp_to_box(x, lo=-50.0, hi=50.0):
    return np.clip(x, lo, hi)

def coefficients(stages):
    """Optimize 24 coefficients to minimize mean spectral-norm error."""
    if time.time() > _DEADLINE - 20:
        # Not enough time; return a sane Strang-like default
        a = [1.0 / stages] * stages
        b = [1.0 / stages] * stages
        return a, b

    _A, _B = _get_AB()
    t_list = [0.25, 0.5, 1.0]
    
    # Precompute exact propagators
    exact = {}
    for t in t_list:
        exact[t] = _expm_hermitian(t, _AB_eigvals, _AB_eigvecs)

    def U_of(a, b, t):
        """Compute product formula U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A) for k=0..11."""
        U = np.eye(64, dtype=complex)
        for k in range(12):
            Eb = _expm_hermitian(b[k] * t, _B_eigvals, _B_eigvecs)
            Ea = _expm_hermitian(a[k] * t, _A_eigvals, _A_eigvecs)
            U = Eb @ Ea @ U
        return U

    def compute_error(a, b):
        total = 0.0
        for t in t_list:
            U = U_of(a, b, t)
            err = U - exact[t]
            total += np.linalg.norm(err, 2)
        return total / len(t_list)

    # Parameterization: use symmetric structure to reduce dimensions
    # a = [a1, a2, a3, a4, a5, a6, a6, a5, a4, a3, a2, a1]
    # b = [b1, b2, b3, b4, b5, b6, b6, b5, b4, b3, b2, b1]
    # Constraints: sum(a) = 1, sum(b) = 1
    # Free params: a1..a5 (a6 = 0.5 - sum(a1..a5)), b1..b5 (b6 = 0.5 - sum(b1..b5))
    # Total: 10 free parameters

    def sym_to_full(theta):
        a_free = theta[:5]
        b_free = theta[5:10]
        
        a6 = 0.5 - np.sum(a_free)
        b6 = 0.5 - np.sum(b_free)
        
        a = np.array([a_free[0], a_free[1], a_free[2], a_free[3], a_free[4], a6, 
                      a6, a_free[4], a_free[3], a_free[2], a_free[1], a_free[0]])
        b = np.array([b_free[0], b_free[1], b_free[2], b_free[3], b_free[4], b6, 
                      b6, b_free[4], b_free[3], b_free[2], b_free[1], b_free[0]])
        
        return a, b

    def obj(theta):
        a, b = sym_to_full(theta)
        a = _clamp_to_box(a)
        b = _clamp_to_box(b)
        return compute_error(a, b)

    # Initial guess: uniform
    theta0 = np.full(10, 1.0/12.0)
    
    best_theta = theta0.copy()
    best_val = obj(theta0)
    
    # Try multiple optimization strategies
    strategies = [
        {'method': 'L-BFGS-B', 'options': {'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}},
        {'method': 'Nelder-Mead', 'options': {'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}},
        {'method': 'Powell', 'options': {'maxiter': 500, 'ftol': 1e-12, 'xtol': 1e-10}},
    ]

    # Generate several starting points
    start_points = [theta0.copy()]
    # Add some perturbations
    for _ in range(5):
        perturbed = theta0 + np.random.normal(0, 0.02, 10)
        start_points.append(perturbed)
    # Add Suzuki-Trotter-inspired starts
    # 4th order coefficients: z1 = 1/(2-2^(1/3)), z2 = -2^(1/3)/(2-2^(1/3)), z3 = 2/(2-2^(1/3))
    cbrt2 = 2 ** (1/3)
    z1 = 1.0 / (2 - cbrt2)
    z2 = -cbrt2 / (2 - cbrt2)
    z3 = 2.0 / (2 - cbrt2)
    # For 12 stages symmetric, try various distributions
    st_starts = [
        np.array([z1/4, z2/4, z3/4, 0, 0, z1/4, z2/4, z3/4, 0, 0]),
        np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
        np.array([0.15, 0.05, 0.1, 0.05, 0.15, 0.15, 0.05, 0.1, 0.05, 0.15]),
    ]
    start_points.extend(st_starts)

    for sp in start_points:
        if time.time() > _DEADLINE - 10:
            break
        for strat in strategies:
            if time.time() > _DEADLINE - 5:
                break
            try:
                res = minimize(obj, sp, method=strat['method'], options=strat['options'])
                if res.fun < best_val:
                    best_val = res.fun
                    best_theta = res.x
            except Exception:
                pass

    # Extract final coefficients
    a_free = best_theta[:5]
    b_free = best_theta[5:10]
    
    a6 = 0.5 - np.sum(a_free)
    b6 = 0.5 - np.sum(b_free)
    
    a = np.array([a_free[0], a_free[1], a_free[2], a_free[3], a_free[4], a6, 
                  a6, a_free[4], a_free[3], a_free[2], a_free[1], a_free[0]])
    b = np.array([b_free[0], b_free[1], b_free[2], b_free[3], b_free[4], b6, 
                  b6, b_free[4], b_free[3], b_free[2], b_free[1], b_free[0]])
    
    a = _clamp_to_box(a).tolist()
    b = _clamp_to_box(b).tolist()

    # Pad/truncate to stages
    while len(a) < stages:
        a.append(0.0)
        b.append(0.0)
    return a[:stages], b[:stages]
