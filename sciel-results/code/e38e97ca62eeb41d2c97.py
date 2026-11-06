import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = time.time() + 140.0  # safety margin before 150s kill

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
        """op (2x2) acting on site (0-indexed)."""
        ops = []
        for s in range(n):
            ops.append(op if s == site else I)
        # build in order site 0 .. site n-1
        result = ops[0]
        for s in range(1, n):
            result = np.kron(result, ops[s])
        return result

    def bond_op(op1, op2, s1, s2):
        ops = []
        for s in range(n):
            if s == s1:
                ops.append(op1)
            elif s == s2:
                ops.append(op2)
            else:
                ops.append(I)
        result = ops[0]
        for s in range(1, n):
            result = np.kron(result, ops[s])
        return result

    # XXZ bond: (1/2)(Sx Sx + Sy Sy) + delta Sz Sz  = (1/4)(sx sx + sy sy) + delta/4 sz sz
    def xxz_bond(s1, s2):
        return (0.25 * bond_op(sx, sx, s1, s2)
                + 0.25 * bond_op(sy, sy, s1, s2)
                + (delta / 4.0) * bond_op(sz, sz, s1, s2))

    # A: odd bonds (sites 1-2, 3-4 in 0-index) + half field on all sites
    A = np.zeros((dim, dim), dtype=complex)
    A += xxz_bond(1, 2)
    A += xxz_bond(3, 4)
    for s in range(n):
        A += (field / 2.0) * site_op(sz, s)

    # B: even bonds (sites 0-1, 2-3, 4-5) + half field on all sites
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

def _expm_hermitian(eigvals, eigvecs, t):
    """Compute exp(-i H t) for Hermitian H using precomputed eigendecomposition."""
    phase = np.exp(-1j * eigvals * t)
    return (eigvecs * phase) @ eigvecs.conj().T

def coefficients(stages):
    """Optimize 24 coefficients to minimize mean spectral-norm error."""
    if time.time() > _DEADLINE - 20:
        # Not enough time; return a sane Strang-like default
        a = [1.0/12.0] * 12
        b = [1.0/12.0] * 12
        return a[:stages], b[:stages]

    A, B = _get_AB()
    t_list = [0.25, 0.5, 1.0]
    
    # Precompute exact propagators
    exact = {}
    for t in t_list:
        exact[t] = _expm_hermitian(_AB_eigvals, _AB_eigvecs, t)

    def _expm_A(t):
        return _expm_hermitian(_A_eigvals, _A_eigvecs, t)
    
    def _expm_B(t):
        return _expm_hermitian(_B_eigvals, _B_eigvecs, t)

    def _sym_to_full(params):
        """
        params: [a1..a5, b1..b5]
        Returns full a (12) and b (12) assuming symmetry a_k = a_{13-k}, b_k = b_{13-k}
        Constraint: sum(a) = 1, sum(b) = 1
        """
        a_half = params[:5]
        b_half = params[5:10]
        
        # Calculate the 6th element to satisfy sum constraint
        # Sum(a) = 2 * sum(a_1..a_6) = 1 => sum(a_1..a_6) = 0.5
        a6 = 0.5 - np.sum(a_half)
        b6 = 0.5 - np.sum(b_half)
        
        a_full = np.array([a_half[0], a_half[1], a_half[2], a_half[3], a_half[4], a6, 
                           a6, a_half[4], a_half[3], a_half[2], a_half[1], a_half[0]])
        b_full = np.array([b_half[0], b_half[1], b_half[2], b_half[3], b_half[4], b6, 
                           b6, b_half[4], b_half[3], b_half[2], b_half[1], b_half[0]])
        
        return a_full, b_full

    def U_of(a, b, t):
        U = np.eye(64, dtype=complex)
        for k in range(12):
            Eb = _expm_B(b[k] * t)
            Ea = _expm_A(a[k] * t)
            U = Eb @ Ea @ U
        return U

    def obj(params):
        a, b = _sym_to_full(params)
        # Check bounds
        if np.any(np.abs(a) > 50) or np.any(np.abs(b) > 50):
            return 1e10
        total = 0.0
        for t in t_list:
            U = U_of(a, b, t)
            err = U - exact[t]
            total += np.linalg.norm(err, 2)
        return total / len(t_list)

    # Initial guess: uniform distribution
    # a1..a6 = 1/12. Sum = 6/12 = 0.5. Correct.
    # So a1..a5 = 1/12, a6 = 0.5 - 5/12 = 1/12.
    x0 = np.full(10, 1.0/12.0)
    
    # Define bounds for parameters to keep coefficients in [-50, 50]
    # Using tighter bounds to ensure a6 stays reasonable
    bounds = [(-1.0, 1.0)] * 10
    
    best_x = x0.copy()
    best_val = obj(x0)
    
    # Run L-BFGS-B
    try:
        res = minimize(obj, x0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 200, 'ftol': 1e-10, 'maxls': 50})
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x
    except Exception:
        pass

    # Try a few random restarts if time permits
    if time.time() < _DEADLINE - 30:
        for i in range(5):
            x_rand = x0 + np.random.randn(10) * 0.1
            x_rand = np.clip(x_rand, -1.0, 1.0)
            try:
                res = minimize(obj, x_rand, method='L-BFGS-B', bounds=bounds, 
                               options={'maxiter': 100, 'ftol': 1e-10})
                if res.fun < best_val:
                    best_val = res.fun
                    best_x = res.x
            except Exception:
                pass

    a_full, b_full = _sym_to_full(best_x)
    
    # Ensure output format
    a = a_full.tolist()
    b = b_full.tolist()
    
    # Pad/truncate to stages
    while len(a) < stages:
        a.append(0.0)
        b.append(0.0)
        
    return a[:stages], b[:stages]
