import numpy as np
import time
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

def _build_hamiltonians():
    """
    Build A (odd bonds + half field) and B (even bonds + half field) for a
    6-site open XXZ Heisenberg chain with anisotropy 0.8 and field 0.3.
    """
    N = 6
    dim = 2 ** N
    gamma = 0.8  # anisotropy
    h = 0.3      # field
    
    # Pauli matrices
    sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    ident = np.eye(2, dtype=complex)
    
    def tensor_product(matrices):
        result = matrices[0]
        for m in matrices[1:]:
            result = np.kron(result, m)
        return result
    
    # Build operators for each site
    def site_op(op, site):
        ops = [ident] * N
        ops[site] = op
        return tensor_product(ops)
    
    # Build A: odd bonds (0-1, 2-3, 4-5) + half field
    A = np.zeros((dim, dim), dtype=complex)
    for j in [0, 2, 4]:
        ops_sx_sx = [ident] * N
        ops_sx_sx[j] = sx
        ops_sx_sx[j+1] = sx
        A += tensor_product(ops_sx_sx)
        
        ops_sy_sy = [ident] * N
        ops_sy_sy[j] = sy
        ops_sy_sy[j+1] = sy
        A += tensor_product(ops_sy_sy)
        
        ops_sz_sz = [ident] * N
        ops_sz_sz[j] = sz
        ops_sz_sz[j+1] = sz
        A += gamma * tensor_product(ops_sz_sz)
    
    for j in range(N):
        A -= (h / 2.0) * site_op(sz, j)
        
    # Ensure Hermitian
    A = (A + A.conj().T) / 2.0

    # Build B: even bonds (1-2, 3-4) + half field
    B = np.zeros((dim, dim), dtype=complex)
    for j in [1, 3]:
        ops_sx_sx = [ident] * N
        ops_sx_sx[j] = sx
        ops_sx_sx[j+1] = sx
        B += tensor_product(ops_sx_sx)
        
        ops_sy_sy = [ident] * N
        ops_sy_sy[j] = sy
        ops_sy_sy[j+1] = sy
        B += tensor_product(ops_sy_sy)
        
        ops_sz_sz = [ident] * N
        ops_sz_sz[j] = sz
        ops_sz_sz[j+1] = sz
        B += gamma * tensor_product(ops_sz_sz)
    
    for j in range(N):
        B -= (h / 2.0) * site_op(sz, j)
        
    # Ensure Hermitian
    B = (B + B.conj().T) / 2.0

    return A, B

# Pre-compute the Hamiltonians and eigen-decompositions
_A, _B = _build_hamiltonians()
_H = _A + _B

# Eigen-decompositions for Hermitian matrices: H = V diag(w) V^dagger
# exp(-i t H) = V diag(exp(-i t w)) V^dagger
_wA, _VA = np.linalg.eigh(_A)
_wB, _VB = np.linalg.eigh(_B)
_wH, _VH = np.linalg.eigh(_H)

_T = [0.25, 0.5, 1.0]
_DIM = 64

def _expm_hermitian_scaled(w, V, scale):
    """Compute exp(-i * scale * H) given eigenvalues w and eigenvectors V."""
    exp_w = np.exp(-1j * scale * w)
    return (V * exp_w) @ V.conj().T

# Pre-compute exact propagators
_exact_U = [ _expm_hermitian_scaled(_wH, _VH, t) for t in _T ]

def _compute_error_from_full(a, b):
    """Compute the mean spectral norm error over the three time points."""
    errors = []
    for t, U_exact in zip(_T, _exact_U):
        U = np.eye(_DIM, dtype=complex)
        for k in range(12):
            if b[k] != 0.0:
                U_B = _expm_hermitian_scaled(_wB, _VB, b[k] * t)
                U = U_B @ U
            if a[k] != 0.0:
                U_A = _expm_hermitian_scaled(_wA, _VA, a[k] * t)
                U = U_A @ U
        err = np.linalg.norm(U - U_exact, 2)
        errors.append(err)
    return np.mean(errors)

def _is_valid(a, b):
    """Check if coefficients are valid."""
    if len(a) != 12 or len(b) != 12:
        return False
    for x in a + b:
        if not np.isfinite(x) or x < -50 or x > 50:
            return False
    if abs(sum(a) - 1.0) > 1e-8 or abs(sum(b) - 1.0) > 1e-8:
        return False
    return True

def _params_to_full(params):
    """
    Convert 10 parameters (a0..a4, b0..b4) to full 12-length symmetric lists.
    a5 and b5 are determined by the sum constraint: sum(a0..a5)=0.5.
    Symmetry: a[11-i] = a[i].
    """
    a_sym = np.zeros(6)
    b_sym = np.zeros(6)
    
    a_sym[:5] = params[:5]
    b_sym[:5] = params[5:10]
    
    # Calculate a5 and b5 to satisfy sum = 0.5
    a_sym[5] = 0.5 - np.sum(a_sym[:5])
    b_sym[5] = 0.5 - np.sum(b_sym[:5])
    
    # Check bounds for the derived values and all values
    if np.any(a_sym < -50) or np.any(a_sym > 50) or np.any(b_sym < -50) or np.any(b_sym > 50):
        return None, None
        
    if np.any(~np.isfinite(a_sym)) or np.any(~np.isfinite(b_sym)):
        return None, None

    a_full = np.zeros(12)
    b_full = np.zeros(12)
    for i in range(6):
        a_full[i] = a_sym[i]
        a_full[11-i] = a_sym[i]
        b_full[i] = b_sym[i]
        b_full[11-i] = b_sym[i]
        
    return a_full, b_full

def _objective(params):
    """
    Objective function for L-BFGS-B.
    params: 10-dim array.
    Returns: error or penalty.
    """
    a, b = _params_to_full(params)
    if a is None:
        return 1e10
    return _compute_error_from_full(a, b)

def coefficients(stages):
    """
    Find optimal 12-stage product formula coefficients using L-BFGS-B on a 10D symmetric parameter space.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Initial guess: Uniform coefficients
    # a_i = 1/12 for all i. sum(a0..a5) = 5/12. a5 = 0.5 - 5/12 = 1/12.
    # b_i = 1/12 for all i.
    x0 = np.array([1.0/12.0] * 10)
    
    # Bounds for the free variables.
    # Since a5 = 0.5 - sum(a0..a4), if a0..a4 are in [-50, 50], a5 can be large.
    # We set loose bounds for the free variables and rely on the penalty in _params_to_full.
    # However, L-BFGS-B works best with tight bounds.
    # Let's use [-10, 10] for free vars, which is generous enough for typical splitting coefficients.
    bounds = [(-10.0, 10.0)] * 10
    
    best_params = None
    best_error = _objective(x0)
    
    if best_error < 1e9:
        best_params = x0.copy()
    
    # Run L-BFGS-B
    try:
        res = minimize(_objective, x0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if res.success and res.fun < best_error:
            best_error = res.fun
            best_params = res.x.copy()
            
        # Try a few random restarts if time permits
        rng = np.random.default_rng(42)
        for _ in range(5):
            if time.time() > deadline - 20:
                break
            # Random start near the uniform solution
            x_rand = x0 + rng.normal(0, 0.05, 10)
            # Clip to bounds
            x_rand = np.clip(x_rand, -10, 10)
            
            res_rand = minimize(_objective, x_rand, method='L-BFGS-B', bounds=bounds, 
                                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res_rand.success and res_rand.fun < best_error:
                best_error = res_rand.fun
                best_params = res_rand.x.copy()
                
    except Exception:
        pass
    
    # If optimization failed or didn't improve, use the initial guess
    if best_params is None:
        best_params = x0
    
    # Convert best_params to full a, b
    a, b = _params_to_full(best_params)
    
    # Fallback if conversion failed (should not happen with valid params)
    if a is None:
        a = [1.0/12.0] * 12
        b = [1.0/12.0] * 12
    else:
        a = list(a)
        b = list(b)
        
    # Final validation and correction
    if not _is_valid(a, b):
        # Force a valid solution
        a = [1.0/12.0] * 12
        b = [1.0/12.0] * 12
        
    return a, b

# EVOLVE-BLOCK-END
