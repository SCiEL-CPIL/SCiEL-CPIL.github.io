"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import scipy.optimize
import scipy.linalg

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

def _expm_hermitian_scaled(w, V, scale):
    """Compute exp(-i * scale * H) given eigenvalues w and eigenvectors V."""
    exp_w = np.exp(-1j * scale * w)
    return (V * exp_w) @ V.conj().T

# Pre-compute exact propagators
_exact_U = [ _expm_hermitian_scaled(_wH, _VH, t) for t in _T ]

def _compute_error(a, b):
    """Compute the mean spectral norm error over the three time points."""
    errors = []
    dim = 64
    for t, U_exact in zip(_T, _exact_U):
        U = np.eye(dim, dtype=complex)
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

def coefficients(stages):
    """
    Find optimal 12-stage product formula coefficients using Dual Annealing global optimization.
    """
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # Fallback: Simple Strang splitting (Inspiration 1)
    # substeps = 6 (since 12 stages / 2 per Strang step)
    # a = [1/12]*12, b = [1/6, 0, 1/6, 0, ...] ?
    # Actually, Strang is 0.5 A, 1 B, 0.5 A.
    # For 12 stages: 6 substeps of (0.5/6 A, 1/6 B, 0.5/6 A)
    # a = [1/12] * 12
    # b = [1/6, 0, 1/6, 0, 1/6, 0, 1/6, 0, 1/6, 0, 1/6, 0]
    # Sum a = 12*(1/12) = 1. Sum b = 6*(1/6) = 1.
    fallback_a = [1.0/12.0] * 12
    fallback_b = [0.0] * 12
    for i in range(0, 12, 2):
        fallback_b[i] = 1.0/6.0
    
    best_params = None
    best_error = _compute_error(fallback_a, fallback_b)
    
    def objective(params):
        nonlocal best_params, best_error
        if time.time() > deadline:
            raise ValueError("Time limit exceeded")
            
        # params are 24 values: a[0..11], b[0..11]
        a = params[:12]
        b = params[12:24]
        
        # Check bounds
        if np.any(a < -50) or np.any(a > 50) or np.any(b < -50) or np.any(b > 50):
            return 1e10
            
        sum_a = np.sum(a)
        sum_b = np.sum(b)
        
        # Normalize to satisfy sum=1 constraint
        # If sum is close to 0, return high penalty
        if abs(sum_a) < 1e-12 or abs(sum_b) < 1e-12:
            return 1e10
            
        a_norm = a / sum_a
        b_norm = b / sum_b
        
        # Check if normalization pushed values out of bounds?
        # The task says coefficients must be in [-50, 50].
        # If a[i] is small and sum is small, a_norm[i] could be large.
        # But we check bounds on the normalized values too?
        # The evaluator checks the returned a, b.
        # So we should check bounds on a_norm, b_norm.
        if np.any(a_norm < -50) or np.any(a_norm > 50) or np.any(b_norm < -50) or np.any(b_norm > 50):
            return 1e10

        err = _compute_error(a_norm, b_norm)
        
        if err < best_error:
            best_error = err
            best_params = (a_norm.copy(), b_norm.copy())
            
        return err

    # Initial guess: Uniform distribution
    x0 = np.ones(24) / 12.0
    # b should sum to 1 too, so 1/12 is fine for initial guess.
    # But Strang is better. Let's use Strang as x0.
    x0 = np.concatenate([fallback_a, fallback_b])
    
    # Bounds for dual annealing
    # We optimize in the space of raw values, then normalize.
    # So bounds can be wide, e.g., [-50, 50]
    bounds = [(-50.0, 50.0)] * 24
    
    try:
        # Use dual_annealing for global search
        # maxiter is not a strict time limit, so we rely on the callback or internal time check
        # dual_annealing doesn't have a direct time limit parameter, so we use a wrapper
        # that raises an exception on timeout, which we catch.
        
        # To prevent infinite loops if the exception isn't caught properly by scipy,
        # we can also limit the number of function evaluations if we track them.
        # However, the exception approach is standard for time-boxing.
        
        result = scipy.optimize.dual_annealing(
            objective, 
            x0, 
            bounds=bounds, 
            seed=42,
            maxiter=10000, # High enough to be time-limited
            # We can also use 'initial_temp' and 'cooling_rate'
        )
        
        # If it finished without exception, use the result
        if result.success or result.fun < best_error:
            a_final = result.x[:12]
            b_final = result.x[12:24]
            sum_a = np.sum(a_final)
            sum_b = np.sum(b_final)
            if abs(sum_a) > 1e-12:
                a_final = a_final / sum_a
            else:
                a_final = fallback_a
            if abs(sum_b) > 1e-12:
                b_final = b_final / sum_b
            else:
                b_final = fallback_b
                
            if _is_valid(a_final, b_final):
                return list(a_final), list(b_final)
                
    except ValueError as e:
        if "Time limit" in str(e):
            pass # Fall back to best found
        else:
            raise
    except Exception:
        pass # Fall back to best found or default

    # Use the best params found during the optimization
    if best_params is not None:
        a_final, b_final = best_params
        if _is_valid(a_final, b_final):
            return list(a_final), list(b_final)

    # Ultimate fallback
    return fallback_a, fallback_b

# EVOLVE-BLOCK-END
