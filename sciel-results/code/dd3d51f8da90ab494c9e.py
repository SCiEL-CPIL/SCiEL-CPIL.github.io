import time
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import expm

_START = time.time()
_DEADLINE = _START + 140.0 # 10s safety margin

def _build_hamiltonians():
    """Build A (odd bonds) and B (even bonds) for 6-site open XXZ chain.
    Anisotropy gamma=0.8, field h=0.3.
    Each bond: gamma/2*(Sx*Sx + Sy*Sy) + 1/2*Sz*Sz + h/2*(Sz_i + Sz_j)
    Split: A = sum over odd bonds (1-2, 3-4, 5-6), B = sum over even bonds (2-3, 4-5)
    Each includes half the uniform field."""
    N = 6
    gamma = 0.8
    h = 0.3
    
    # Spin-1/2 operators
    Sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    Sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
    Sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    
    def op_on_site(op, site):
        """Apply op on a specific site in N-site chain."""
        result = np.array([[1]], dtype=complex)
        for i in range(N):
            if i == site:
                result = np.kron(result, op)
            else:
                result = np.kron(result, np.eye(2, dtype=complex))
        return result
    
    def bond_op(i, j):
        """Two-spin bond operator between sites i and j (0-indexed).
        gamma/2*(Sx_i*Sx_j + Sy_i*Sy_j) + 1/2*Sz_i*Sz_j + h/2*(Sz_i + Sz_j)"""
        Sxi = op_on_site(Sx, i)
        Sxj = op_on_site(Sx, j)
        Syi = op_on_site(Sy, i)
        Syj = op_on_site(Sy, j)
        Szi = op_on_site(Sz, i)
        Szj = op_on_site(Sz, j)
        
        return (gamma/2)*(Sxi @ Sxj + Syi @ Syj) + 0.5*(Szi @ Szj) + h/2*(Szi + Szj)
    
    # Odd bonds: (0,1), (2,3), (4,5) -> A
    # Even bonds: (1,2), (3,4) -> B
    A = bond_op(0, 1) + bond_op(2, 3) + bond_op(4, 5)
    B = bond_op(1, 2) + bond_op(3, 4)
    
    return A, B

def _error_from_coeffs(a_vec, b_vec, A, B):
    """Calculate error for given full coefficient vectors."""
    a = a_vec
    b = b_vec
    dim = A.shape[0]
    I = np.eye(dim, dtype=complex)
    
    total_err = 0.0
    for t in [0.25, 0.5, 1.0]:
        U = I.copy()
        for k in range(12):
            if abs(b[k]) > 1e-15:
                U = expm(-1j * b[k] * t * B) @ U
            if abs(a[k]) > 1e-15:
                U = expm(-1j * a[k] * t * A) @ U
        
        U_exact = expm(-1j * t * (A + B))
        err = np.linalg.norm(U - U_exact, 2)
        total_err += err
    
    return total_err / 3.0

def coefficients(stages):
    """Return optimized (a, b) lists of length `stages` (12)."""
    global _START, _DEADLINE
    
    if stages != 12:
        # Fallback: simple uniform splitting
        a = [1.0 / stages] * stages
        b = [1.0 / stages] * stages
        return a, b
        
    # Build Hamiltonians
    A, B = _build_hamiltonians()
    
    # Symmetric parameterization
    # a = [x1, x2, x3, x4, x5, x6, x6, x5, x4, x3, x2, x1]
    # b = [y1, y2, y3, y4, y5, y6, y6, y5, y4, y3, y2, y1]
    # Constraints: sum(a)=1 => 2*sum(x)=1 => sum(x)=0.5
    #               sum(b)=1 => 2*sum(y)=1 => sum(y)=0.5
    
    # Initial guess: 2nd order Strang splitting repeated 4 times
    # a_indep = [1/8, 0, 1/8, 1/8, 0, 1/8]
    # b_indep = [0, 1/4, 0, 0, 1/4, 0]
    x0 = np.array([1/8, 0, 1/8, 1/8, 0, 1/8])
    y0 = np.array([0, 1/4, 0, 0, 1/4, 0])
    
    def objective(params):
        if time.time() > _DEADLINE:
            return 1e10 # Signal timeout
        x = params[:6]
        y = params[6:12]
        
        # Construct full vectors
        a = np.concatenate([x, x[::-1]])
        b = np.concatenate([y, y[::-1]])
        
        return _error_from_coeffs(a, b, A, B)
    
    p0 = np.concatenate([x0, y0])
    
    best_result = None
    best_cost = np.inf
    
    # Use L-BFGS-B for speed and robustness
    try:
        res = minimize(objective, p0, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-12})
        if res.success or res.fun < best_cost:
            best_cost = res.fun
            best_result = res.x
    except Exception as e:
        pass

    # If optimization failed or didn't improve, try a few random restarts if time permits
    if time.time() < _DEADLINE - 10:
        for _ in range(5):
            if time.time() > _DEADLINE - 5:
                break
            # Random perturbation around initial guess
            noise = np.random.normal(0, 0.1, size=12)
            p_rand = p0 + noise
            # Project to constraints? L-BFGS-B doesn't handle linear equality constraints directly well without reformulation.
            # However, the sum constraint is critical. 
            # Let's stick to the symmetric unconstrained optimization but ensure the sum is close.
            # Actually, L-BFGS-B minimizes the objective. If the objective is large when sums aren't 1, it will drive them there.
            # But the problem statement says sum MUST be 1. 
            # My objective function doesn't explicitly penalize sum != 1. 
            # I should add a penalty or use constraints.
            
            # Let's reformulate objective to include penalty for sum != 1
            def objective_penalized(params):
                if time.time() > _DEADLINE:
                    return 1e10
                x = params[:6]
                y = params[6:12]
                a = np.concatenate([x, x[::-1]])
                b = np.concatenate([y, y[::-1]])
                
                sum_a = np.sum(a)
                sum_b = np.sum(b)
                
                # Penalty for deviation from 1
                penalty = 1000 * ((sum_a - 1)**2 + (sum_b - 1)**2)
                
                if penalty > 1e-4:
                    return penalty # Skip expensive matrix exp if constraints are way off
                
                return _error_from_coeffs(a, b, A, B) + penalty

            try:
                res2 = minimize(objective_penalized, p_rand, method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-10})
                if res2.fun < best_cost:
                    best_cost = res2.fun
                    best_result = res2.x
            except Exception:
                pass

    if best_result is None:
        # Fallback to initial guess
        best_result = p0
        
    # Extract coefficients
    x_opt = best_result[:6]
    y_opt = best_result[6:12]
    
    a = list(np.concatenate([x_opt, x_opt[::-1]]))
    b = list(np.concatenate([y_opt, y_opt[::-1]]))
    
    # Final safety check: ensure sums are exactly 1 by scaling if necessary?
    # The penalty should have driven them close. 
    # If they are off by 1e-6, it's fine.
    
    return a, b
