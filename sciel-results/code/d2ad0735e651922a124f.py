import numpy as np
import time
import sys

def _build_hamiltonians():
    """
    Build the Hamiltonian parts A and B for the 6-site XXZ Heisenberg chain.
    A: Odd bonds (1-2, 3-4, 5-6) + half field
    B: Even bonds (2-3, 4-5) + half field
    """
    # Spin operators
    Sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    Sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
    Sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    
    # System size
    N = 6
    dim = 2 ** N
    
    # Helper to create tensor product identity
    def ident(n):
        return np.eye(2)
    
    # Helper to apply operator at site k (0-indexed)
    def apply_op(op, k):
        # Create list of operators
        ops = [ident(2) for _ in range(N)]
        ops[k] = op
        # Tensor product
        res = ops[0]
        for op in ops[1:]:
            res = np.kron(res, op)
        return res
    
    # Helper to apply two-site operator at sites k, k+1
    def apply_two_op(op1, op2, k):
        ops = [ident(2) for _ in range(N)]
        ops[k] = op1
        ops[k+1] = op2
        res = ops[0]
        for op in ops[1:]:
            res = np.kron(res, op)
        return res

    J = 1.0
    delta = 0.8
    h = 0.3
    
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    # Field terms: half in A, half in B
    field_sum = np.zeros((dim, dim), dtype=complex)
    for k in range(N):
        field_sum += apply_op(Sz, k)
    
    A += 0.5 * h * field_sum
    B += 0.5 * h * field_sum
    
    # Bond terms
    # Odd bonds: 0-1, 2-3, 4-5 (indices 0..5)
    for k in range(0, N, 2):
        if k + 1 < N:
            term = J * (
                apply_two_op(Sx, Sx, k) + 
                apply_two_op(Sy, Sy, k) + 
                delta * apply_two_op(Sz, Sz, k)
            )
            A += term
            
    # Even bonds: 1-2, 3-4
    for k in range(1, N, 2):
        if k + 1 < N:
            term = J * (
                apply_two_op(Sx, Sx, k) + 
                apply_two_op(Sy, Sy, k) + 
                delta * apply_two_op(Sz, Sz, k)
            )
            B += term
            
    return A, B

def _spectral_norm(A):
    """Compute spectral norm (largest singular value) of matrix A."""
    # Using svd for accuracy, or norm 2
    # np.linalg.norm(A, 2) uses svd under the hood
    return np.linalg.norm(A, 2)

def coefficients(stages):
    """
    Optimize 12-stage symmetric product formula coefficients.
    """
    if stages != 12:
        # Fallback: uniform Strang-like
        substeps = stages // 2
        a = [0.5 / substeps] * (2 * substeps)
        b = []
        for _ in range(substeps):
            b += [1.0 / substeps, 0.0]
        while len(a) < stages:
            a.append(0.0)
            b.append(0.0)
        return a, b

    start_time = time.time()
    deadline = start_time + 55.0 # Leave 5s margin for overhead
    
    try:
        A, B = _build_hamiltonians()
        H = A + B
        dim = A.shape[0]
        
        times = [0.25, 0.5, 1.0]
        # Precompute exact propagators
        U_exacts = []
        for t in times:
            U_exacts.append(np.linalg.matrix_exp(-1j * t * H))
        
        # Define objective function
        # x = [a1..a6, b1..b6]
        # Symmetry: a_k = a_{13-k}, b_k = b_{13-k}
        # Sum constraints: sum(a1..a6) = 0.5, sum(b1..b6) = 0.5
        
        def objective(x):
            if time.time() > deadline:
                return 1e10 # Penalize if over time
            
            a_half = x[:6]
            b_half = x[6:12]
            
            # Check bounds
            if np.any(a_half < -50) or np.any(a_half > 50) or np.any(b_half < -50) or np.any(b_half > 50):
                return 1e10
                
            # Construct full a, b
            a = list(a_half) + list(a_half[::-1])
            b = list(b_half) + list(b_half[::-1])
            
            # Check sum constraints (should be close, but optimizer enforces)
            # For robustness, normalize if drift occurs? 
            # SLSQP enforces constraints, so we assume they are satisfied.
            
            total_error = 0.0
            U_approx = np.eye(dim, dtype=complex)
            
            for i, t in enumerate(times):
                U = np.eye(dim, dtype=complex)
                # Product from k=1 to 12: E_1 ... E_12?
                # Task: U(t) = prod_{k=1}^{12} exp(-i b_k t B) exp(-i a_k t A)
                # Standard math notation: product k=1 to N means E_1 * E_2 * ... * E_N
                # So U = E_1 E_2 ... E_12
                # U v = E_12 (... (E_1 v) ...)
                # So we apply E_1 first, then E_2, etc.
                # Matrix multiplication: U_total = E_12 @ ... @ E_1
                # If we accumulate U = I, then U = E_k @ U?
                # U = I
                # k=1: U = E_1 @ I = E_1
                # k=2: U = E_2 @ U = E_2 @ E_1
                # ...
                # k=12: U = E_12 @ ... @ E_1
                # Yes.
                
                for k in range(12):
                    Ek = np.linalg.matrix_exp(-1j * b[k] * t * B) @ np.linalg.matrix_exp(-1j * a[k] * t * A)
                    U = Ek @ U
                
                err = _spectral_norm(U - U_exacts[i])
                total_error += err
                
            return total_error / len(times)
        
        # Initial guess
        # Uniform distribution
        a_init = [1.0/12.0] * 6
        b_init = [1.0/12.0] * 6
        x0 = np.array(a_init + b_init)
        
        # Constraints
        cons = [
            {'type': 'eq', 'fun': lambda x: np.sum(x[:6]) - 0.5},
            {'type': 'eq', 'fun': lambda x: np.sum(x[6:12]) - 0.5}
        ]
        
        bounds = [(-50, 50)] * 12
        
        from scipy.optimize import minimize
        
        best_x = x0
        best_err = objective(x0)
        
        # Try SLSQP
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, 
                           options={'maxiter': 200, 'ftol': 1e-8})
            if time.time() < deadline:
                if res.fun < best_err:
                    best_err = res.fun
                    best_x = res.x
        except Exception as e:
            pass
        
        # Try L-BFGS-B with reduced variables if SLSQP failed or didn't improve much
        # This might be overkill if SLSQP works well.
        # Given the small size, SLSQP should be fine.
        
        # Extract coefficients
        a_half = best_x[:6]
        b_half = best_x[6:12]
        
        a = list(a_half) + list(a_half[::-1])
        b = list(b_half) + list(b_half[::-1])
        
        # Final check and normalization to ensure exact sums
        sa = sum(a)
        sb = sum(b)
        if sa != 0:
            a = [x / sa for x in a]
        if sb != 0:
            b = [x / sb for x in b]
            
        # Ensure finite
        if not all(np.isfinite(x) for x in a) or not all(np.isfinite(x) for x in b):
            return [1.0/12.0]*12, [1.0/12.0]*12
            
        return a, b

    except Exception as e:
        # Fallback to uniform if anything goes wrong
        return [1.0/12.0]*12, [1.0/12.0]*12
