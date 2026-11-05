import numpy as np
import time
import scipy.linalg
import scipy.optimize

# Global cache for Hamiltonians to avoid rebuilding on every call
_HAMILTONIANS_CACHE = {}

def _build_hamiltonians():
    """
    Builds the A and B matrices for the 6-site XXZ Heisenberg chain.
    A: Odd bonds + half field
    B: Even bonds + half field
    """
    if 'A' in _HAMILTONIANS_CACHE:
        return _HAMILTONIANS_CACHE['A'], _HAMILTONIANS_CACHE['B']

    N = 6
    dim = 2 ** N
    
    # Spin operators
    sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    
    # Helper to build tensor product operator
    def build_op(op_list):
        op = np.array([[1]], dtype=complex)
        for o in op_list:
            op = np.kron(op, o)
        return op

    # Identity
    I = np.eye(2, dtype=complex)
    
    anisotropy = 0.8
    field = 0.3
    
    H_odd = np.zeros((dim, dim), dtype=complex)
    H_even = np.zeros((dim, dim), dtype=complex)
    H_field = np.zeros((dim, dim), dtype=complex)
    
    # Bonds: 1-2, 3-4, 5-6 are odd (indices 0-1, 2-3, 4-5)
    # Bonds: 2-3, 4-5 are even (indices 1-2, 3-4)
    # Note: Python 0-indexed. Sites 0..5.
    # Odd bonds: (0,1), (2,3), (4,5)
    # Even bonds: (1,2), (3,4)
    
    for j in range(N - 1):
        # Build S^x_j S^x_{j+1} + S^y_j S^y_{j+1} + delta S^z_j S^z_{j+1}
        # We need to place operators at sites j and j+1
        
        def get_bond_op(j, k):
            # Create list of operators for N sites
            ops = [I] * N
            # We need sx, sy, sz at j and k
            # Since we need sums, we build each term separately or combine
            # Term 1: sx_j sx_k
            ops1 = [I]*N
            ops1[j] = sx
            ops1[k] = sx
            op1 = build_op(ops1)
            
            # Term 2: sy_j sy_k
            ops2 = [I]*N
            ops2[j] = sy
            ops2[k] = sy
            op2 = build_op(ops2)
            
            # Term 3: sz_j sz_k
            ops3 = [I]*N
            ops3[j] = sz
            ops3[k] = sz
            op3 = build_op(ops3)
            
            return op1 + op2 + anisotropy * op3

        bond_op = get_bond_op(j, j+1)
        
        if j % 2 == 0:
            # j=0 (0-1), j=2 (2-3), j=4 (4-5) -> Odd
            H_odd += bond_op
        else:
            # j=1 (1-2), j=3 (3-4) -> Even
            H_even += bond_op
            
        # Field term: h * S^z_j
        # "each including half of the uniform field"
        # Total field is h * sum S^z_j.
        # A gets 0.5 * h * sum S^z_j.
        # B gets 0.5 * h * sum S^z_j.
        # So we just add 0.5 * h * S^z_j to both?
        # The prompt says "A and B are ... halves ... each including half of the uniform field".
        # This implies A = H_odd + 0.5 H_field_total and B = H_even + 0.5 H_field_total.
        # So we add 0.5 * h * S^z_j to H_odd and H_even? No, that would double count if we sum them.
        # A+B = H_odd + H_even + H_field_total.
        # If A = H_odd + 0.5 H_field and B = H_even + 0.5 H_field, then A+B is correct.
        
        # So we accumulate H_field separately and add 0.5 to each.
        
    # Accumulate field
    for j in range(N):
        ops = [I] * N
        ops[j] = sz
        H_field += field * build_op(ops)
        
    A = H_odd + 0.5 * H_field
    B = H_even + 0.5 * H_field
    
    _HAMILTONIANS_CACHE['A'] = A
    _HAMILTONIANS_CACHE['B'] = B
    return A, B

def _compute_error(params, A, B, times):
    """
    params: 10 free parameters (6 a's, 6 b's) for symmetric 12-stage splitting.
    """
    # Reconstruct a and b
    # Symmetry: a_k = a_{13-k}.
    # Free a's: a_1..a_6.
    # a_7=a_6, a_8=a_5, a_9=a_4, a_10=a_3, a_11=a_2, a_12=a_1.
    # Same for b.
    
    a_free = params[:6]
    b_free = params[6:]
    
    # Enforce sum constraint: sum(a) = 1, sum(b) = 1.
    # Current sum of symmetric expansion:
    # sum(a) = 2 * sum(a_free)
    # We need sum(a) = 1 => sum(a_free) = 0.5.
    # This is a linear constraint. It's easier to optimize over 5 free params + 1 determined?
    # Or just penalize deviation?
    # Better: Use a transformation or just optimize 10 params and project?
    # Actually, L-BFGS-B doesn't handle linear equality constraints well directly.
    # We can fix a_6 = 0.5 - sum(a_1..a_5) and b_6 = 0.5 - sum(b_1..b_5).
    # This leaves 9 free parameters.
    
    # Let's restart with 9 params: a1..a5, b1..b5.
    # a6 = 0.5 - (a1+...+a5)
    # b6 = 0.5 - (b1+...+b5)
    
    # But the function signature inside minimize needs to match.
    # I will define the inner function to take 9 params.
    
    a1, a2, a3, a4, a5 = params[:5]
    b1, b2, b3, b4, b5 = params[5:]
    
    a6 = 0.5 - (a1 + a2 + a3 + a4 + a5)
    b6 = 0.5 - (b1 + b2 + b3 + b4 + b5)
    
    a = [a1, a2, a3, a4, a5, a6, a6, a5, a4, a3, a2, a1]
    b = [b1, b2, b3, b4, b5, b6, b6, b5, b4, b3, b2, b1]
    
    # Check bounds [-50, 50]
    # If any out of bounds, return large penalty
    if any(x < -50 or x > 50 for x in a) or any(x < -50 or x > 50 for x in b):
        return 1e10
    
    total_error = 0.0
    for t in times:
        # Exact
        H_total = A + B
        U_exact = scipy.linalg.expm(-1j * t * H_total)
        
        # Product
        U_prod = np.eye(64, dtype=complex)
        # Order k=1..12
        # U = M_12 ... M_1
        # M_k = exp(-i b_k t B) exp(-i a_k t A)
        # So we apply M_1 first to identity?
        # Matrix multiplication: U = M_12 @ M_11 @ ... @ M_1
        # Let's build it from right to left (M_1 is rightmost)
        # Or just iterate k=1 to 12 and do U = M_k @ U?
        # If U starts as I, and we do U = M_1 @ U, then U = M_1.
        # Then U = M_2 @ M_1.
        # This matches prod_{k=1}^12 M_k.
        
        U_curr = np.eye(64, dtype=complex)
        for k in range(12):
            exp_B = scipy.linalg.expm(-1j * b[k] * t * B)
            exp_A = scipy.linalg.expm(-1j * a[k] * t * A)
            M_k = exp_B @ exp_A
            U_curr = M_k @ U_curr
            
        # Error
        diff = U_curr - U_exact
        # Spectral norm
        # Use singular values
        s = scipy.linalg.svd(diff, compute_uv=False)
        total_error += s[0]
        
    return total_error / len(times)

def coefficients(stages):
    if stages != 12:
        # Fallback
        substeps = max(1, stages // 2)
        a = [0.5 / substeps] * (2 * substeps)
        b = []
        for _ in range(substeps):
            b += [1.0 / substeps, 0.0]
        while len(a) < stages:
            a.append(0.0)
            b.append(0.0)
        return a, b

    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin
    
    A, B = _build_hamiltonians()
    times = [0.25, 0.5, 1.0]
    
    # Initial guess: 4th order symmetric coefficients
    w = 1.0 / (2.0 * (2.0 - 2.0 ** (1.0 / 3.0)))
    a_sub = [w / 3.0, (0.5 - w) / 3.0, (0.5 - w) / 3.0, w / 3.0]
    b_sub = [(0.5 - w) / 3.0, w / 3.0, w / 3.0, (0.5 - w) / 3.0]
    
    a_init = []
    b_init = []
    for _ in range(3):
        a_init.extend(a_sub)
        b_init.extend(b_sub)
        
    # Extract free params from init
    a_init = np.array(a_init)
    b_init = np.array(b_init)
    
    # a1..a5
    x0 = np.array([a_init[0], a_init[1], a_init[2], a_init[3], a_init[4],
                   b_init[0], b_init[1], b_init[2], b_init[3], b_init[4]])
    
    # Bounds for free params
    # a_i, b_i in [-50, 50]
    bounds = [(-50, 50)] * 10
    
    best_params = x0
    best_error = _compute_error(x0, A, B, times)
    
    # Optimization
    try:
        # Use L-BFGS-B
        result = scipy.optimize.minimize(
            _compute_error,
            x0,
            args=(A, B, times),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if result.fun < best_error:
            best_error = result.fun
            best_params = result.x
            
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 10:
        # Return best found
        pass
    else:
        # Try Nelder-Mead for local refinement if time permits
        try:
            result2 = scipy.optimize.minimize(
                _compute_error,
                best_params,
                args=(A, B, times),
                method='Nelder-Mead',
                options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            if result2.fun < best_error:
                best_error = result2.fun
                best_params = result2.x
        except Exception:
            pass

    # Reconstruct a and b from best_params
    a1, a2, a3, a4, a5 = best_params[:5]
    b1, b2, b3, b4, b5 = best_params[5:]
    
    a6 = 0.5 - (a1 + a2 + a3 + a4 + a5)
    b6 = 0.5 - (b1 + b2 + b3 + b4 + b5)
    
    a = [a1, a2, a3, a4, a5, a6, a6, a5, a4, a3, a2, a1]
    b = [b1, b2, b3, b4, b5, b6, b6, b5, b4, b3, b2, b1]
    
    # Final sanity check
    if not all(np.isfinite(x) for x in a) or not all(np.isfinite(x) for x in b):
        # Fallback to initial
        a = a_init.tolist()
        b = b_init.tolist()
        
    return a, b
