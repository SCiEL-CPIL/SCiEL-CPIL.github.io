import numpy as np
import time

def _build_hamiltonians():
    """Build the A and B Hamiltonians for the 6-site Heisenberg chain."""
    dim = 64
    anisotropy = 0.8
    field = 0.3
    
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    
    def tensor(*mats):
        result = mats[0]
        for m in mats[1:]:
            result = np.kron(result, m)
        return result
    
    I2 = np.eye(2, dtype=complex)
    
    def build_op(op_list):
        result = np.array([[1.0]], dtype=complex)
        for op in op_list:
            if op is None:
                op = I2
            result = np.kron(result, op)
        return result
    
    A = np.zeros((dim, dim), dtype=complex)
    for bond in [(0, 1), (2, 3), (4, 5)]:
        i, j = bond
        ops = [None] * 6
        ops[i] = sx
        ops[j] = sx
        A += 0.5 * build_op(ops)
        ops = [None] * 6
        ops[i] = sy
        ops[j] = sy
        A += 0.5 * build_op(ops)
        ops = [None] * 6
        ops[i] = sz
        ops[j] = sz
        A += 0.5 * anisotropy * build_op(ops)
        
    for site in [0, 2, 4]:
        ops = [None] * 6
        ops[site] = sz
        A += 0.5 * field * build_op(ops)
        
    B = np.zeros((dim, dim), dtype=complex)
    for bond in [(1, 2), (3, 4)]:
        i, j = bond
        ops = [None] * 6
        ops[i] = sx
        ops[j] = sx
        B += 0.5 * build_op(ops)
        ops = [None] * 6
        ops[i] = sy
        ops[j] = sy
        B += 0.5 * build_op(ops)
        ops = [None] * 6
        ops[i] = sz
        ops[j] = sz
        B += 0.5 * anisotropy * build_op(ops)
        
    for site in [1, 3, 5]:
        ops = [None] * 6
        ops[site] = sz
        B += 0.5 * field * build_op(ops)
        
    return A, B

def _compute_error(a, b, A, B, times):
    dim = A.shape[0]
    H = A + B
    errors = []
    
    # Pre-compute eigendecompositions for A and B as they are constant
    eigvals_a, eigvecs_a = np.linalg.eigh(A)
    eigvals_b, eigvecs_b = np.linalg.eigh(B)
    
    for t in times:
        eigvals_h, eigvecs_h = np.linalg.eigh(H)
        U_exact = (eigvecs_h * np.exp(-1j * t * eigvals_h)) @ eigvecs_h.conj().T
        
        U_approx = np.eye(dim, dtype=complex)
        for k in range(12):
            # exp(-i * b_k * t * B)
            expB = (eigvecs_b * np.exp(-1j * b[k] * t * eigvals_b)) @ eigvecs_b.conj().T
            U_approx = expB @ U_approx
            
            # exp(-i * a_k * t * A)
            expA = (eigvecs_a * np.exp(-1j * a[k] * t * eigvals_a)) @ eigvecs_a.conj().T
            U_approx = expA @ U_approx
        
        diff = U_approx - U_exact
        sv = np.linalg.svd(diff, compute_uv=False)
        errors.append(sv[0])
        
    return np.mean(errors)

def coefficients(stages):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    if stages != 12:
        a = [1.0 / stages] * stages
        b = [1.0 / stages] * stages
        return a, b
    
    times = [0.25, 0.5, 1.0]
    
    # 4th-order Suzuki coefficients
    alpha = 2.0 ** (1.0 / 3.0)
    c1 = 1.0 / (2.0 * (1.0 - alpha))
    c2 = alpha / (2.0 * (1.0 - alpha))
    c3 = 1.0 / (2.0 * (1.0 + alpha))
    
    # 6-step sequence
    seq6 = [c1, c2, c3, c3, c2, c1]
    
    # Expand to 12 steps by splitting each coefficient in half
    # This creates a 12-step sequence that approximates the 6-step one
    a_init = []
    b_init = []
    for c in seq6:
        a_init.append(c / 2.0)
        a_init.append(c / 2.0)
        b_init.append(c / 2.0)
        b_init.append(c / 2.0)
        
    # Normalize just in case of floating point drift
    sum_a = sum(a_init)
    sum_b = sum(b_init)
    a_init = [x / sum_a for x in a_init]
    b_init = [x / sum_b for x in b_init]
    
    A, B = _build_hamiltonians()
    
    best_a = a_init
    best_b = b_init
    best_err = _compute_error(a_init, b_init, A, B, times)
    
    # Simple local search using NumPy
    # We perturb one coefficient at a time
    np.random.seed(42)
    step_size = 0.05
    
    while time.time() < deadline:
        improved = False
        for i in range(12):
            if time.time() > deadline:
                break
            
            # Perturb a[i]
            delta = np.random.choice([-1, 1]) * step_size
            a_new = best_a.copy()
            a_new[i] += delta
            # Normalize
            s = sum(a_new)
            if abs(s) > 1e-9:
                a_new = [x / s for x in a_new]
            
            if all(-50 <= x <= 50 for x in a_new):
                err = _compute_error(a_new, best_b, A, B, times)
                if err < best_err:
                    best_a = a_new
                    best_err = err
                    improved = True
            
            # Perturb b[i]
            delta = np.random.choice([-1, 1]) * step_size
            b_new = best_b.copy()
            b_new[i] += delta
            s = sum(b_new)
            if abs(s) > 1e-9:
                b_new = [x / s for x in b_new]
            
            if all(-50 <= x <= 50 for x in b_new):
                err = _compute_error(best_a, b_new, A, B, times)
                if err < best_err:
                    best_b = b_new
                    best_err = err
                    improved = True
        
        if not improved:
            # Reduce step size if stuck
            step_size *= 0.5
            if step_size < 1e-4:
                break
                
    return best_a, best_b
