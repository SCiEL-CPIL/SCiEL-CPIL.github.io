import numpy as np
import time
import sys
import scipy.optimize

def coefficients(stages):
    """
    Compute 12-stage product formula coefficients for Heisenberg chain time evolution.
    Returns (a, b) where each is a list of 12 floats summing to 1.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    # Build the Hamiltonians
    n = 6
    dim = 2 ** n
    delta = 0.8
    h = 0.3

    # Pauli matrices
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)

    def kron_all(ops):
        result = np.array([[1]], dtype=complex)
        for op in ops:
            result = np.kron(result, op)
        return result

    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)

    Jz = 1.0
    Jxy = delta * Jz

    def add_bond(H, i, j):
        ops = [np.eye(2) for _ in range(n)]
        ops[i] = sz
        ops[j] = sz
        H += Jz * kron_all(ops)

        ops = [np.eye(2) for _ in range(n)]
        ops[i] = sx
        ops[j] = sx
        H += (Jxy / 2) * kron_all(ops)

        ops = [np.eye(2) for _ in range(n)]
        ops[i] = sy
        ops[j] = sy
        H += (Jxy / 2) * kron_all(ops)

    # A: Odd bonds (0,1), (2,3), (4,5) + half field
    # B: Even bonds (1,2), (3,4) + half field
    # Note: The problem statement says "A and B are the odd- and even-bond halves (each including half of the uniform field)"
    # So A has bonds (0,1), (2,3), (4,5) and h/2 * sum sz
    # B has bonds (1,2), (3,4) and h/2 * sum sz
    
    for (i, j) in [(0, 1), (2, 3), (4, 5)]:
        add_bond(A, i, j)

    for (i, j) in [(1, 2), (3, 4)]:
        add_bond(B, i, j)

    for i in range(n):
        ops = [np.eye(2) for _ in range(n)]
        ops[i] = sz
        field_term = h * 0.5 * kron_all(ops)
        A += field_term
        B += field_term

    # Precompute eigendecompositions for A, B, and A+B
    vals_A, vecs_A = np.linalg.eigh(A)
    vals_B, vecs_B = np.linalg.eigh(B)
    vals_AB, vecs_AB = np.linalg.eigh(A + B)

    times = [0.25, 0.5, 1.0]

    def expm_A(t):
        """exp(-i * t * A)"""
        return vecs_A @ np.diag(np.exp(-1j * t * vals_A)) @ vecs_A.conj().T

    def expm_B(t):
        """exp(-i * t * B)"""
        return vecs_B @ np.diag(np.exp(-1j * t * vals_B)) @ vecs_B.conj().T

    def expm_AB(t):
        """exp(-i * t * (A+B))"""
        return vecs_AB @ np.diag(np.exp(-1j * t * vals_AB)) @ vecs_AB.conj().T

    def compute_error_full(a_full, b_full, t):
        """Compute spectral norm of U(t) - exp(-i t (A+B))"""
        U = np.eye(dim, dtype=complex)
        # Product formula: U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A) in order k=1..12
        # Note: Matrix multiplication order. 
        # If we apply operator O1 then O2, the matrix is O2 @ O1.
        # The product is usually written U = U_12 ... U_1.
        # So U = (exp(-i b_12 t B) exp(-i a_12 t A)) @ ... @ (exp(-i b_1 t B) exp(-i a_1 t A))
        # In the code, if we do U = step_k @ U, starting with U=I:
        # U = step_1 @ I = step_1
        # U = step_2 @ step_1
        # ...
        # U = step_12 @ ... @ step_1
        # This matches the standard convention.
        
        for k in range(len(a_full)):
            step = expm_B(b_full[k] * t) @ expm_A(a_full[k] * t)
            U = step @ U
        U_exact = expm_AB(t)
        diff = U - U_exact
        s = np.linalg.svd(diff, compute_uv=False)
        return s[0]

    def objective_sym(params):
        """
        params: 12-element array.
        First 6 are a_1..a_6, last 6 are b_1..b_6.
        We construct the full 12-element vectors with symmetry:
        a = [a1, a2, a3, a4, a5, a6, a6, a5, a4, a3, a2, a1]
        b = [b1, b2, b3, b4, b5, b6, b6, b5, b4, b3, b2, b1]
        Then normalize them to sum to 1.
        """
        if time.time() > deadline:
            return 1e10

        a_half = params[:6]
        b_half = params[6:12]

        # Construct full vectors
        a_full = np.concatenate([a_half, a_half[::-1]])
        b_full = np.concatenate([b_half, b_half[::-1]])

        # Normalize
        a_sum = np.sum(a_full)
        b_sum = np.sum(b_full)
        
        # Handle edge cases where sum is close to 0
        if abs(a_sum) < 1e-12:
            a_full = np.ones(12) / 12.0
        else:
            a_full = a_full / a_sum
            
        if abs(b_sum) < 1e-12:
            b_full = np.ones(12) / 12.0
        else:
            b_full = b_full / b_sum

        total_err = 0.0
        for t in times:
            total_err += compute_error_full(a_full, b_full, t)
        return total_err / len(times)

    # Initial guess: uniform
    # If a_full is uniform, a_half must be uniform too.
    # sum(a_full) = 2 * sum(a_half) = 1 => sum(a_half) = 0.5
    # So a_i = 0.5/6 = 1/12
    x0 = np.ones(12) / 12.0
    best_x = x0.copy()
    best_obj = objective_sym(x0)
    
    # Optimization
    # We will use scipy.optimize.minimize with Powell method, which is robust for this.
    # We will do a few restarts.
    
    np.random.seed(42)
    
    # Define a wrapper for minimize that respects the deadline
    def run_optimization(x_start, maxiter=200):
        try:
            # Powell method is good for this.
            # We don't use constraints in minimize, we handle normalization in objective.
            res = scipy.optimize.minimize(
                objective_sym, 
                x_start, 
                method='Powell', 
                options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            return res.x, res.fun
        except Exception:
            return x_start, objective_sym(x_start)

    # Run initial optimization
    res_x, res_f = run_optimization(best_x, maxiter=300)
    if res_f < best_obj:
        best_obj = res_f
        best_x = res_x

    # Random restarts
    n_restarts = 0
    max_restarts = 10
    while time.time() < deadline - 10:
        n_restarts += 1
        # Perturb the best solution
        # Scale of perturbation: 0.1
        x_rand = best_x + 0.1 * np.random.randn(12)
        
        # Also try some structured initial guesses
        if n_restarts % 3 == 0:
            # Try a guess based on known 4th order Suzuki-Trotter coefficients scaled
            # 4th order coefficients for 2-stage:
            # a1 = 1/(2-2^(1/3)), a2 = -2^(1/3)/(2-2^(1/3))
            # This is for 2 stages. For 6 stages (3 pairs), we can just use random or scaled.
            pass

        res_x, res_f = run_optimization(x_rand, maxiter=100)
        if res_f < best_obj:
            best_obj = res_f
            best_x = res_x

    # Extract final coefficients
    a_half = best_x[:6]
    b_half = best_x[6:12]
    
    a_full = np.concatenate([a_half, a_half[::-1]])
    b_full = np.concatenate([b_half, b_half[::-1]])

    # Final normalization and clipping
    a_full = np.clip(a_full, -50, 50)
    b_full = np.clip(b_full, -50, 50)
    
    a_sum = np.sum(a_full)
    b_sum = np.sum(b_full)
    
    if abs(a_sum) > 1e-12:
        a_full = a_full / a_sum
    else:
        a_full = np.ones(12) / 12.0
        
    if abs(b_sum) > 1e-12:
        b_full = b_full / b_sum
    else:
        b_full = np.ones(12) / 12.0

    return [float(x) for x in a_full], [float(x) for x in b_full]
