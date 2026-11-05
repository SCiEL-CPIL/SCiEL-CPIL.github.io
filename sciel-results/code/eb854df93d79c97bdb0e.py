"""Product-formula coefficients for 6-site Heisenberg chain time evolution.

Strategy:
  1. Build the exact Hamiltonian H = A + B for a 6-site open XXZ chain
     with anisotropy Δ=0.8, field h=0.3, split into odd/even bond halves.
  2. Use a Fourier-series parameterization for the coefficients to enforce
     smoothness and symmetry, reducing the search space.
  3. Run a custom optimization (random search + coordinate descent) to
     minimize the mean spectral-norm error over t = {0.25, 0.5, 1.0}.
  4. Return the best a, b found, with a deadline guard.
"""
import time
import numpy as np

# ---- Build the Hamiltonian pieces A and B ----
def _build_H():
    """Build A (odd bonds + half field) and B (even bonds + half field) for
    a 6-site open XXZ chain with Δ=0.8, h=0.3."""
    N = 6
    d = 2 ** N  # 64
    delta = 0.8
    h = 0.3

    # Single-spin operators
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)

    def kron_chain(ops_list):
        """Kron product of a list of 2x2 matrices (left to right)."""
        result = np.array([[1.0+0j]], dtype=complex)
        for op in ops_list:
            result = np.kron(result, op)
        return result

    def two_site_op(site_a, site_b, op_a, op_b):
        """Kron product with op_a at site_a, op_b at site_b, identity elsewhere."""
        ops = [I2] * N
        ops[site_a] = op_a
        ops[site_b] = op_b
        return kron_chain(ops)

    # XXZ bond term: (1/2)(Sx⊗Sx + Sy⊗Sy + Δ Sz⊗Sz)
    # where S = σ/2
    Sx = sx / 2.0
    Sy = sy / 2.0
    Sz = sz / 2.0

    A = np.zeros((d, d), dtype=complex)
    B = np.zeros((d, d), dtype=complex)

    # Odd bonds: sites (0,1), (2,3), (4,5) -> A
    # Even bonds: sites (1,2), (3,4) -> B
    for (i, j) in [(0, 1), (2, 3), (4, 5)]:
        A += 0.5 * (two_site_op(i, j, Sx, Sx) +
                    two_site_op(i, j, Sy, Sy) +
                    delta * two_site_op(i, j, Sz, Sz))
    for (i, j) in [(1, 2), (3, 4)]:
        B += 0.5 * (two_site_op(i, j, Sx, Sx) +
                    two_site_op(i, j, Sy, Sy) +
                    delta * two_site_op(i, j, Sz, Sz))

    # Uniform field: h * sum Sz_j  -> split evenly between A and B
    field = np.zeros((d, d), dtype=complex)
    for j in range(N):
        field += two_site_op(j, j, Sz, I2)  # Sz at site j
    A += 0.5 * h * field
    B += 0.5 * h * field

    return A, B


def _matrix_exp(M):
    """Compute matrix exponential of a square matrix M using eigendecomposition.
    Assumes M is diagonalizable (which is true for Hermitian matrices)."""
    w, v = np.linalg.eig(M)
    # M = v * diag(w) * v^-1
    # exp(M) = v * diag(exp(w)) * v^-1
    # For Hermitian M, v is unitary, so v^-1 = v.conj().T
    # However, np.linalg.eig does not guarantee orthogonality for non-Hermitian.
    # A and B are Hermitian, so we can use this.
    # To be safe against numerical issues with non-normal matrices (though ours are normal),
    # we use the standard inverse.
    v_inv = np.linalg.inv(v)
    return v @ np.diag(np.exp(w)) @ v_inv


def _spectral_norm(M):
    """Compute spectral norm (largest singular value) of a matrix."""
    return np.linalg.norm(M, 2)


def _fourier_coeffs_to_ab(fourier_params, stages=12):
    """Convert Fourier parameters to coefficient vectors a and b.
    
    We use a truncated Fourier series to generate smooth, symmetric coefficients.
    For a symmetric sequence of length N=12, we can represent it as:
    c_k = c_0 + sum_{m=1}^{M} (A_m cos(2*pi*m*k/N) + B_m sin(2*pi*m*k/N))
    
    However, to enforce sum(c) = 1 and symmetry c_k = c_{N-1-k}, 
    we can simplify. A symmetric sequence can be represented by its first half.
    Alternatively, we just use a few cosine terms which are inherently symmetric 
    if centered properly.
    
    Let's use a simpler parameterization:
    a_k = base_a + amp_a1 * cos(2*pi*k/12 + phase_a1) + amp_a2 * cos(2*pi*2*k/12 + phase_a2)
    b_k = base_b + amp_b1 * cos(2*pi*k/12 + phase_b1) + amp_b2 * cos(2*pi*2*k/12 + phase_b2)
    
    Then we normalize so sum(a)=1 and sum(b)=1.
    Parameters: [base_a, amp_a1, phase_a1, amp_a2, phase_a2, base_b, amp_b1, phase_b1, amp_b2, phase_b2]
    Total 10 parameters.
    """
    k = np.arange(stages)
    
    # Unpack params
    base_a = fourier_params[0]
    amp_a1 = fourier_params[1]
    phase_a1 = fourier_params[2]
    amp_a2 = fourier_params[3]
    phase_a2 = fourier_params[4]
    
    base_b = fourier_params[5]
    amp_b1 = fourier_params[6]
    phase_b1 = fourier_params[7]
    amp_b2 = fourier_params[8]
    phase_b2 = fourier_params[9]
    
    # Generate raw coefficients
    a_raw = base_a + amp_a1 * np.cos(2*np.pi*k/stages + phase_a1) + amp_a2 * np.cos(2*np.pi*2*k/stages + phase_a2)
    b_raw = base_b + amp_b1 * np.cos(2*np.pi*k/stages + phase_b1) + amp_b2 * np.cos(2*np.pi*2*k/stages + phase_b2)
    
    # Normalize to sum 1
    sum_a = np.sum(a_raw)
    sum_b = np.sum(b_raw)
    
    if abs(sum_a) < 1e-15:
        a_raw = np.ones(stages) / stages
    else:
        a_raw = a_raw / sum_a
        
    if abs(sum_b) < 1e-15:
        b_raw = np.ones(stages) / stages
    else:
        b_raw = b_raw / sum_b
        
    # Clip to [-50, 50] and re-normalize
    a_raw = np.clip(a_raw, -50, 50)
    b_raw = np.clip(b_raw, -50, 50)
    
    sum_a = np.sum(a_raw)
    sum_b = np.sum(b_raw)
    
    if abs(sum_a) < 1e-15:
        a_raw = np.ones(stages) / stages
    else:
        a_raw = a_raw / sum_a
        
    if abs(sum_b) < 1e-15:
        b_raw = np.ones(stages) / stages
    else:
        b_raw = b_raw / sum_b
        
    return a_raw, b_raw


def _evaluate_ab(a, b, A, B, t_values):
    """Calculate mean spectral norm error for given a, b."""
    total_err = 0.0
    N = len(a)
    for t in t_values:
        U = np.eye(64, dtype=complex)
        for k in range(N):
            # U = exp(-i b_k t B) U exp(-i a_k t A) ? 
            # Contract says: U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A) in order k=1..12
            # Usually product formula means U = U_1 U_2 ... U_12
            # Let's stick to the contract: "in order k = 1..12"
            # This implies U = U_12 ... U_1 if we apply to a vector from left?
            # Or U = U_1 U_2 ... U_12?
            # Standard physics notation: U(t) = U_N ... U_1 means U_1 is applied first.
            # But the contract says "prod_k ... in order k=1..12".
            # Let's assume standard matrix multiplication order: U = U_1 @ U_2 @ ... @ U_12
            # where U_k = exp(-i b_k t B) exp(-i a_k t A)
            
            U_b = _matrix_exp(-1j * b[k] * t * B)
            U_a = _matrix_exp(-1j * a[k] * t * A)
            U = U @ (U_b @ U_a)
            
        U_exact = _matrix_exp(-1j * t * (A + B))
        err = _spectral_norm(U - U_exact)
        total_err += err
    return total_err / len(t_values)


def coefficients(stages):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Build Hamiltonian
    A, B = _build_H()
    
    if stages != 12:
        a = [1.0 / stages] * stages
        b = [1.0 / stages] * stages
        return a, b
    
    t_values = [0.25, 0.5, 1.0]
    
    # Initial best: uniform split
    best_a = np.ones(12) / 12
    best_b = np.ones(12) / 12
    best_score = _evaluate_ab(best_a, best_b, A, B, t_values)
    
    # Fourier parameters: [base_a, amp_a1, phase_a1, amp_a2, phase_a2, base_b, amp_b1, phase_b1, amp_b2, phase_b2]
    # We will perform a simple random search + coordinate descent on these 10 parameters.
    
    num_params = 10
    best_params = np.zeros(num_params)
    
    # Seed random
    np.random.seed(42)
    
    # Define search ranges
    # base should be around 1/12 ~ 0.08
    # amps should be small, phases arbitrary
    bounds = [
        (0.0, 0.2),   # base_a
        (-0.1, 0.1),  # amp_a1
        (0.0, 2*np.pi), # phase_a1
        (-0.05, 0.05), # amp_a2
        (0.0, 2*np.pi), # phase_a2
        (0.0, 0.2),   # base_b
        (-0.1, 0.1),  # amp_b1
        (0.0, 2*np.pi), # phase_b1
        (-0.05, 0.05), # amp_b2
        (0.0, 2*np.pi)  # phase_b2
    ]
    
    # Random search with local refinement
    max_iters = 200 # Adjust based on time
    
    for i in range(max_iters):
        if time.time() > deadline - 10.0:
            break
            
        # Generate random candidate
        params = np.array([np.random.uniform(low, high) for low, high in bounds])
        
        a, b = _fourier_coeffs_to_ab(params, stages)
        score = _evaluate_ab(a, b, A, B, t_values)
        
        if score < best_score:
            best_score = score
            best_params = params.copy()
            best_a = a.copy()
            best_b = b.copy()
            
            # Local refinement: coordinate descent
            step_sizes = [0.05, 0.05, 0.5, 0.02, 0.5, 0.05, 0.05, 0.5, 0.02, 0.5]
            for _ in range(5): # Few steps of refinement
                if time.time() > deadline - 10.0:
                    break
                improved = False
                for j in range(num_params):
                    # Try +/- step
                    for direction in [1, -1]:
                        new_params = best_params.copy()
                        new_params[j] += direction * step_sizes[j]
                        # Wrap phases
                        if j in [2, 4, 7, 9]:
                            new_params[j] = np.mod(new_params[j], 2*np.pi)
                        
                        a_new, b_new = _fourier_coeffs_to_ab(new_params, stages)
                        score_new = _evaluate_ab(a_new, b_new, A, B, t_values)
                        
                        if score_new < best_score:
                            best_score = score_new
                            best_params = new_params
                            best_a = a_new
                            best_b = b_new
                            improved = True
                            break # Move to next parameter
                    
                    if not improved:
                        # Reduce step size if no improvement in this parameter
                        step_sizes[j] *= 0.5
    
    # Final validation and conversion to list
    a_list = [max(-50.0, min(50.0, float(x))) for x in best_a]
    b_list = [max(-50.0, min(50.0, float(x))) for x in best_b]
    
    # Ensure sums are exactly 1 (floating point correction)
    sum_a = sum(a_list)
    sum_b = sum(b_list)
    if abs(sum_a) > 1e-15:
        a_list = [x / sum_a for x in a_list]
    if abs(sum_b) > 1e-15:
        b_list = [x / sum_b for x in b_list]
        
    return a_list, b_list
