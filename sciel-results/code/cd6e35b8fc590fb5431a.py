"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _enforce_symmetry(cell, size):
    """Enforce 4-fold rotational and mirror symmetry about center."""
    sym = np.zeros((size, size), dtype=int)
    for i in range(size // 2 + 1):
        for j in range(size // 2 + 1):
            val = cell[i][j]
            sym[i][j] = val
            sym[size - 1 - i][j] = val
            sym[i][size - 1 - j] = val
            sym[size - 1 - i][size - 1 - j] = val
    return sym

def _score_cell(cell, size, eps_high=12.0, Npw=15, n_kpoints=30):
    """Score a cell using plane-wave method with specified basis size."""
    half = (Npw - 1) // 2
    eps = np.array(cell, dtype=float)
    eps_inv = np.zeros_like(eps)
    eps_inv[eps > 0.5] = 1.0 / eps_high
    eps_inv[eps <= 0.5] = 1.0
    
    G_vectors = []
    for gx in range(-half, half + 1):
        for gy in range(-half, half + 1):
            G_vectors.append((gx, gy))
    N = len(G_vectors)
    
    # Compute epsilon^{-1} Fourier coefficients using FFT for speed
    eps_F = np.zeros((N, N), dtype=complex)
    for idx_g, (gx, gy) in enumerate(G_vectors):
        for idx_h, (hx, hy) in enumerate(G_vectors):
            dgx = hx - gx
            dgy = hy - gy
            if dgx == 0 and dgy == 0:
                eps_F[idx_g][idx_h] = np.mean(eps_inv)
            else:
                # Use FFT-based computation
                # e^{2pi i (dgx * x + dgy * y) / size}
                x = np.arange(size)
                phase_x = np.exp(2j * np.pi * dgx * x / size)
                phase_y = np.exp(2j * np.pi * dgy * x / size)
                s = np.sum(eps_inv * np.outer(phase_y, phase_x))
                eps_F[idx_g][idx_h] = s / (size * size)
    
    # Define wave vectors in irreducible BZ
    # Square lattice: irreducible wedge is 0 <= ky <= kx, kx in [0, 0.5]
    k_points = []
    # Generate points along boundaries and interior
    for i in range(n_kpoints):
        t = i / (n_kpoints - 1)
        # Mix of boundary and interior points
        if i % 3 == 0:
            # Gamma-X line
            k_points.append((0.5 * t, 0.0))
        elif i % 3 == 1:
            # Gamma-M line  
            k_points.append((0.5 * t, 0.5 * t))
        else:
            # X-M line
            k_points.append((0.5, 0.5 * t))
    
    min_omega5 = float('inf')
    max_omega4 = 0.0
    
    for (kx, ky) in k_points:
        Kx = 2 * np.pi * kx
        Ky = 2 * np.pi * ky
        
        # Build TM eigenvalue matrix
        # For TM: A[G,G'] = (G+K)^2 * eps^{-1}_{G-G'}
        A = np.zeros((N, N), dtype=complex)
        for idx_g, (gx, gy) in enumerate(G_vectors):
            kgx = gx + kx
            kgy = gy + ky
            kg2 = kgx * kgx + kgy * kgy
            for idx_h, (hx, hy) in enumerate(G_vectors):
                A[idx_g][idx_h] = kg2 * eps_F[idx_h][idx_g]
        
        try:
            eigvals = np.linalg.eigvalsh(A)
            eigvals = np.sort(eigvals)
            if len(eigvals) >= 5:
                omega4 = np.sqrt(max(eigvals[3], 0))
                omega5 = np.sqrt(max(eigvals[4], 0))
                max_omega4 = max(max_omega4, omega4)
                min_omega5 = min(min_omega5, omega5)
        except:
            pass
    
    if max_omega4 > 0 and min_omega5 > 0:
        gap = 2.0 * (min_omega5 - max_omega4) / (min_omega5 + max_omega4)
        return gap
    return -1.0

def _chebyshev_field_to_cell(coeffs, size, threshold=0.5):
    """
    Convert Chebyshev coefficients to a binary cell with C4v symmetry.
    coeffs: [c0, c1, c2, ...] where c_k are coefficients for T_k(r) basis
    The field is defined as f(x,y) = sum_k c_k * T_k(r) where r = sqrt(x^2+y^2)
    and T_k is the k-th Chebyshev polynomial of the first kind.
    C4v symmetry is enforced by using r (radial coordinate) which is invariant
    under rotations, and then applying pixel-level symmetry enforcement.
    """
    x = (np.arange(size) + 0.5) / size - 0.5
    y = (np.arange(size) + 0.5) / size - 0.5
    X, Y = np.meshgrid(x, y, indexing='ij')
    r = np.sqrt(X**2 + Y**2)
    
    # Scale r to [-1, 1] for Chebyshev polynomials
    # The maximum r in the unit cell is sqrt(0.5^2 + 0.5^2) = sqrt(0.5) ≈ 0.707
    r_max = np.sqrt(0.5)
    r_scaled = np.clip(r / r_max, -1.0, 1.0)
    
    # Compute Chebyshev polynomials T_k(r_scaled)
    field = np.zeros_like(r_scaled)
    for k, c in enumerate(coeffs):
        if k == 0:
            field += c * 1.0
        elif k == 1:
            field += c * r_scaled
        else:
            # Use recurrence: T_k(x) = 2x*T_{k-1}(x) - T_{k-2}(x)
            T_prev2 = np.ones_like(r_scaled)  # T_0
            T_prev1 = r_scaled.copy()         # T_1
            for m in range(2, k + 1):
                T_curr = 2 * r_scaled * T_prev1 - T_prev2
                T_prev2 = T_prev1
                T_prev1 = T_curr
            field += c * T_prev1
    
    # Threshold to binary
    cell = (field > threshold).astype(int)
    
    # Enforce exact symmetry
    cell = _enforce_symmetry(cell, size)
    return cell

def _random_chebyshev_init(n_coeffs, max_coeff=1.0):
    """Initialize random Chebyshev coefficients."""
    coeffs = np.zeros(n_coeffs)
    for i in range(n_coeffs):
        # Higher order terms should have smaller coefficients
        scale = max_coeff / (1.0 + 0.5 * i)
        coeffs[i] = random.uniform(-scale, scale)
    # Ensure the DC term is reasonable
    coeffs[0] = random.uniform(0.0, 1.0)
    return coeffs

def design(problem):
    """
    Optimize photonic crystal using Chebyshev level-set representation.
    The unit cell is defined by a smooth level-set function in Chebyshev basis,
    constrained to C4v symmetry. We use stochastic hill-climbing with adaptive
    step sizes to optimize the coefficients.
    """
    size = problem["pixels"]
    start_time = time.time()
    deadline = start_time + 110.0  # Leave margin before 120s limit
    
    random.seed(42)
    np.random.seed(42)
    
    n_coeffs = 8  # Number of Chebyshev coefficients
    best_cell = None
    best_score = -1.0
    
    # Initial coefficients: start with a simple radial profile
    def init_coeffs():
        coeffs = np.zeros(n_coeffs)
        # DC term: controls overall fill fraction
        coeffs[0] = random.uniform(0.0, 0.5)
        # First order: controls radial gradient
        coeffs[1] = random.uniform(-0.5, 0.5)
        # Higher orders: smaller perturbations
        for i in range(2, n_coeffs):
            coeffs[i] = random.uniform(-0.2, 0.2) / (i - 1)
        return coeffs
    
    # Try multiple initializations
    n_inits = 4
    for init_i in range(n_inits):
        if time.time() > deadline - 20:
            break
        
        coeffs = init_coeffs()
        cell = _chebyshev_field_to_cell(coeffs, size)
        score = _score_cell(cell, size, Npw=13, n_kpoints=20)
        
        if score > best_score:
            best_score = score
            best_cell = cell.copy()
        
        # Stochastic hill-climbing with adaptive step sizes
        current_coeffs = coeffs.copy()
        current_cell = cell.copy()
        current_score = score
        
        # Adaptive step sizes for each coefficient
        step_sizes = np.ones(n_coeffs) * 0.1
        min_step = 0.001
        max_step = 0.5
        
        max_iter = 300
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            # Pick a random coefficient to perturb
            idx = random.randint(0, n_coeffs - 1)
            delta = random.gauss(0, step_sizes[idx])
            delta = np.clip(delta, -max_step, max_step)
            
            new_coeffs = current_coeffs.copy()
            new_coeffs[idx] += delta
            
            new_cell = _chebyshev_field_to_cell(new_coeffs, size)
            new_score = _score_cell(new_cell, size, Npw=11, n_kpoints=15)
            
            if new_score > current_score:
                # Accept the move
                current_coeffs = new_coeffs
                current_cell = new_cell
                current_score = new_score
                
                # Increase step size slightly (exploration)
                step_sizes[idx] = min(step_sizes[idx] * 1.1, max_step)
                
                if new_score > best_score:
                    best_score = new_score
                    best_cell = new_cell.copy()
            else:
                # Reject the move
                # Decrease step size (exploitation)
                step_sizes[idx] = max(step_sizes[idx] * 0.9, min_step)
        
        # After hill-climbing, do fine-tuning with smaller steps
        current_coeffs = best_cell is not None and _chebyshev_field_to_cell(current_coeffs, size).shape == (size, size) and current_coeffs or coeffs
        
        # Re-initialize from best found so far
        if best_cell is not None:
            # Reverse-engineer approximate coefficients from best cell (simplified: just continue from current)
            pass
    
    # Final refinement: local pixel search around best cell
    if best_cell is not None and time.time() < deadline - 10:
        cell = best_cell.copy()
        unique_positions = []
        for i in range(size // 2 + 1):
            for j in range(size // 2 + 1):
                unique_positions.append((i, j))
        
        for _ in range(500):
            if time.time() > deadline - 5:
                break
            idx = random.randint(0, len(unique_positions) - 1)
            i, j = unique_positions[idx]
            old_val = cell[i][j]
            cell[i][j] = 1 - old_val
            cell = _enforce_symmetry(cell, size)
            score = _score_cell(cell, size, Npw=13, n_kpoints=20)
            if score > best_score:
                best_score = score
                best_cell = cell.copy()
            else:
                cell[i][j] = old_val
                cell = _enforce_symmetry(cell, size)
    
    # Convert to nested list
    if best_cell is None:
        # Fallback: simple circle
        best_cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                if x * x + y * y < 0.2 ** 2:
                    best_cell[i][j] = 1
    
    result = [[int(best_cell[i][j]) for j in range(size)] for i in range(size)]
    return result
# EVOLVE-BLOCK-END
