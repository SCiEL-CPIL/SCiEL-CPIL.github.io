"""Photonic crystal unit cell design for maximum TM band gap."""
import numpy as np
from scipy.linalg import eigh
import time

# EVOLVE-BLOCK-START
def design(problem):
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # Deadline guard: must finish within 200s wall-clock, use 190s budget
    t_start = time.time()
    deadline = t_start + 190.0
    
    # Fast coarse-grained evaluator using reduced plane-wave set
    def coarse_evaluate(cell, n_pw=25, k_points=None):
        """Evaluate band gap using coarse plane-wave expansion."""
        if k_points is None:
            k_points = _get_k_points_coarse(n_k=8)
        
        # Compute Fourier coefficients of 1/epsilon for the cell
        # Use downsampled approach for speed
        eps = np.where(np.array(cell, dtype=float) == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps
        
        # Compute Fourier coefficients on a grid
        N = size
        # Use FFT for fast Fourier coefficient computation
        fourier_coeffs = np.fft.fft2(inv_eps) / (N * N)
        
        # Map to plane wave indices
        # Plane waves: G = (mx, my) * 2*pi/L, m,n in range
        # For n_pw plane waves, use |m|,|n| <= n_pw//2
        half = n_pw // 2
        pw_indices = []
        for mx in range(-half, half + 1):
            for my in range(-half, half + 1):
                if mx * mx + my * my <= half * half:
                    pw_indices.append((mx, my))
        
        n_pw_actual = len(pw_indices)
        
        # Build the TM band structure matrices
        # For TM: eigenvalue problem with (k+G)^2 / eps(k+G) 
        # Actually for TM polarization: the eigenvalue equation is
        # sum_G' (k+G)^2 * (1/eps)_{G-G'} * c_{G'} = omega^2 * c_G
        
        best_gap = 0.0
        best_min_5 = 0.0
        best_max_4 = 0.0
        
        for kx, ky in k_points:
            # Build matrix A where A[i,j] = (k+G_i)·(k+G_j) * <1/eps>_{G_i - G_j}
            # Actually for TM: the Hamiltonian is H_{G,G'} = (k+G)^2 * delta_{G,G'} / eps... 
            # No, for TM polarization in 2D: 
            # The eigenvalue equation is: sum_{G'} (k+G)^2 * (1/eps)_{G-G'} * c_{G'} = omega^2 c_G
            # where (1/eps)_{G} are Fourier coefficients of 1/eps
            
            # Build the matrix
            A = np.zeros((n_pw_actual, n_pw_actual), dtype=complex)
            for i, (mx_i, my_i) in enumerate(pw_indices):
                kx_i = kx + 2 * np.pi * mx_i
                ky_i = ky + 2 * np.pi * my_i
                for j, (mx_j, my_j) in enumerate(pw_indices):
                    dmx = mx_i - mx_j
                    dmy = my_i - my_j
                    # Get Fourier coefficient of 1/eps at (dmx, dmy)
                    # fourier_coeffs[dmy + N, dmx + N] if in range
                    idx_x = dmx % N
                    idx_y = dmy % N
                    fc = fourier_coeffs[idx_y, idx_x]
                    A[i, j] = (kx_i * kx_i + ky_i * ky_i) * fc
            
            # Solve eigenvalue problem
            try:
                eigvals = eigh(A, eigvals_only=True)
                eigvals = np.real(eigvals)
                eigvals = np.sort(eigvals)
                
                # We need bands 4 and 5 (0-indexed: 3 and 4)
                if len(eigvals) >= 5:
                    # Take the 4th and 5th bands
                    omega_4 = eigvals[3]  # 4th band (index 3)
                    omega_5 = eigvals[4]  # 5th band (index 4)
                    
                    gap = omega_5 - omega_4
                    if gap > best_gap:
                        best_gap = gap
                        best_min_5 = omega_5
                        best_max_4 = omega_4
            except:
                continue
        
        # Compute gap-midgap ratio
        if best_min_5 + best_max_4 > 0:
            ratio = 2 * best_gap / (best_min_5 + best_max_4)
        else:
            ratio = 0.0
        
        return ratio
    
    def _get_k_points_coarse(n_k=8):
        """Get K points in irreducible BZ for square lattice."""
        # Irreducible BZ: 0 <= kx <= ky, kx >= 0, ky >= 0, kx+ky <= pi
        # Actually for square lattice, irreducible BZ is the triangle with vertices
        # (0,0), (pi,0), (pi/2,pi/2)
        points = []
        # Sample along the edges and interior
        for i in range(n_k + 1):
            # Along kx axis
            kx = np.pi * i / n_k
            points.append((kx, 0.0))
            # Along ky axis  
            points.append((0.0, kx))
            # Along kx=ky line
            if i > 0:
                k_val = np.pi / 2 * i / n_k
                points.append((k_val, k_val))
        # Add some interior points
        for i in range(1, n_k):
            for j in range(1, n_k):
                kx = np.pi * i / n_k
                ky = np.pi * j / n_k
                if kx + ky <= np.pi and kx <= ky:
                    points.append((kx, ky))
        return points
    
    # Parametric design with C4v symmetry
    # Features: central ring, cross arms, corner dots
    def make_cell(params):
        """Create cell from parameters: [ring_inner, ring_outer, cross_width, corner_radius, corner_pos]"""
        ring_inner = params[0]
        ring_outer = params[1]
        cross_width = params[2]
        corner_radius = params[3]
        corner_pos = params[4]
        
        cell = np.zeros((size, size), dtype=int)
        
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                
                # Central ring
                r = np.sqrt(x * x + y * y)
                if ring_inner <= r <= ring_outer:
                    cell[i, j] = 1
                
                # Cross arms (horizontal and vertical)
                if abs(x) <= cross_width / 2 and abs(y) <= 0.5:
                    cell[i, j] = 1
                if abs(y) <= cross_width / 2 and abs(x) <= 0.5:
                    cell[i, j] = 1
                
                # Corner dots (at positions symmetric under C4v)
                # Four corners of a square
                for cx, cy in [(corner_pos, corner_pos), (corner_pos, -corner_pos), 
                               (-corner_pos, corner_pos), (-corner_pos, -corner_pos)]:
                    if (x - cx) ** 2 + (y - cy) ** 2 <= corner_radius ** 2:
                        cell[i, j] = 1
        
        return cell.tolist()
    
    # Local search using coarse evaluator
    def local_search(initial_params, max_iters=50, step=0.02):
        """Simple coordinate descent with coarse evaluation."""
        current_params = list(initial_params)
        current_score = coarse_evaluate(make_cell(current_params))
        
        for _ in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            improved = False
            for dim in range(len(current_params)):
                for direction in [-1, 1]:
                    trial_params = list(current_params)
                    trial_params[dim] += direction * step
                    # Clip to valid ranges
                    trial_params[0] = max(0.0, min(0.4, trial_params[0]))  # ring_inner
                    trial_params[1] = max(0.05, min(0.45, trial_params[1]))  # ring_outer
                    trial_params[2] = max(0.05, min(0.3, trial_params[2]))  # cross_width
                    trial_params[3] = max(0.0, min(0.15, trial_params[3]))  # corner_radius
                    trial_params[4] = max(0.2, min(0.45, trial_params[4]))  # corner_pos
                    
                    if trial_params[0] >= trial_params[1]:
                        continue
                    
                    trial_cell = make_cell(trial_params)
                    trial_score = coarse_evaluate(trial_cell)
                    
                    if trial_score > current_score:
                        current_params = trial_params
                        current_score = trial_score
                        improved = True
                        break
            if not improved:
                # Try smaller step
                step *= 0.7
                if step < 0.005:
                    break
        
        return current_params, current_score
    
    # Try multiple initial configurations
    best_cell = None
    best_score = -1
    
    # Initial guesses for different geometric families
    initial_guesses = [
        [0.1, 0.25, 0.15, 0.0, 0.35],   # Ring + cross
        [0.15, 0.3, 0.1, 0.05, 0.35],    # Ring + cross + corners
        [0.05, 0.2, 0.2, 0.08, 0.3],     # Thin ring + wide cross + corners
        [0.2, 0.35, 0.12, 0.06, 0.38],   # Thick ring + narrow cross + corners
        [0.0, 0.15, 0.25, 0.1, 0.32],    # No ring, wide cross, big corners
    ]
    
    for guess in initial_guesses:
        if time.time() > deadline - 15:
            break
        
        params, score = local_search(guess, max_iters=30, step=0.02)
        
        if score > best_score:
            best_score = score
            best_cell = make_cell(params)
    
    # If no good result, fall back to seed design
    if best_cell is None or best_score < 0:
        cell = []
        for i in range(size):
            x = (i + 0.5) / size - 0.5
            cell.append([1 if x * x + ((j + 0.5) / size - 0.5) ** 2 < 0.2 ** 2 else 0 for j in range(size)])
        return cell
    
    # Final refinement with finer steps
    if time.time() < deadline - 5:
        # Extract params from best_cell is complex, so just do a few more local searches
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            # Perturb and re-search
            perturbed = [p + np.random.uniform(-0.02, 0.02) for p in 
                        [0.15, 0.28, 0.15, 0.06, 0.35]]
            params, score = local_search(perturbed, max_iters=10, step=0.01)
            if score > best_score:
                best_score = score
                best_cell = make_cell(params)
    
    return best_cell
# EVOLVE-BLOCK-END
