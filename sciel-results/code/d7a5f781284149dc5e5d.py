import math
import time
import numpy as np

def design(problem):
    """
    Design a symmetric 32x32-pixel unit cell to maximize the TM band gap
    between bands 4 and 5 using a Fourier-harmonic parameterization with
    local optimization in coefficient space.
    """
    size = problem["pixels"]  # 32
    eps_low = problem["eps_low"]   # 1
    eps_high = problem["eps_high"] # 12
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 115  # 115s out of 120s budget (with margin)
    
    def quadrant_to_full(q):
        """Convert a 16x16 quadrant to a full 32x32 symmetric cell."""
        q = np.array(q, dtype=int)
        top = np.hstack([q, q[:, ::-1]])
        bottom = top[::-1, :]
        return np.vstack([top, bottom])
    
    def full_to_quadrant(cell):
        """Extract the top-left 16x16 quadrant from a full 32x32 cell."""
        return np.array(cell[:16, :16], dtype=int)
    
    def compute_gap(cell_2d):
        """
        Compute the gap-midgap ratio between TM bands 4 and 5.
        Uses a plane-wave expansion with 289 basis functions.
        """
        N = 32
        eps = np.where(np.array(cell_2d) == 1, eps_high, eps_low)
        
        # Use 289 plane waves: (2*8+1)^2 = 17^2 = 289
        npw = 8
        G = []
        for mx in range(-npw, npw+1):
            for my in range(-npw, npw+1):
                G.append((2*np.pi*mx, 2*np.pi*my))
        G = np.array(G)
        nG = len(G)
        
        # Compute inverse dielectric Fourier coefficients
        inv_eps = 1.0 / eps
        x = (np.arange(N) + 0.5) / N
        y = (np.arange(N) + 0.5) / N
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        inv_eps_coeffs = np.zeros(nG, dtype=complex)
        for idx, (gx, gy) in enumerate(G):
            inv_eps_coeffs[idx] = np.sum(inv_eps * np.exp(-1j * (gx * X + gy * Y))) / (N * N)
        
        # Key k-points in the irreducible Brillouin zone
        k_points = [
            (0.0, 0.0),           # Gamma
            (np.pi, 0.0),         # M
            (np.pi/2, np.pi/2),   # X
            (np.pi/2, 0.0),       # Midpoint Gamma-M
            (np.pi/2, np.pi/4),
            (3*np.pi/8, np.pi/8),
            (np.pi/4, np.pi/4),
            (np.pi/3, np.pi/6),
            (2*np.pi/5, np.pi/5),
            (np.pi/2, np.pi/6),
        ]
        
        max_omega4 = 0.0
        min_omega5 = np.inf
        
        for kx, ky in k_points:
            kvec = np.array([kx, ky])
            k_plus_G = kvec + G  # shape (nG, 2)
            
            # Build the eigenvalue matrix for TM polarization
            kG_sq = np.sum(k_plus_G**2, axis=1)  # shape (nG,)
            
            # Build the A matrix: A[G, G'] = (1/eps)_{G-G'}
            G_dict = {}
            for idx, (gx, gy) in enumerate(G):
                G_dict[(gx, gy)] = idx
            
            A = np.zeros((nG, nG), dtype=complex)
            for i in range(nG):
                for j in range(nG):
                    G_diff = G[i] - G[j]
                    key = (G_diff[0], G_diff[1])
                    if key in G_dict:
                        A[i, j] = inv_eps_coeffs[G_dict[key]]
            
            # The eigenvalue matrix is: H = A * diag(kG_sq)
            H = A * kG_sq[np.newaxis, :]
            
            # Make Hermitian
            H = (H + H.conj().T) / 2
            
            # Get eigenvalues
            eigvals = np.linalg.eigvalsh(H)
            eigvals = np.sort(eigvals.real)
            
            if len(eigvals) >= 6:
                omega4 = eigvals[4]
                omega5 = eigvals[5]
                max_omega4 = max(max_omega4, omega4)
                min_omega5 = min(min_omega5, omega5)
        
        if min_omega5 == np.inf or max_omega4 == 0:
            return 0.0
        
        gap = min_omega5 - max_omega4
        midgap = (min_omega5 + max_omega4) / 2
        
        if midgap <= 0:
            return 0.0
        
        return gap / midgap
    
    # Fourier-harmonic parameterization
    # We represent the cell as a thresholded superposition of D4-symmetric harmonics
    # Each harmonic is a real-valued function with specific (mx, my) components
    
    def fourier_harmonic(mx, my, x, y):
        """
        Generate a D4-symmetric harmonic pattern with fundamental (mx, my).
        For D4 symmetry, we need cos/sin combinations that are invariant under
        90-degree rotations and reflections.
        """
        # For (mx, my) = (2, 0): cos(2*2*pi*x) is D4 symmetric
        # For (mx, my) = (0, 2): cos(2*2*pi*y) is D4 symmetric
        # For (mx, my) = (2, 2): cos(2*2*pi*x)*cos(2*2*pi*y) is D4 symmetric
        # For (mx, my) = (4, 0): cos(4*2*pi*x) is D4 symmetric
        
        if mx == 2 and my == 0:
            return np.cos(2 * np.pi * 2 * x)
        elif mx == 0 and my == 2:
            return np.cos(2 * np.pi * 2 * y)
        elif mx == 2 and my == 2:
            return np.cos(2 * np.pi * 2 * x) * np.cos(2 * np.pi * 2 * y)
        elif mx == 4 and my == 0:
            return np.cos(2 * np.pi * 4 * x)
        elif mx == 0 and my == 4:
            return np.cos(2 * np.pi * 4 * y)
        elif mx == 4 and my == 4:
            return np.cos(2 * np.pi * 4 * x) * np.cos(2 * np.pi * 4 * y)
        else:
            return np.zeros_like(x)
    
    def generate_cell_from_params(params, threshold=0.0):
        """
        Generate a 32x32 cell from Fourier harmonic parameters.
        params: list of (amplitude, phase) for each harmonic
        """
        x = (np.arange(size) + 0.5) / size
        y = (np.arange(size) + 0.5) / size
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Define the harmonics to use
        harmonics = [
            (2, 0),
            (0, 2),
            (2, 2),
            (4, 0),
            (0, 4),
            (4, 4),
        ]
        
        # Start with zero field
        field = np.zeros((size, size))
        
        # Add each harmonic with its amplitude and phase
        for idx, (mx, my) in enumerate(harmonics):
            amp, phase = params[idx]
            h = fourier_harmonic(mx, my, X, Y)
            field += amp * np.cos(phase) * h - amp * np.sin(phase) * np.sin(phase * np.pi / 2) * h
        
        # Threshold to binary
        cell = (field > threshold).astype(int)
        
        return cell
    
    # Initial parameters: start with a configuration similar to 2x2 rods
    # The (2,0) and (0,2) harmonics create a 2x2 pattern
    initial_params = [
        (1.0, 0.0),   # (2,0)
        (1.0, 0.0),   # (0,2)
        (0.5, 0.0),   # (2,2)
        (0.2, 0.0),   # (4,0)
        (0.2, 0.0),   # (0,4)
        (0.1, 0.0),   # (4,4)
    ]
    initial_threshold = 0.0
    
    best_cell = generate_cell_from_params(initial_params, initial_threshold)
    best_score = compute_gap(best_cell)
    
    # Local search in parameter space
    # Parameters: 6 amplitudes, 6 phases, 1 threshold = 13 parameters
    n_params = 13
    
    def sample_params():
        """Sample random parameters near the current best."""
        params = []
        for i in range(6):
            amp = max(0.0, initial_params[i][0] + np.random.normal(0, 0.3))
            phase = initial_params[i][1] + np.random.normal(0, 0.5)
            params.append((amp, phase))
        threshold = initial_threshold + np.random.normal(0, 0.2)
        return params, threshold
    
    def eval_params(params, threshold):
        cell = generate_cell_from_params(params, threshold)
        return compute_gap(cell)
    
    # Perform local search
    step_count = 0
    max_steps = 5000
    best_params = initial_params.copy()
    best_threshold = initial_threshold
    
    # First, do a coarse grid search over some key parameters
    if time.time() < deadline - 30:
        # Try different threshold values
        for thresh in [-0.5, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5]:
            if time.time() > deadline - 10:
                break
            cell = generate_cell_from_params(initial_params, thresh)
            score = compute_gap(cell)
            if score > best_score:
                best_score = score
                best_cell = cell
                best_threshold = thresh
        
        # Try varying the (2,0) and (0,2) amplitudes
        for amp20 in [0.5, 0.7, 0.9, 1.0, 1.2, 1.5]:
            if time.time() > deadline - 10:
                break
            params = initial_params.copy()
            params[0] = (amp20, 0.0)
            params[1] = (amp20, 0.0)
            cell = generate_cell_from_params(params, best_threshold)
            score = compute_gap(cell)
            if score > best_score:
                best_score = score
                best_cell = cell
                best_params = params.copy()
    
    # Now do random local search
    current_params = best_params.copy()
    current_threshold = best_threshold
    
    while time.time() < deadline and step_count < max_steps:
        step_count += 1
        
        # Perturb parameters
        new_params = []
        for i in range(6):
            amp = max(0.0, current_params[i][0] + np.random.normal(0, 0.15))
            phase = current_params[i][1] + np.random.normal(0, 0.2)
            new_params.append((amp, phase))
        new_threshold = current_threshold + np.random.normal(0, 0.1)
        
        cell = generate_cell_from_params(new_params, new_threshold)
        score = compute_gap(cell)
        
        if score > best_score:
            best_score = score
            best_cell = cell
            best_params = new_params.copy()
            best_threshold = new_threshold
            current_params = new_params.copy()
            current_threshold = new_threshold
        else:
            # With some probability, accept worse solutions (simulated annealing-like)
            if np.random.random() < 0.1 and score > 0:
                current_params = new_params.copy()
                current_threshold = new_threshold
    
    # Final local pixel search on the best cell
    quadrant = full_to_quadrant(best_cell)
    best_quadrant = quadrant.copy()
    
    pixel_steps = 0
    max_pixel_steps = 2000
    
    while time.time() < deadline - 5 and pixel_steps < max_pixel_steps:
        pixel_steps += 1
        
        i = np.random.randint(0, 16)
        j = np.random.randint(0, 16)
        
        quadrant[i, j] = 1 - quadrant[i, j]
        
        full_cell = quadrant_to_full(quadrant)
        score = compute_gap(full_cell)
        
        if score > best_score:
            best_score = score
            best_cell = full_cell
            best_quadrant = quadrant.copy()
        else:
            quadrant[i, j] = 1 - quadrant[i, j]
    
    # Return the best cell found
    best_full_cell = quadrant_to_full(best_quadrant)
    
    # Convert to nested list
    return best_full_cell.tolist()
