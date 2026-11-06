"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def design(problem):
    """Optimize D4-symmetric dielectric pattern for 4th-5th TM band gap using CA patterns."""
    size = problem["pixels"]
    eps_high = problem.get("eps_high", 12)
    start_time = time.time()
    deadline = start_time + 210  # 20s safety margin before 230s kill
    
    def enforce_d4_symmetry(cell):
        """Average cell over all 8 D4 symmetry operations."""
        sym = np.zeros_like(cell, dtype=float)
        sym += cell
        sym += np.rot90(cell, 1)
        sym += np.rot90(cell, 2)
        sym += np.rot90(cell, 3)
        sym += np.fliplr(cell)
        sym += np.flipud(cell)
        sym += np.fliplr(np.flipud(cell))
        sym += np.rot90(np.fliplr(cell), 1)
        return (sym / 8.0 > 0.5).astype(np.uint8)
    
    def evolve_ca(seed, steps, rule=110):
        """Evolve a 1D elementary CA, then tile to 2D with D4 symmetry."""
        n = len(seed)
        state = seed.copy()
        for _ in range(steps):
            left = np.roll(state, 1)
            right = np.roll(state, -1)
            pattern = (left << 2) | (state << 1) | right
            new_state = np.array([(rule >> p) & 1 for p in pattern], dtype=np.uint8)
            state = new_state
        
        # Tile the 1D CA state into a 2D pattern
        # Use the CA state as both row and column patterns
        ca2d = np.outer(state, state)
        # Scale to fit size x size using nearest-neighbor interpolation
        if size != n:
            scale = size / n
            row_indices = (np.arange(size) * scale).astype(int)
            col_indices = (np.arange(size) * scale).astype(int)
            # Clip to valid range
            row_indices = np.clip(row_indices, 0, n-1)
            col_indices = np.clip(col_indices, 0, n-1)
            ca2d = ca2d[np.ix_(row_indices, col_indices)]
        cell = ca2d.astype(np.uint8)
        return enforce_d4_symmetry(cell)
    
    def score_cell_approx(cell):
        """
        Approximate band gap score using Fourier analysis.
        Higher score = better potential for 4th-5th TM band gap.
        We look for strong Fourier components at relevant wavevectors.
        """
        eps = 1 + (eps_high - 1) * cell
        eps_centered = eps - eps.mean()
        
        # Compute 2D FFT using numpy
        eps_fft = np.fft.fft2(eps_centered)
        eps_mag = np.abs(eps_fft)
        
        # For a 32x32 cell, the relevant G-vectors for 4th-5th band gap
        # are around |G| ~ 4-8 in units of 2π/L
        # We want strong components at these wavevectors
        
        # Create a grid of G-vector magnitudes
        kx = np.fft.fftfreq(size, d=1.0/size)
        ky = np.fft.fftfreq(size, d=1.0/size)
        KX, KY = np.meshgrid(kx, ky)
        Kmag = np.sqrt(KX**2 + KY**2)
        
        # For 4th-5th TM bands, we care about G-vectors with |G| in range ~3-7
        # (in units where lattice constant = 1)
        mask_low = (Kmag > 2.5) & (Kmag < 4.0)
        mask_high = (Kmag >= 4.0) & (Kmag < 6.5)
        
        # Score: ratio of power in high-G to low-G region
        # Strong high-G components indicate good Bragg scattering for higher bands
        power_low = np.sum(eps_mag[mask_low]**2)
        power_high = np.sum(eps_mag[mask_high]**2)
        
        if power_low < 1e-10:
            return 0.0
        
        # Also penalize too high or too low dielectric fraction
        frac = cell.mean()
        frac_penalty = 1.0 - abs(frac - 0.45) * 2.0  # Optimal around 45%
        frac_penalty = max(0.0, frac_penalty)
        
        return (power_high / (power_low + 1e-10)) * frac_penalty
    
    # Generate diverse CA-based patterns
    best_cell = None
    best_score = -1
    
    # Try various CA seeds and parameters
    np.random.seed(42)
    
    # Different 1D CA seed patterns (must be symmetric for D4)
    seed_patterns = [
        # Single seed
        np.array([0]*8 + [1] + [0]*7, dtype=np.uint8),
        # Two seeds
        np.array([0]*4 + [1,0,1] + [0]*4, dtype=np.uint8),
        # Three seeds
        np.array([0]*2 + [1,0,1,0,1] + [0]*2, dtype=np.uint8),
        # Checkerboard-like
        np.array([1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0], dtype=np.uint8),
        # Dense
        np.array([1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1], dtype=np.uint8),
        # Sparse
        np.array([1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0], dtype=np.uint8),
        # Alternating pairs
        np.array([1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0], dtype=np.uint8),
        # Random symmetric seeds
        np.array([1,0,0,1,0,1,1,0,0,1,1,0,0,1,0,1], dtype=np.uint8),
        np.array([0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0], dtype=np.uint8),
    ]
    
    # CA rules to try
    rules = [110, 184, 226, 30, 90, 150]
    
    # Number of evolution steps
    steps_range = [1, 2, 3, 4, 5, 6]
    
    # Also try some analytical patterns as fallback
    def make_square_pillar(half_width, rounded=False):
        x = np.linspace(-0.5, 0.5, size, endpoint=False)
        y = np.linspace(-0.5, 0.5, size, endpoint=False)
        X, Y = np.meshgrid(x, y)
        
        if rounded:
            # Square with rounded corners
            r = half_width * 0.3
            mask = (np.abs(X) < half_width) & (np.abs(Y) < half_width)
            corners = [(half_width, half_width), (-half_width, half_width),
                      (half_width, -half_width), (-half_width, -half_width)]
            for cx, cy in corners:
                dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
                mask &= (dist > r)
        else:
            mask = (np.abs(X) < half_width) & (np.abs(Y) < half_width)
        
        cell = mask.astype(np.uint8)
        return enforce_d4_symmetry(cell)
    
    def make_cross(arm_width):
        x = np.linspace(-0.5, 0.5, size, endpoint=False)
        y = np.linspace(-0.5, 0.5, size, endpoint=False)
        X, Y = np.meshgrid(x, y)
        mask = (np.abs(X) < arm_width) | (np.abs(Y) < arm_width)
        cell = mask.astype(np.uint8)
        return enforce_d4_symmetry(cell)
    
    # Evaluate analytical patterns first (fast)
    analytical_candidates = [
        make_square_pillar(0.15),
        make_square_pillar(0.2),
        make_square_pillar(0.25),
        make_square_pillar(0.15, rounded=True),
        make_square_pillar(0.2, rounded=True),
        make_square_pillar(0.25, rounded=True),
        make_cross(0.08),
        make_cross(0.1),
        make_cross(0.12),
    ]
    
    for cell in analytical_candidates:
        if time.time() > deadline:
            break
        s = score_cell_approx(cell)
        if s > best_score:
            best_score = s
            best_cell = cell
    
    # Now try CA patterns
    for seed in seed_patterns:
        for rule in rules:
            for steps in steps_range:
                if time.time() > deadline:
                    break
                cell = evolve_ca(seed, steps, rule)
                s = score_cell_approx(cell)
                if s > best_score:
                    best_score = s
                    best_cell = cell
    
    # If no good pattern found, use a reasonable default
    if best_cell is None:
        best_cell = make_square_pillar(0.2)
    
    # Final check: ensure dielectric fraction is reasonable
    frac = best_cell.mean()
    if frac < 0.2 or frac > 0.7:
        # Fall back to a known-good pattern
        best_cell = make_square_pillar(0.2)
    
    return best_cell.tolist()
# EVOLVE-BLOCK-END
