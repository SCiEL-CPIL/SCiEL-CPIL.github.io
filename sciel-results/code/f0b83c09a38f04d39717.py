import numpy as np
import time
import math
import random

# Precompute G vectors
N_PW = 8
G_range = np.arange(-N_PW, N_PW + 1)
Gx, Gy = np.meshgrid(G_range, G_range, indexing='ij')
G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
Gx_flat = G_vectors[:, 0]
Gy_flat = G_vectors[:, 1]
Diff_x = Gx_flat[:, None] - Gx_flat[None, :]
Diff_y = Gy_flat[:, None] - Gy_flat[None, :]
idx_x = (Diff_x + 16) % 32
idx_y = (Diff_y + 16) % 32

def generate_k_points(num_points=30):
    """Generate 30 k-points in the irreducible BZ of a square lattice."""
    points = []
    # Gamma-X: kx in [0, pi], ky = 0
    for i in range(10):
        kx = (i + 0.5) * (np.pi / 10)
        points.append((kx, 0))
        
    # X-M: kx = pi, ky in [0, pi]
    for i in range(10):
        ky = (i + 0.5) * (np.pi / 10)
        points.append((np.pi, ky))
        
    # M-Gamma: kx = ky
    for i in range(10):
        t = (i + 0.5) * (np.pi / 10)
        points.append((t, t))
        
    return np.array(points)

K_POINTS = generate_k_points()
KG_SQ = []
for k in K_POINTS:
    kx, ky = k
    kg = k[None, :] + G_vectors * (2 * np.pi)
    kg_sq = np.sum(kg**2, axis=1)
    KG_SQ.append(kg_sq)
KG_SQ = np.array(KG_SQ)

def evaluate_cell(cell, verbose=False):
    """
    Evaluate the gap-midgap ratio for a 32x32 cell.
    cell: list of lists of 0/1.
    """
    size = len(cell)
    # 1 = dielectric (eps=12), 0 = air (eps=1)
    # mu = 1/eps
    eps = np.array([[12 if c == 1 else 1 for c in row] for row in cell], dtype=float)
    mu = 1.0 / eps
    
    # Compute Fourier coefficients of mu
    mu_fft = np.fft.fft2(mu)
    mu_fft_shifted = np.fft.fftshift(mu_fft)
    mu_diff = mu_fft_shifted[idx_x, idx_y] / (size * size)
    
    min_omega_5 = float('inf')
    max_omega_4 = float('-inf')
    
    for ki in range(len(K_POINTS)):
        kg_sq = KG_SQ[ki]
        # Avoid division by zero if any (k+G) is 0, which shouldn't happen for k in BZ except Gamma
        # But Gamma is not in our 30 points (we use midpoints).
        # However, if k is such that k+G=0, it's a singular case. 
        # In our setup, k is in (0, pi], G is integer multiples of 2pi.
        # k+G=0 implies k = -G. Since k is in first quadrant BZ, this only happens at Gamma (0,0) with G=(0,0).
        # We don't sample Gamma. So kg_sq > 0 always.
        
        B_inv_sqrt = 1.0 / np.sqrt(kg_sq)
        S = mu_diff * B_inv_sqrt[:, None] * B_inv_sqrt[None, :]
        
        # Only compute 5 smallest eigenvalues for speed
        # np.linalg.eigvalsh is fast for 289x289
        evals = np.linalg.eigvalsh(S)
        w4 = np.sqrt(evals[3])
        w5 = np.sqrt(evals[4])
        
        max_omega_4 = max(max_omega_4, w4)
        min_omega_5 = min(min_omega_5, w5)
        
    gap = min_omega_5 - max_omega_4
    midgap = (min_omega_5 + max_omega_4) / 2.0
    if midgap == 0:
        return 0.0
    return gap / midgap

def generate_perlin_noise_2d(size, scale, seed):
    """
    Generate a 2D Perlin-like noise field.
    We use a simple value noise with interpolation for speed and simplicity,
    or a sum of random sinusoids (Fourier-based noise) which is faster and smoother.
    Let's use Fourier-based noise (random spectral synthesis) as it's more controllable and fast.
    """
    rng = np.random.RandomState(seed)
    noise = np.zeros((size, size))
    
    # Number of random frequencies to sum
    n_freqs = 10
    for _ in range(n_freqs):
        # Random frequency in 2D
        # Scale controls the roughness. Higher scale -> higher freqs -> rougher.
        # We want frequencies that fit in the 32x32 grid.
        # Let's pick integer-ish frequencies to maintain periodicity? 
        # No, the cell is a unit cell, so the pattern should be periodic.
        # Using integer frequencies ensures periodicity.
        fx = rng.randint(1, int(size/scale) + 1)
        fy = rng.randint(1, int(size/scale) + 1)
        
        # Random phase
        px = rng.uniform(0, 2*np.pi)
        py = rng.uniform(0, 2*np.pi)
        
        # Random amplitude
        amp = rng.uniform(0.5, 1.0)
        
        x = np.arange(size)
        y = np.arange(size)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Add sinusoidal component
        noise += amp * np.sin(2*np.pi*fx*X/size + px) * np.cos(2*np.pi*fy*Y/size + py)
        
    return noise

def make_cell_from_params(params):
    """
    params: [seed, scale, threshold, freq_mult, phase_shift]
    Returns a 32x32 cell (list of lists)
    """
    seed = int(params[0])
    scale = params[1]
    threshold = params[2]
    freq_mult = params[3]
    phase_shift = params[4]
    
    size = 32
    # Generate noise
    noise = generate_perlin_noise_2d(size, scale, seed)
    
    # Apply frequency multiplier? 
    # Actually, 'scale' in generate_perlin_noise_2d controls the max freq.
    # Let's just use scale as the main control.
    
    # Normalize noise to [0, 1]
    noise_min = np.min(noise)
    noise_max = np.max(noise)
    if noise_max - noise_min > 0:
        noise = (noise - noise_min) / (noise_max - noise_min)
    else:
        noise = np.zeros_like(noise)
    
    # Threshold
    cell = (noise > threshold).astype(int)
    
    # Enforce C4v symmetry
    # The problem requires:
    # 1. cell == cell[::-1] row-wise (Horizontal symmetry)
    # 2. cell == cell[::-1] col-wise (Vertical symmetry)
    # 3. cell == cell.T (Transpose symmetry)
    # This implies the top-left 8x8 quadrant defines the whole cell.
    
    half = size // 2
    # Extract top-left quadrant
    q = cell[:half, :half]
    
    new_cell = np.zeros((size, size), dtype=int)
    
    # Fill quadrants
    # Top-Left
    new_cell[:half, :half] = q
    # Top-Right (Flip TL horizontally)
    new_cell[:half, half:] = q[:, ::-1]
    # Bottom-Left (Flip TL vertically)
    new_cell[half:, :half] = q[::-1, :]
    # Bottom-Right (Flip TL both)
    new_cell[half:, half:] = q[::-1, ::-1]
    
    # Check Transpose Symmetry
    # new_cell should be symmetric.
    # Does this construction guarantee new_cell == new_cell.T?
    # new_cell[i][j] = q[i][j]
    # new_cell[j][i] = q[j][i]
    # For new_cell[i][j] == new_cell[j][i], we need q[i][j] == q[j][i].
    # So the quadrant q must be symmetric.
    # We can enforce this by averaging q and q.T
    q_sym = (q + q.T) / 2.0
    q_sym = (q_sym > 0.5).astype(int)
    
    # Rebuild with symmetric quadrant
    new_cell = np.zeros((size, size), dtype=int)
    new_cell[:half, :half] = q_sym
    new_cell[:half, half:] = q_sym[:, ::-1]
    new_cell[half:, :half] = q_sym[::-1, :]
    new_cell[half:, half:] = q_sym[::-1, ::-1]
    
    return new_cell.tolist()

def design(problem):
    start_time = time.time()
    deadline = start_time + 115  # safety margin
    
    size = 32
    
    # Initial parameters
    # seed, scale, threshold, freq_mult, phase_shift
    # scale: 1 (fine) to 10 (coarse)
    # threshold: 0.2 to 0.8
    
    current_params = [random.randint(0, 1000), 4.0, 0.5, 1.0, 0.0]
    current_cell = make_cell_from_params(current_params)
    current_score = evaluate_cell(current_cell)
    
    best_params = current_params[:]
    best_score = current_score
    best_cell = [row[:] for row in current_cell]
    
    # Simulated Annealing
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.995
    
    while time.time() < deadline and T > T_min:
        # Perturb parameters
        new_params = current_params[:]
        
        # Change seed with low probability to jump to a new noise landscape
        if random.random() < 0.1:
            new_params[0] = random.randint(0, 1000)
        else:
            # Small change in scale
            new_params[1] = max(1.0, min(10.0, new_params[1] + random.gauss(0, 0.5)))
            # Small change in threshold
            new_params[2] = max(0.1, min(0.9, new_params[2] + random.gauss(0, 0.05)))
            
        new_cell = make_cell_from_params(new_params)
        new_score = evaluate_cell(new_cell)
        
        # Acceptance criterion
        if new_score > current_score or random.random() < math.exp((new_score - current_score) / T):
            current_params = new_params
            current_cell = new_cell
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_params = current_params[:]
                best_cell = [row[:] for row in current_cell]
                
        T *= cooling_rate
        
    return best_cell
