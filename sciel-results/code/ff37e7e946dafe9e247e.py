import numpy as np
import time

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

# Precompute k-points and (k+G)^2 for each k-point
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

# Precompute inverse sqrt of KG_SQ for each k-point
KG_INV_SQRT = [1.0 / np.sqrt(kg_sq) for kg_sq in KG_SQ]

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
        B_inv_sqrt = KG_INV_SQRT[ki]
        S = mu_diff * B_inv_sqrt[:, None] * B_inv_sqrt[None, :]
        
        # Symmetrize to ensure Hermiticity
        S = (S + S.T) / 2.0
        
        # Compute eigenvalues using NumPy
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

def flip_pixel(cell, i, j):
    """
    Flip pixel at i, j in the 8x8 quadrant, and update the full cell.
    i, j in 0..7.
    """
    size = 32
    # Get current value
    val = cell[i][j]
    new_val = 1 - val
    
    # Positions to update
    idxs = [
        (i, j), (i, size-1-j), (size-1-i, j), (size-1-i, size-1-j),
        (j, i), (j, size-1-i), (size-1-j, i), (size-1-j, size-1-i)
    ]
    seen = set()
    for r, c in idxs:
        if (r, c) not in seen:
            seen.add((r, c))
            cell[r][c] = new_val
            
    return cell

def make_cell_from_quadrant(q):
    """
    Create a full 32x32 cell from an 8x8 symmetric quadrant.
    q must be symmetric: q[i][j] == q[j][i].
    """
    size = 32
    cell = [[0]*size for _ in range(size)]
    for i in range(8):
        for j in range(8):
            val = q[i][j]
            cell[i][j] = val
            cell[i][size-1-j] = val
            cell[size-1-i][j] = val
            cell[size-1-i][size-1-j] = val
    return cell

def perlin_noise_2d(size, seed, freq=4.0):
    """
    Generate a simple 2D noise field. We'll use a simple value noise with
    smooth interpolation, which is easier to implement than true Perlin noise
    but captures the multi-scale structure needed.
    """
    rng = np.random.RandomState(seed)
    # Generate random values on a coarse grid
    grid_size = max(2, int(freq))
    coarse = rng.rand(grid_size, grid_size)
    
    # Interpolate to full size
    # Use bilinear interpolation
    x_coords = np.linspace(0, grid_size - 1, size)
    y_coords = np.linspace(0, grid_size - 1, size)
    
    x0 = np.floor(x_coords).astype(int)
    x1 = np.minimum(x0 + 1, grid_size - 1)
    y0 = np.floor(y_coords).astype(int)
    y1 = np.minimum(y0 + 1, grid_size - 1)
    
    dx = x_coords - x0
    dy = y_coords - y0
    
    # Smoothstep for better interpolation
    dx = dx * dx * (3 - 2 * dx)
    dy = dy * dy * (3 - 2 * dy)
    
    field = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            v00 = coarse[y0[i], x0[j]]
            v01 = coarse[y0[i], x1[j]]
            v10 = coarse[y1[i], x0[j]]
            v11 = coarse[y1[i], x1[j]]
            
            v0 = v00 * (1 - dx[j]) + v01 * dx[j]
            v1 = v10 * (1 - dx[j]) + v11 * dx[j]
            field[i, j] = v0 * (1 - dy[i]) + v1 * dy[i]
            
    return field

def generate_noise_cell(seed, freq, threshold):
    """
    Generate a C4v-symmetric cell from a noise field.
    """
    size = 32
    field = perlin_noise_2d(size, seed, freq)
    
    # Make the field symmetric under C4v
    # Average over the 8 symmetry operations
    sym_field = np.zeros((size, size))
    
    # Identity
    sym_field += field
    
    # Rotate 90
    sym_field += np.rot90(field, 1)
    
    # Rotate 180
    sym_field += np.rot90(field, 2)
    
    # Rotate 270
    sym_field += np.rot90(field, 3)
    
    # Flip x
    sym_field += field[:, ::-1]
    
    # Flip y
    sym_field += field[::-1, :]
    
    # Flip x + rotate 90
    sym_field += np.rot90(field[:, ::-1], 1)
    
    # Flip y + rotate 90
    sym_field += np.rot90(field[::-1, :], 1)
    
    sym_field /= 8.0
    
    # Threshold
    cell = (sym_field > threshold).astype(int)
    
    # Ensure strict C4v symmetry by constructing from quadrant
    # Extract top-left 8x8
    q = [[cell[i][j] for j in range(8)] for i in range(8)]
    
    # Enforce symmetry in quadrant
    for i in range(8):
        for j in range(i, 8):
            val = (q[i][j] + q[j][i]) // 2
            q[i][j] = val
            q[j][i] = val
            
    return make_cell_from_quadrant(q)

def design(problem):
    start_time = time.time()
    deadline = start_time + 115  # safety margin
    
    size = 32
    candidates = []
    
    # Rod
    cell_rod = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if x*x + y*y < 0.2**2:
                cell_rod[i][j] = 1
    candidates.append(cell_rod)
    
    # Checkerboard
    cell_cb = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            if (i + j) % 2 == 0:
                cell_cb[i][j] = 1
    candidates.append(cell_cb)
    
    # Inverted Checkerboard
    cell_icb = [[1 - cell_cb[i][j] for j in range(size)] for i in range(size)]
    candidates.append(cell_icb)
    
    # Cross (Horizontal and Vertical bars)
    cell_cross = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            if (abs(i - 15.5) < 2.5 or abs(j - 15.5) < 2.5):
                cell_cross[i][j] = 1
    candidates.append(cell_cross)
    
    # Ring (annulus) - dielectric ring with air center
    cell_ring = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            r = np.sqrt(x*x + y*y)
            if 0.15 < r < 0.35:
                cell_ring[i][j] = 1
    candidates.append(cell_ring)
    
    # Thick ring
    cell_ring2 = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            r = np.sqrt(x*x + y*y)
            if 0.12 < r < 0.42:
                cell_ring2[i][j] = 1
    candidates.append(cell_ring2)
    
    # Air ring (inverted ring)
    cell_air_ring = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            r = np.sqrt(x*x + y*y)
            if 0.15 < r < 0.35:
                cell_air_ring[i][j] = 0
            else:
                cell_air_ring[i][j] = 1
    candidates.append(cell_air_ring)
    
    # Diamond shape
    cell_diamond = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if abs(x) + abs(y) < 0.25:
                cell_diamond[i][j] = 1
    candidates.append(cell_diamond)
    
    # Hexagonal-like pattern
    cell_hex = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            # Approximate hexagonal symmetry
            if max(abs(x), abs(y), abs(x+y)/1.5, abs(x-y)/1.5) < 0.22:
                cell_hex[i][j] = 1
    candidates.append(cell_hex)
    
    # Cross with hole - cross shape with air center
    cell_cross_hole = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            r = np.sqrt(x*x + y*y)
            in_cross = (abs(x) < 0.12 or abs(y) < 0.12)
            in_hole = r < 0.08
            if in_cross and not in_hole:
                cell_cross_hole[i][j] = 1
    candidates.append(cell_cross_hole)
    
    # Square ring
    cell_sq_ring = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            max_coord = max(abs(x), abs(y))
            if 0.15 < max_coord < 0.35:
                cell_sq_ring[i][j] = 1
    candidates.append(cell_sq_ring)
    
    # Additional candidates: various ring widths and positions
    for r_inner, r_outer in [(0.1, 0.25), (0.2, 0.4), (0.15, 0.3), (0.25, 0.45)]:
        cell_r = [[0]*size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                r = np.sqrt(x*x + y*y)
                if r_inner < r < r_outer:
                    cell_r[i][j] = 1
        candidates.append(cell_r)
    
    # Air-filled ring (inverted)
    for r_inner, r_outer in [(0.1, 0.25), (0.2, 0.4)]:
        cell_ar = [[0]*size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                r = np.sqrt(x*x + y*y)
                if r_inner < r < r_outer:
                    cell_ar[i][j] = 0
                else:
                    cell_ar[i][j] = 1
        candidates.append(cell_ar)
    
    # Small dots pattern
    for dot_r in [0.08, 0.12, 0.15]:
        cell_dots = [[0]*size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                # Place dots at (0,0), (0.5,0), (0,0.5), (0.5,0.5) in unit cell
                for dx in [0, 0.5]:
                    for dy in [0, 0.5]:
                        dx2 = min(abs(x - dx), 1 - abs(x - dx))
                        dy2 = min(abs(y - dy), 1 - abs(y - dy))
                        if dx2*dx2 + dy2*dy2 < dot_r**2:
                            cell_dots[i][j] = 1
        candidates.append(cell_dots)
    
    # Superellipse shapes
    for n in [2, 4, 6, 8]:
        for r in [0.15, 0.2, 0.25, 0.3]:
            cell_se = [[0]*size for _ in range(size)]
            for i in range(size):
                for j in range(size):
                    x = (i + 0.5) / size - 0.5
                    y = (j + 0.5) / size - 0.5
                    if abs(x)**n + abs(y)**n < r**n:
                        cell_se[i][j] = 1
            candidates.append(cell_se)
    
    # Noise-based candidates
    for seed in range(20):
        for freq in [2, 4, 6]:
            for thresh in [0.3, 0.4, 0.5]:
                cell_noise = generate_noise_cell(seed, freq, thresh)
                candidates.append(cell_noise)
    
    best_cell = None
    best_score = -float('inf')
    
    for c in candidates:
        s = evaluate_cell(c)
        if s > best_score:
            best_score = s
            best_cell = [row[:] for row in c]
            
    # Simulated Annealing search
    current_cell = [row[:] for row in best_cell]
    current_score = best_score
    T = 0.1  # Initial temperature
    T_min = 0.001
    cooling = 0.995
    
    while time.time() < deadline and T > T_min:
        i = np.random.randint(0, 8)
        j = np.random.randint(0, 8)
        
        new_cell = [row[:] for row in current_cell]
        flip_pixel(new_cell, i, j)
        
        new_score = evaluate_cell(new_cell)
        delta = new_score - current_score
        
        if delta > 0 or np.random.random() < np.exp(delta / T):
            current_cell = new_cell
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_cell = [row[:] for row in current_cell]
        
        T *= cooling
        
        # Occasionally do a multi-flip move
        if np.random.random() < 0.1:
            n_flips = np.random.randint(2, 5)
            new_cell2 = [row[:] for row in current_cell]
            for _ in range(n_flips):
                ii = np.random.randint(0, 8)
                jj = np.random.randint(0, 8)
                flip_pixel(new_cell2, ii, jj)
            new_score2 = evaluate_cell(new_cell2)
            delta2 = new_score2 - current_score
            if delta2 > 0 or np.random.random() < np.exp(delta2 / T):
                current_cell = new_cell2
                current_score = new_score2
                if current_score > best_score:
                    best_score = current_score
                    best_cell = [row[:] for row in current_cell]
        
        T *= cooling
    
    # Final polish: try small perturbations around the best cell
    for _ in range(50):
        if time.time() > deadline - 5:
            break
        i = np.random.randint(0, 8)
        j = np.random.randint(0, 8)
        new_cell = [row[:] for row in best_cell]
        flip_pixel(new_cell, i, j)
        s = evaluate_cell(new_cell)
        if s > best_score:
            best_score = s
            best_cell = new_cell
    
    return best_cell
