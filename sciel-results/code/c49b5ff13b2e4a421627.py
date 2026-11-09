import numpy as np
import time
import math

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
    eps_arr = np.array(cell, dtype=float)
    eps_arr[eps_arr == 1] = 12.0
    eps_arr[eps_arr == 0] = 1.0
    mu = 1.0 / eps_arr
    
    # Compute Fourier coefficients of mu
    mu_fft = np.fft.fft2(mu)
    mu_fft_shifted = np.fft.fftshift(mu_fft)
    mu_diff = mu_fft_shifted[idx_x, idx_y] / (size * size)
    
    min_omega_5 = float('inf')
    max_omega_4 = float('-inf')
    
    for ki in range(len(K_POINTS)):
        B_inv_sqrt = KG_INV_SQRT[ki]
        S = mu_diff * B_inv_sqrt[:, None] * B_inv_sqrt[None, :]
        
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

def quadrant_to_cell(q):
    """
    Convert an 8x8 symmetric quadrant to a 32x32 cell.
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

def cell_to_quadrant(cell):
    """
    Extract the 8x8 quadrant from a 32x32 cell.
    """
    return [row[:8] for row in cell[:8]]

def flip_pixel(cell, i, j):
    """
    Flip pixel at i, j in the 8x8 quadrant, and update the full cell.
    i, j in 0..7.
    """
    size = 32
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

def generate_initial_candidates():
    """
    Generate a diverse set of initial candidate cells.
    """
    size = 32
    candidates = []
    
    # 1. Small central rod
    cell_rod = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if x*x + y*y < 0.15**2:
                cell_rod[i][j] = 1
    candidates.append(cell_rod)
    
    # 2. Larger rod
    cell_rod2 = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if x*x + y*y < 0.25**2:
                cell_rod2[i][j] = 1
    candidates.append(cell_rod2)
    
    # 3. Ring (dielectric ring)
    cell_ring = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            r = np.sqrt(x*x + y*y)
            if 0.15 < r < 0.35:
                cell_ring[i][j] = 1
    candidates.append(cell_ring)
    
    # 4. Thick ring
    cell_ring2 = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            r = np.sqrt(x*x + y*y)
            if 0.12 < r < 0.42:
                cell_ring2[i][j] = 1
    candidates.append(cell_ring2)
    
    # 5. Air ring (inverted ring)
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
    
    # 6. Cross
    cell_cross = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            if (abs(i - 15.5) < 2.5 or abs(j - 15.5) < 2.5):
                cell_cross[i][j] = 1
    candidates.append(cell_cross)
    
    # 7. Checkerboard
    cell_cb = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            if (i + j) % 2 == 0:
                cell_cb[i][j] = 1
    candidates.append(cell_cb)
    
    # 8. Diamond
    cell_diamond = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if abs(x) + abs(y) < 0.25:
                cell_diamond[i][j] = 1
    candidates.append(cell_diamond)
    
    # 9. Square ring
    cell_sq_ring = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            max_coord = max(abs(x), abs(y))
            if 0.15 < max_coord < 0.35:
                cell_sq_ring[i][j] = 1
    candidates.append(cell_sq_ring)
    
    # 10. Random cells
    for _ in range(5):
        q = [[0]*8 for _ in range(8)]
        for i in range(8):
            for j in range(8):
                q[i][j] = 1 if np.random.random() > 0.5 else 0
        # Enforce symmetry in quadrant
        for i in range(8):
            for j in range(i, 8):
                if q[i][j] != q[j][i]:
                    val = 1 if np.random.random() > 0.5 else 0
                    q[i][j] = val
                    q[j][i] = val
        cell = quadrant_to_cell(q)
        candidates.append(cell)
        
    return candidates

def design(problem):
    start_time = time.time()
    deadline = start_time + 115  # safety margin
    
    # Generate initial candidates
    candidates = generate_initial_candidates()
    
    best_cell = None
    best_score = -float('inf')
    
    for c in candidates:
        s = evaluate_cell(c)
        if s > best_score:
            best_score = s
            best_cell = [row[:] for row in c]
            
    # Local Search with adaptive multi-flip moves
    current_cell = [row[:] for row in best_cell]
    current_score = best_score
    stagnation = 0
    total_iters = 0
    
    # Track best cells for restarts
    best_cells = [[row[:] for row in best_cell], best_score]
    
    while time.time() < deadline:
        total_iters += 1
        # Adaptive multi-flip: increase probability and size with stagnation
        flip_prob = min(0.3 + stagnation * 0.01, 0.7)
        if stagnation > 20 and np.random.random() < flip_prob:
            # Multi-flip move
            n_flips = min(2 + stagnation // 10, 6)
            new_cell = [row[:] for row in current_cell]
            for _ in range(n_flips):
                i = np.random.randint(0, 8)
                j = np.random.randint(0, 8)
                flip_pixel(new_cell, i, j)
            new_score = evaluate_cell(new_cell)
            if new_score > current_score:
                current_cell = new_cell
                current_score = new_score
                if new_score > best_score:
                    best_cell = [row[:] for row in current_cell]
                    best_score = new_score
                stagnation = 0
            else:
                stagnation += 1
        else:
            # Single flip
            i = np.random.randint(0, 8)
            j = np.random.randint(0, 8)
            
            new_cell = [row[:] for row in current_cell]
            flip_pixel(new_cell, i, j)
            
            new_score = evaluate_cell(new_cell)
            
            if new_score > current_score:
                current_cell = new_cell
                current_score = new_score
                if new_score > best_score:
                    best_cell = [row[:] for row in current_cell]
                    best_score = new_score
                stagnation = 0
            else:
                stagnation += 1
                
        # Restart if stagnation is high
        if stagnation > 50:
            # Pick a random best cell or generate new random
            if np.random.random() < 0.5:
                current_cell = [row[:] for row in best_cells[0]]
                current_score = best_cells[1]
            else:
                q = [[0]*8 for _ in range(8)]
                for i in range(8):
                    for j in range(8):
                        q[i][j] = 1 if np.random.random() > 0.5 else 0
                for i in range(8):
                    for j in range(i, 8):
                        if q[i][j] != q[j][i]:
                            val = 1 if np.random.random() > 0.5 else 0
                            q[i][j] = val
                            q[j][i] = val
                current_cell = quadrant_to_cell(q)
                current_score = evaluate_cell(current_cell)
                if current_score > best_score:
                    best_cell = [row[:] for row in current_cell]
                    best_score = current_score
            stagnation = 0
            
    return best_cell
