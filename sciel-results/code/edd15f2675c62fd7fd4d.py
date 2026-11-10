import numpy as np
import time
import random

SIZE = 32
QUAD = 8  # top-left quadrant size

# Precompute G vectors for plane-wave expansion
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

# Precompute k-points
def generate_k_points(num_points=30):
    points = []
    for i in range(10):
        kx = (i + 0.5) * (np.pi / 10)
        points.append((kx, 0))
    for i in range(10):
        ky = (i + 0.5) * (np.pi / 10)
        points.append((np.pi, ky))
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
KG_INV_SQRT = [1.0 / np.sqrt(kg_sq) for kg_sq in KG_SQ]

def evaluate_cell(cell, verbose=False):
    size = len(cell)
    eps_arr = np.array(cell, dtype=float)
    eps_arr[eps_arr == 1] = 12.0
    eps_arr[eps_arr == 0] = 1.0
    mu = 1.0 / eps_arr
    
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
    size = SIZE
    cell = [[0]*size for _ in range(size)]
    for i in range(QUAD):
        for j in range(QUAD):
            val = q[i][j]
            cell[i][j] = val
            cell[i][size-1-j] = val
            cell[size-1-i][j] = val
            cell[size-1-i][size-1-j] = val
    return cell

def cell_to_quadrant(cell):
    return [row[:QUAD] for row in cell[:QUAD]]

def flip_pixel(cell, i, j):
    size = SIZE
    val = cell[i][j]
    new_val = 1 - val
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

def random_quadrant():
    q = [[0]*QUAD for _ in range(QUAD)]
    for i in range(QUAD):
        for j in range(QUAD):
            q[i][j] = 1 if random.random() > 0.5 else 0
    return q

def design(problem):
    start_time = time.time()
    deadline = start_time + 115
    
    # Phase 1: Generate diverse initial candidates
    candidates = []
    
    # Random quadrants
    for _ in range(20):
        q = random_quadrant()
        candidates.append(quadrant_to_cell(q))
    
    # Structured patterns
    # Ring
    q_ring = [[0]*QUAD for _ in range(QUAD)]
    for i in range(QUAD):
        for j in range(QUAD):
            x = (i + 0.5) / SIZE - 0.5
            y = (j + 0.5) / SIZE - 0.5
            r = np.sqrt(x*x + y*y)
            if 0.15 < r < 0.35:
                q_ring[i][j] = 1
    candidates.append(quadrant_to_cell(q_ring))
    
    # Cross
    q_cross = [[0]*QUAD for _ in range(QUAD)]
    for i in range(QUAD):
        for j in range(QUAD):
            x = (i + 0.5) / SIZE - 0.5
            y = (j + 0.5) / SIZE - 0.5
            if abs(x) < 0.12 or abs(y) < 0.12:
                q_cross[i][j] = 1
    candidates.append(quadrant_to_cell(q_cross))
    
    # Checkerboard
    q_cb = [[0]*QUAD for _ in range(QUAD)]
    for i in range(QUAD):
        for j in range(QUAD):
            if (i + j) % 2 == 0:
                q_cb[i][j] = 1
    candidates.append(quadrant_to_cell(q_cb))
    
    # Evaluate all candidates
    best_cell = None
    best_score = -float('inf')
    for c in candidates:
        s = evaluate_cell(c)
        if s > best_score:
            best_score = s
            best_cell = [row[:] for row in c]
    
    # Phase 2: Local search with simulated annealing-like acceptance
    current_cell = [row[:] for row in best_cell]
    current_score = best_score
    temperature = 0.05
    cooling_rate = 0.995
    min_temp = 0.001
    stagnation = 0
    
    while time.time() < deadline:
        # Generate a single-flip move
        i = random.randint(0, QUAD - 1)
        j = random.randint(0, QUAD - 1)
        new_cell = [row[:] for row in current_cell]
        flip_pixel(new_cell, i, j)
        new_score = evaluate_cell(new_cell)
        
        delta = new_score - current_score
        accept = False
        if delta > 0:
            accept = True
        elif temperature > min_temp:
            # Accept worse moves with probability exp(delta / temperature)
            if random.random() < np.exp(delta / temperature):
                accept = True
        
        if accept:
            current_cell = new_cell
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_cell = [row[:] for row in current_cell]
                stagnation = 0
            else:
                stagnation += 1
            temperature *= cooling_rate
        else:
            stagnation += 1
            # Occasionally increase temperature to escape local minima
            if stagnation > 100 and random.random() < 0.1:
                temperature = max(temperature * 2, 0.02)
    
    return best_cell
