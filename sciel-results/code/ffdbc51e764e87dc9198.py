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
        B_inv_sqrt = 1.0 / np.sqrt(kg_sq)
        S = mu_diff * B_inv_sqrt[:, None] * B_inv_sqrt[None, :]
        
        # Compute eigenvalues
        # The matrix S is symmetric.
        eigvals = np.linalg.eigvalsh(S)
        
        # Sort eigenvalues (eigvalsh returns sorted)
        # We need the 4th and 5th smallest (indices 3 and 4)
        w4 = np.sqrt(eigvals[3])
        w5 = np.sqrt(eigvals[4])
        
        max_omega_4 = max(max_omega_4, w4)
        min_omega_5 = min(min_omega_5, w5)
        
    gap = min_omega_5 - max_omega_4
    midgap = (min_omega_5 + max_omega_4) / 2.0
    if midgap == 0:
        return 0.0
    return gap / midgap

def expand_Q_to_32(Q):
    """
    Expand a 16x16 symmetric quadrant Q to a 32x32 symmetric cell.
    Q[i][j] is defined for i,j in 0..15.
    The full cell C is defined by C[i][j] = Q[f(i)][f(j)]
    where f(x) = x if x < 16 else 31-x.
    This ensures C is symmetric under row/col reversal and transpose.
    """
    C = [[0]*32 for _ in range(32)]
    for i in range(32):
        fi = i if i < 16 else 31 - i
        for j in range(32):
            fj = j if j < 16 else 31 - j
            C[i][j] = Q[fi][fj]
    return C

def generate_random_Q():
    """
    Generate a random 16x16 symmetric quadrant Q.
    """
    Q = [[0]*16 for _ in range(16)]
    for i in range(16):
        for j in range(i, 16):
            val = 1 if np.random.random() > 0.5 else 0
            Q[i][j] = val
            Q[j][i] = val
    return Q

def flip_Q_pixel(Q, i, j):
    """
    Flip pixel at i, j in the 16x16 quadrant Q.
    i, j in 0..15.
    Returns the modified Q.
    """
    new_val = 1 - Q[i][j]
    Q[i][j] = new_val
    Q[j][i] = new_val
    return Q

def design(problem):
    start_time = time.time()
    deadline = start_time + 115  # safety margin
    
    candidates_Q = []
    
    # Generate some structured Q candidates
    # 1. All 0
    Q0 = [[0]*16 for _ in range(16)]
    candidates_Q.append(Q0)
    
    # 2. All 1
    Q1 = [[1]*16 for _ in range(16)]
    candidates_Q.append(Q1)
    
    # 3. Random
    for _ in range(5):
        candidates_Q.append(generate_random_Q())
        
    # 4. Structured: Center heavy
    Qc = [[0]*16 for _ in range(16)]
    for i in range(16):
        for j in range(16):
            if i < 8 and j < 8:
                Qc[i][j] = 1
                Qc[j][i] = 1
    candidates_Q.append(Qc)
    
    best_Q = None
    best_score = -float('inf')
    
    for Q in candidates_Q:
        cell = expand_Q_to_32(Q)
        s = evaluate_cell(cell)
        if s > best_score:
            best_score = s
            best_Q = [row[:] for row in Q]
            
    # Local Search
    current_Q = [row[:] for row in best_Q]
    current_score = best_score
    stagnation = 0
    
    while time.time() < deadline:
        # Single flip in 16x16 quadrant
        i = np.random.randint(0, 16)
        j = np.random.randint(0, 16)
        
        new_Q = [row[:] for row in current_Q]
        flip_Q_pixel(new_Q, i, j)
        
        cell = expand_Q_to_32(new_Q)
        new_score = evaluate_cell(cell)
        
        if new_score > current_score:
            current_Q = new_Q
            current_score = new_score
            best_Q = [row[:] for row in current_Q]
            best_score = current_score
            stagnation = 0
        else:
            stagnation += 1
            # If stagnated, try a random restart or multi-flip?
            # For now, just continue.
            
    return expand_Q_to_32(best_Q)
