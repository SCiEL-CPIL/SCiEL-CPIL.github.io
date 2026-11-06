"""Photonic crystal unit cell design for maximum TM band gap between bands 4 and 5."""
import numpy as np
from scipy.linalg import eigh
import time

# EVOLVE-BLOCK-START
def design(problem):
    """Design a 32x32 D4-symmetric photonic crystal unit cell maximizing TM gap between bands 4 and 5."""
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Use fewer plane waves for speed: 121 (11x11)
    # m, n in [-5, 5]
    G_vals = np.array([[2*np.pi*m, 2*np.pi*n] for m in range(-5, 6) for n in range(-5, 6)])
    n_G = len(G_vals)
    
    # Precompute G_diff indices for fast lookup
    G_to_idx = {}
    for idx, (gx, gy) in enumerate(G_vals):
        G_to_idx[(gx, gy)] = idx
    
    # K-points: use 12 key points in irreducible BZ
    k_points = [
        (0, 0), (np.pi/4, 0), (np.pi/2, 0), (3*np.pi/4, 0), (np.pi, 0),
        (np.pi, np.pi/4), (np.pi, np.pi/2), (np.pi, 3*np.pi/4),
        (3*np.pi/4, np.pi/4), (np.pi/2, np.pi/4), (np.pi/4, np.pi/4),
        (np.pi/4, 0)
    ]
    
    # Pixel positions
    x = (np.arange(size) + 0.5) / size
    X, Y = np.meshgrid(x, x, indexing='ij')
    
    def set_fundamental(values):
        cell = np.zeros((size, size), dtype=np.float64)
        idx = 0
        for i in range(16):
            for j in range(i, 16):
                val = values[idx]
                idx += 1
                cell[i, j] = val
                cell[j, 31-i] = val
                cell[31-i, 31-j] = val
                cell[31-j, i] = val
                cell[i, 31-j] = val
                cell[31-j, 31-i] = val
                cell[31-i, j] = val
                cell[j, i] = val
        return cell
    
    def compute_eps_fourier(cell):
        eps_map = np.where(cell > 0.5, eps_high, eps_low)
        eps_fourier = np.zeros(n_G, dtype=np.float64)
        
        for idx_G in range(n_G):
            Gx, Gy = G_vals[idx_G]
            if Gx == 0 and Gy == 0:
                eps_fourier[idx_G] = np.mean(eps_map)
            else:
                phase = np.cos(Gx * X + Gy * Y)
                eps_fourier[idx_G] = np.mean(eps_map * phase)
        
        return eps_fourier
    
    def compute_score(cell):
        eps_fourier = compute_eps_fourier(cell)
        
        max_omega4 = 0
        min_omega5 = float('inf')
        
        for kx, ky in k_points:
            k_plus_G = G_vals + np.array([kx, ky])
            k2 = np.sum(k_plus_G**2, axis=1)
            
            M = np.zeros((n_G, n_G), dtype=np.float64)
            for i in range(n_G):
                for j in range(n_G):
                    G_diff = tuple(G_vals[i] - G_vals[j])
                    if G_diff in G_to_idx:
                        diff_idx = G_to_idx[G_diff]
                        eps_val = eps_fourier[diff_idx]
                        if abs(eps_val) > 1e-10:
                            M[i, j] = k2[i] / eps_val
            
            try:
                eigvals = eigh(M, eigvals_only=True, subset_by_index=[3, 4])
                if len(eigvals) >= 2:
                    max_omega4 = max(max_omega4, eigvals[0])
                    min_omega5 = min(min_omega5, eigvals[1])
            except:
                pass
        
        if min_omega5 == float('inf'):
            return -1
        
        gap = min_omega5 - max_omega4
        midgap = (min_omega5 + max_omega4) / 2
        if midgap > 0:
            return 2 * gap / midgap
        else:
            return -1
    
    # Initial: dielectric rod
    cell = np.zeros((size, size), dtype=np.float64)
    center = (size - 1) / 2.0
    for i in range(size):
        for j in range(size):
            if (i - center)**2 + (j - center)**2 < (0.2 * size)**2:
                cell[i, j] = 1
    
    def extract_fundamental(cell):
        values = []
        for i in range(16):
            for j in range(i, 16):
                values.append(1 if cell[i, j] > 0.5 else 0)
        return np.array(values, dtype=np.float64)
    
    current_values = extract_fundamental(cell)
    current_score = compute_score(cell)
    best_values = current_values.copy()
    best_score = current_score
    
    # Simulated annealing with deadline
    start_time = time.time()
    deadline = 220
    temperature = 0.3
    cooling_rate = 0.99
    
    while time.time() < deadline:
        idx = np.random.randint(136)
        new_values = current_values.copy()
        new_values[idx] = 1 - new_values[idx]
        
        new_cell = set_fundamental(new_values)
        new_score = compute_score(new_cell)
        
        delta = new_score - current_score
        if delta > 0 or (temperature > 0.01 and np.random.random() < np.exp(delta / temperature)):
            current_values = new_values
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_values = current_values.copy()
        
        temperature *= cooling_rate
    
    best_cell = set_fundamental(best_values)
    return best_cell.astype(int).tolist()

# EVOLVE-BLOCK-END
