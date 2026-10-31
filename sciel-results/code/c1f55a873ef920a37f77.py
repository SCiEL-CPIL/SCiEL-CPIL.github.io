"""Heilbronn triangles: maximize minimum triangle area for 11 points in unit equilateral triangle."""
import numpy as np
import time
from itertools import combinations

N = 11

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2

def bary_to_cart(bary):
    """Convert barycentric coords (u, v) to Cartesian (x, y)."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def calc_areas_bary(bary):
    """Calculate areas of all triangles formed by N points in barycentric coords."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    ui = u[tri_idx[:, 0]]
    vi = v[tri_idx[:, 0]]
    wi = w[tri_idx[:, 0]]
    
    uj = u[tri_idx[:, 1]]
    vj = v[tri_idx[:, 1]]
    wj = w[tri_idx[:, 1]]
    
    uk = u[tri_idx[:, 2]]
    vk = v[tri_idx[:, 2]]
    wk = w[tri_idx[:, 2]]
    
    det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
    
    return np.abs(det)

def clamp_bary(bary, eps=1e-8):
    """Clamp barycentric coordinates to valid triangle region."""
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    sums = u + v
    invalid = sums > (1 - eps)
    if np.any(invalid):
        scale = (1 - eps) / np.where(sums > 0, sums, 1.0)
        u[invalid] *= scale[invalid]
        v[invalid] *= scale[invalid]
        
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    return np.column_stack([u, v])

def hard_min_obj(bary_flat):
    """Objective: minimize negative of min triangle area."""
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def generate_hexagonal_init():
    """Generate hexagonal lattice-like initial configuration."""
    best_bary = None
    best_val = -np.inf
    
    for s in np.linspace(0.08, 0.45, 30):
        points = []
        for i in range(6):
            for j in range(6):
                u = i * s
                v = j * s
                if u + v <= 1 and len(points) < N:
                    points.append([u, v])
        
        if len(points) == N:
            bary = np.array(points)
            val = -hard_min_obj(bary.flatten())
            if val > best_val:
                best_val = val
                best_bary = bary.copy()
    
    if best_bary is None:
        rng = np.random.default_rng(42)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        best_bary = np.column_stack([u, v])
    
    return best_bary

def generate_random_init(seed=None):
    """Generate random valid barycentric coordinates."""
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return np.column_stack([u, v])

def simulated_annealing(bary_init, start_time, deadline, rng_seed=42):
    """
    Simulated annealing search in barycentric space.
    Accepts worse solutions with probability exp(-delta/T) to escape local minima.
    """
    rng = np.random.default_rng(rng_seed)
    
    # Current and best solutions
    current_bary = clamp_bary(bary_init.copy())
    current_val = -hard_min_obj(current_bary.flatten())
    
    best_bary = current_bary.copy()
    best_val = current_val
    
    # Annealing schedule
    T_start = 0.02
    T_end = 0.0001
    n_iterations = 200000
    
    # Adaptive step size
    base_step = 0.03
    
    for it in range(n_iterations):
        # Check time
        if time.time() > deadline - 2:
            break
        
        # Temperature
        alpha = it / n_iterations
        T = T_start * (T_end / T_start) ** alpha
        
        # Generate perturbation
        # Randomly perturb 1-3 points
        n_perturb = rng.integers(1, 4)
        indices = rng.choice(N, size=n_perturb, replace=False)
        
        new_bary = current_bary.copy()
        for idx in indices:
            # Perturb in barycentric space
            du = rng.normal(0, base_step)
            dv = rng.normal(0, base_step)
            new_bary[idx, 0] += du
            new_bary[idx, 1] += dv
        
        new_bary = clamp_bary(new_bary)
        new_val = -hard_min_obj(new_bary.flatten())
        
        # Acceptance criterion
        delta = new_val - current_val
        if delta > 0 or rng.random() < np.exp(delta / T):
            current_bary = new_bary
            current_val = new_val
            
            # Update best
            if current_val > best_val:
                best_val = current_val
                best_bary = current_bary.copy()
        
        # Occasionally reheat to escape deep local minima
        if it % 50000 == 0 and it > 0:
            # Add a larger perturbation
            idx = rng.integers(0, N)
            current_bary[idx, 0] += rng.normal(0, 0.1)
            current_bary[idx, 1] += rng.normal(0, 0.1)
            current_bary = clamp_bary(current_bary)
            current_val = -hard_min_obj(current_bary.flatten())
    
    return best_bary, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    best_bary = None
    best_val = -np.inf
    
    def update_best(bary):
        """Update best if improved."""
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        val = -hard_min_obj(bary.flatten())
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False
    
    # 1. Start with hexagonal lattice initialization
    hex_init = generate_hexagonal_init()
    update_best(hex_init)
    
    # 2. Run simulated annealing from hex init
    if best_bary is not None:
        sa_bary, sa_val = simulated_annealing(
            best_bary, start_time, deadline, rng_seed=42
        )
        if sa_val > best_val:
            best_bary = sa_bary
            best_val = sa_val
    
    # 3. Run SA from random initializations
    for i in range(5):
        if time.time() > deadline - 30:
            break
        rand_init = generate_random_init(seed=100 + i)
        sa_bary, sa_val = simulated_annealing(
            rand_init, start_time, deadline, rng_seed=200 + i
        )
        if sa_val > best_val:
            best_bary = sa_bary
            best_val = sa_val
    
    # 4. Local refinement: small perturbations around best
    if best_bary is not None and time.time() < deadline - 10:
        rng = np.random.default_rng(500)
        for i in range(10000):
            if time.time() > deadline - 5:
                break
            
            # Small perturbation
            new_bary = best_bary.copy()
            idx = rng.integers(0, N)
            new_bary[idx, 0] += rng.normal(0, 0.01)
            new_bary[idx, 1] += rng.normal(0, 0.01)
            new_bary = clamp_bary(new_bary)
            
            new_val = -hard_min_obj(new_bary.flatten())
            if new_val > best_val:
                best_val = new_val
                best_bary = new_bary
    
    # 5. Final SA polish with very low temperature
    if best_bary is not None and time.time() < deadline - 5:
        sa_bary, sa_val = simulated_annealing(
            best_bary, start_time, deadline, rng_seed=999
        )
        if sa_val > best_val:
            best_bary = sa_bary
            best_val = sa_val
    
    # If all failed, generate a random valid set
    if best_bary is None:
        best_bary = generate_random_init(seed=42)
    
    # Convert to Cartesian
    bary_final = clamp_bary(best_bary, eps=1e-7)
    cart = bary_to_cart(bary_final)
    
    # Final safety check: ensure all points are strictly inside
    for i in range(N):
        x, y = cart[i]
        if y < 1e-9:
            cart[i, 1] = 1e-9
        if y > np.sqrt(3) * x - 1e-9:
            cart[i, 0] = max(cart[i, 0], y / np.sqrt(3) + 1e-6)
        if y + np.sqrt(3) * x > np.sqrt(3) - 1e-9:
            cart[i, 0] = min(cart[i, 0], (np.sqrt(3) - y) / np.sqrt(3) - 1e-6)
            
    for i in range(N):
        x, y = cart[i]
        if y < 1e-9:
             cart[i, 1] = 1e-9
             
    return cart
