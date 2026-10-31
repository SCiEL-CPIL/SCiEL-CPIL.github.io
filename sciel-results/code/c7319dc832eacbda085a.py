import numpy as np
from scipy.optimize import minimize, differential_evolution
from itertools import combinations
import time

N = 11
TIME_LIMIT = 145.0  # Safety margin for 150s limit

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

def clamp_bary(bary, eps=1e-9):
    """Clamp barycentric coordinates to the strictly valid triangle region."""
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
    """Objective using hard min."""
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def generate_hexagonal_init():
    """
    Generate a hexagonal lattice-like initial configuration.
    """
    best_bary = None
    best_val = -np.inf
    
    for scale in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]:
        pts = []
        max_k = int(1.0 / scale) + 1
        for k in range(max_k + 1):
            w_val = k * scale
            if w_val > 1.0:
                break
            num_pts_in_row = int((1 - w_val) / scale) + 1
            for j in range(num_pts_in_row):
                u_val = j * scale
                v_val = (1 - w_val) - u_val
                if u_val >= -1e-9 and v_val >= -1e-9 and u_val + v_val <= 1 + 1e-9:
                    pts.append([u_val, v_val])
        
        if len(pts) >= N:
            # Try different subsets
            rng = np.random.default_rng(42)
            for _ in range(5):
                if len(pts) == N:
                    bary = np.array(pts)
                else:
                    idx = rng.choice(len(pts), N, replace=False)
                    bary = np.array(pts)[idx]
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

def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using Nelder-Mead for high precision.
    """
    start_time = time.time()
    try:
        result = minimize(
            hard_min_obj,
            bary_init.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-16, 'adaptive': True}
        )
        new_bary = clamp_bary(result.x.reshape(N, 2))
        # Ensure it's valid and better or equal
        if -hard_min_obj(new_bary.flatten()) >= -hard_min_obj(bary_init.flatten()) - 1e-12:
            return new_bary
    except Exception:
        pass
    return bary_init

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

    # Initialize with a good configuration
    current_bary = generate_hexagonal_init()
    current_val = -hard_min_obj(current_bary.flatten())
    
    best_bary = current_bary.copy()
    best_val = current_val

    # Simulated Annealing Parameters
    # Initial temperature should be comparable to the scale of the objective function changes
    # Objective is area, max area ~ 0.43, min area ~ 0.01. Changes are ~ 0.01.
    T = 0.05 
    T_min = 1e-8
    alpha = 0.995  # Cooling rate
    rng = np.random.default_rng(123)

    # Move size for random point displacement
    move_sigma = 0.1

    while time.time() < deadline - 10.0:
        # Generate a new candidate
        new_bary = current_bary.copy()
        
        # Randomly choose a move type:
        # 1. Move one point randomly (Gaussian perturbation)
        # 2. Swap two points (less useful for this objective but good for exploration)
        # 3. Move one point towards another (reduction of empty space)
        
        move_type = rng.integers(0, 3)
        
        if move_type == 0:
            # Perturb a single point
            idx = rng.integers(0, N)
            delta_u = rng.normal(0, move_sigma)
            delta_v = rng.normal(0, move_sigma)
            new_bary[idx, 0] += delta_u
            new_bary[idx, 1] += delta_v
            new_bary = clamp_bary(new_bary)
            
        elif move_type == 1:
            # Swap two points? No, swapping doesn't change the set of points, so it's useless for this objective.
            # Instead, let's do a "reflection" or "rotation" of a point around another.
            # Or just another random move with different sigma.
            idx = rng.integers(0, N)
            # Larger move
            delta_u = rng.normal(0, move_sigma * 2)
            delta_v = rng.normal(0, move_sigma * 2)
            new_bary[idx, 0] += delta_u
            new_bary[idx, 1] += delta_v
            new_bary = clamp_bary(new_bary)
            
        else:
            # Move a point towards the centroid of the triangle or another point to fill gaps?
            # Let's try moving a point towards the point it is closest to, but with a step size.
            # This might be too greedy. Let's stick to random perturbations but vary the sigma.
            idx = rng.integers(0, N)
            # Small fine-tuning move
            delta_u = rng.normal(0, move_sigma * 0.1)
            delta_v = rng.normal(0, move_sigma * 0.1)
            new_bary[idx, 0] += delta_u
            new_bary[idx, 1] += delta_v
            new_bary = clamp_bary(new_bary)

        new_val = -hard_min_obj(new_bary.flatten())
        
        # Acceptance criterion
        if new_val > current_val:
            current_bary = new_bary
            current_val = new_val
            if current_val > best_val:
                best_val = current_val
                best_bary = current_bary.copy()
        else:
            # Accept worse solution with probability exp((new_val - current_val) / T)
            # Note: new_val and current_val are negative (minimization).
            # delta = new_val - current_val (negative)
            # prob = exp(delta / T)
            delta = new_val - current_val
            if delta < 0:
                prob = np.exp(delta / T)
                if rng.random() < prob:
                    current_bary = new_bary
                    current_val = new_val
        
        # Cooling
        T *= alpha
        if T < T_min:
            T = T_min
            # Restart temperature if stuck? Or just keep going?
            # Let's reheat slightly if we haven't improved in a while.
            # For simplicity, we just let it cool to min and keep exploring at min T (which is effectively greedy + random noise).
            # Actually, at T_min, exp(delta/T) is very small for any negative delta, so it becomes greedy.
            # To maintain exploration, we should perhaps restart T periodically.
            # Let's implement a simple restart: if no improvement for 1000 steps, reheat.
            # But tracking steps is overhead. Let's just let it run. The random moves with small sigma will still explore locally.
            # However, SA usually benefits from reheating.
            # Let's add a simple reheat logic based on time.
            pass

    # Final polish with Nelder-Mead
    if time.time() < deadline - 2.0:
        polished = refine_local(best_bary, time_limit=deadline - time.time() - 1.0)
        polished_val = -hard_min_obj(polished.flatten())
        if polished_val > best_val:
            best_val = polished_val
            best_bary = polished

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart
