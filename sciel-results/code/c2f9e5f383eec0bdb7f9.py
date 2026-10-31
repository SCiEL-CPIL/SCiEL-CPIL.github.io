import numpy as np
import time
from itertools import combinations

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

def clamp_bary(bary, eps=1e-6):
    """Clamp barycentric coordinates to the strictly valid triangle region."""
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    
    # Ensure u, v >= eps
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    # Ensure u + v <= 1 - eps
    sums = u + v
    invalid = sums > (1 - eps)
    if np.any(invalid):
        # Scale down u and v for invalid points to satisfy u+v <= 1-eps
        # We want to preserve the ratio u:v roughly, so scale both by same factor
        scale = (1 - eps) / np.where(sums > 0, sums, 1.0)
        u[invalid] *= scale[invalid]
        v[invalid] *= scale[invalid]
        
    # Final clip just in case
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    return np.column_stack([u, v])

def soft_min_obj(bary, beta=50.0):
    """
    Soft minimum of areas using log-sum-exp.
    Minimization of this function pushes the smallest areas up.
    """
    areas = calc_areas_bary(bary)
    # Avoid log(0)
    areas = np.maximum(areas, 1e-15)
    
    # We want to maximize the min area.
    # -log(sum(exp(-beta * area))) approximates -min(area) * beta for large beta.
    # Minimizing -log(sum(exp(-beta * area))) is equivalent to maximizing log(sum(exp(-beta * area))).
    # Wait, standard soft-min is (1/beta) * log(sum(exp(-beta * x_i))).
    # As beta -> inf, this approaches min(x_i).
    # We want to MAXIMIZE min(x_i). So we MINIMIZE -soft_min(x_i).
    
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    # Numerical stability: subtract max
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min_val = log_sum_exp / beta
    
    return -soft_min_val

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
                val = min(calc_areas_bary(bary))
                if val > best_val:
                    best_val = val
                    best_bary = bary.copy()
                
    if best_bary is None:
        rng = np.random.default_rng(42)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        best_bary = np.column_stack([u, v])
        
    return best_bary

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # Initial best
    best_bary = generate_hexagonal_init()
    best_val = min(calc_areas_bary(best_bary))
    
    # Simulated Annealing Parameters
    # Start with a relatively high temperature
    T_current = 0.01
    T_min = 1e-6
    cooling_rate = 0.995  # Slow cooling to allow more exploration
    max_iter = 200000  # Upper bound, but time-limited
    
    rng = np.random.default_rng(123)
    
    # We will use a "kick" strategy:
    # 1. Calculate all areas.
    # 2. Identify the smallest area triangle.
    # 3. Perturb one of the vertices of that triangle significantly.
    # 4. Accept if better, or with probability exp(-delta/T) if worse.
    
    current_bary = best_bary.copy()
    current_val = best_val
    
    iter_count = 0
    while time.time() < deadline - 5:
        iter_count += 1
        
        # Calculate areas to find the bottleneck
        areas = calc_areas_bary(current_bary)
        min_area_idx = np.argmin(areas)
        min_area_val = areas[min_area_idx]
        
        # Get the indices of the points forming the smallest triangle
        tri_pts = tri_idx[min_area_idx]
        
        # Randomly pick one of these points to move
        pt_to_move = tri_pts[rng.integers(0, 3)]
        
        # Generate a perturbation
        # Strategy: Move the point towards the centroid of the other two points in the triangle?
        # Or just a random kick in barycentric space.
        # A "smart" kick: move away from the other two points to increase area?
        # Actually, moving *away* from the line connecting the other two increases area.
        
        # Let's try a random kick with magnitude proportional to T
        # We need to be careful with barycentric constraints.
        
        # 1. Random direction in barycentric space
        # u, v changes must keep u+v <= 1 and u,v >= 0.
        # Simple approach: generate new random point near current, or just add noise and clamp.
        
        # Let's use a "local search" step:
        # Pick a random point (not necessarily the worst one, to avoid local traps)
        # Or stick to the worst point strategy for efficiency.
        
        # Let's try moving the worst point towards a random point in the triangle?
        # No, that might make it worse.
        
        # Better: Random kick with adaptive size.
        step_size = 0.05 * (1 + T_current) # Scale step with temperature
        
        # Generate random delta in barycentric coordinates
        # dU, dV such that dU + dV is small to maintain validity roughly
        dU = rng.uniform(-step_size, step_size)
        dV = rng.uniform(-step_size, step_size)
        
        new_u = current_bary[pt_to_move, 0] + dU
        new_v = current_bary[pt_to_move, 1] + dV
        
        # Construct new bary array
        new_bary = current_bary.copy()
        new_bary[pt_to_move] = [new_u, new_v]
        new_bary = clamp_bary(new_bary)
        
        # Calculate new objective
        # We use the hard min for acceptance to be precise, but soft min for smoothness?
        # For SA, hard min is usually fine if T is high enough.
        # Let's use hard min for speed and precision in acceptance.
        new_areas = calc_areas_bary(new_bary)
        new_val = np.min(new_areas)
        
        delta = new_val - current_val
        
        # Acceptance criterion
        if delta > 0:
            current_bary = new_bary
            current_val = new_val
            
            if current_val > best_val:
                best_val = current_val
                best_bary = current_bary.copy()
        else:
            # Accept worse solution with probability exp(delta / T)
            # delta is negative, so exp(negative / T) < 1
            prob = np.exp(delta / T_current)
            if rng.random() < prob:
                current_bary = new_bary
                current_val = new_val
        
        # Cool down
        T_current *= cooling_rate
        if T_current < T_min:
            T_current = T_min
            
        # Occasionally, if we haven't improved in a long time, do a "jump" restart
        # But SA handles this via temperature. If T is low and we are stuck, we might need to reheat.
        # Simple reheat: if no improvement for 5000 steps, increase T slightly.
        # We don't track "last_improvement" easily here without extra vars, but we can just let it run.
        
    # Final polish: A few steps of local hill climbing on the best found
    # Just run a few iterations of greedy improvement
    for _ in range(100):
        improved = False
        areas = calc_areas_bary(best_bary)
        min_area_idx = np.argmin(areas)
        tri_pts = tri_idx[min_area_idx]
        
        # Try small moves for each point in the worst triangle
        for pt_idx in tri_pts:
            for _ in range(10):
                # Small random kick
                step_size = 0.01
                dU = rng.uniform(-step_size, step_size)
                dV = rng.uniform(-step_size, step_size)
                
                new_bary = best_bary.copy()
                new_bary[pt_idx] = [best_bary[pt_idx, 0] + dU, best_bary[pt_idx, 1] + dV]
                new_bary = clamp_bary(new_bary)
                
                new_val = np.min(calc_areas_bary(new_bary))
                if new_val > best_val:
                    best_val = new_val
                    best_bary = new_bary
                    improved = True
        
        if not improved:
            break

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart

# EVOLVE-BLOCK-END
