"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time


def _bary_to_cart(bary):
    """Convert barycentric coordinates (n,3) to Cartesian (n,2).
    Barycentric: (a,b,c) means point = a*(0,0) + b*(1,0) + c*(0.5, sqrt(3)/2)
    So x = b + 0.5*c, y = c*sqrt(3)/2"""
    b, c = bary[:, 1], bary[:, 2]
    x = b + 0.5 * c
    y = c * (np.sqrt(3) / 2)
    return np.column_stack([x, y])


def _cart_to_bary(points):
    """Convert Cartesian (n,2) to barycentric (n,3)."""
    x = points[:, 0]
    y = points[:, 1]
    c = y / (np.sqrt(3) / 2)
    b = x - 0.5 * c
    a = 1.0 - b - c
    return np.column_stack([a, b, c])


def _min_triangle_area_fast(points):
    """Compute min area of any triangle formed by the n points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            d = points[j] - points[i]
            diff_k = points - points[i]
            cross = d[0] * diff_k[:, 1] - d[1] * diff_k[:, 0]
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            mask[j] = False
            areas = 0.5 * np.abs(cross[mask])
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
    return min_area


def _random_bary_init(rng, n_points):
    """Generate random barycentric coordinates inside the triangle."""
    # Use Dirichlet-like sampling: generate 3 random uniforms, normalize
    raw = rng.exponential(scale=1.0, size=(n_points, 3))
    bary = raw / raw.sum(axis=1, keepdims=True)
    # Add small noise to avoid exact boundaries
    bary = bary + 0.02 * rng.uniform(size=(n_points, 3))
    bary = bary / bary.sum(axis=1, keepdims=True)
    return bary


def _structured_bary_inits():
    """Create diverse structured initial configurations in barycentric coords."""
    configs = []
    
    # Config 0: Symmetric - points near vertices and center
    bary0 = np.array([
        [0.9, 0.05, 0.05],
        [0.05, 0.9, 0.05],
        [0.05, 0.05, 0.9],
        [0.333, 0.333, 0.333],
        [0.6, 0.2, 0.2],
        [0.2, 0.6, 0.2],
        [0.2, 0.2, 0.6],
        [0.5, 0.25, 0.25],
        [0.25, 0.5, 0.25],
        [0.25, 0.25, 0.5],
        [0.4, 0.35, 0.25],
    ])
    configs.append(bary0)
    
    # Config 1: Three rows
    bary1 = np.array([
        [0.7, 0.2, 0.1], [0.5, 0.3, 0.2], [0.3, 0.4, 0.3], [0.1, 0.5, 0.4],
        [0.6, 0.3, 0.1], [0.4, 0.4, 0.2], [0.2, 0.5, 0.3], [0.0, 0.6, 0.4],
        [0.5, 0.35, 0.15], [0.3, 0.45, 0.25], [0.1, 0.55, 0.35],
    ])
    configs.append(bary1)
    
    # Config 2: Ring-like
    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    r = 0.3
    bary2 = np.zeros((11, 3))
    bary2[0] = [1/3, 1/3, 1/3]
    for i in range(8):
        a = 1/3 + r * np.cos(angles[i]) / 3
        b = 1/3 + r * np.sin(angles[i]) / 3
        c = 1/3 - (a + b) + 2/3
        # Normalize to ensure they sum to 1 and are positive
        bary2[i+1] = np.array([1/3, 1/3, 1/3]) + 0.15 * np.array([np.cos(angles[i]), np.sin(angles[i]), -np.cos(angles[i])-np.sin(angles[i])])
    bary2[9] = [0.2, 0.4, 0.4]
    bary2[10] = [0.4, 0.2, 0.4]
    # Normalize all
    bary2 = np.maximum(bary2, 0.01)
    bary2 = bary2 / bary2.sum(axis=1, keepdims=True)
    configs.append(bary2)
    
    # Config 3: Grid in barycentric space
    grid_vals = [0.1, 0.3, 0.5, 0.7, 0.9]
    bary3_list = []
    for a in grid_vals:
        for b in grid_vals:
            c = 1 - a - b
            if c >= 0.05:
                bary3_list.append([a, b, c])
    # Select 11 diverse ones
    if len(bary3_list) >= 11:
        bary3 = np.array(bary3_list[:11])
    else:
        bary3 = np.array(bary3_list)
        while len(bary3) < 11:
            bary3 = np.vstack([bary3, [0.33, 0.33, 0.34]])
    configs.append(bary3[:11])
    
    return configs


def _simulated_annealing(bary_init, rng, max_iters=50000, t_start=0.1, t_end=0.001, 
                          cooling=0.9999, deadline=None):
    """Simulated annealing in barycentric space."""
    n = len(bary_init)
    current_bary = bary_init.copy()
    current_points = _bary_to_cart(current_bary)
    current_score = _min_triangle_area_fast(current_points)
    
    best_bary = current_bary.copy()
    best_score = current_score
    
    t = t_start
    alpha = np.log(t_start / t_end) / max_iters
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Generate perturbation
        # Random walk in barycentric space
        delta = rng.normal(0, 0.05 * t / t_start + 0.001, size=(n, 3))
        new_bary = current_bary + delta
        
        # Project back to valid barycentric (non-negative, sum to 1)
        new_bary = np.maximum(new_bary, 0)
        new_bary = new_bary / new_bary.sum(axis=1, keepdims=True)
        
        new_points = _bary_to_cart(new_bary)
        new_score = _min_triangle_area_fast(new_points)
        
        # Accept if better, or with probability exp((new - current)/t)
        delta_score = new_score - current_score
        if delta_score > 0:
            current_bary = new_bary
            current_points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_bary = current_bary.copy()
        else:
            # Metropolis acceptance
            if t > 1e-10:
                log_prob = delta_score / t
                if log_prob > -50:  # Avoid underflow
                    if np.log(rng.uniform()) < log_prob:
                        current_bary = new_bary
                        current_points = new_points
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_bary = current_bary.copy()
        
        # Cool
        t = max(t * (1 - alpha), t_end)
    
    return best_bary, best_score


def _local_refine_bary(bary, rng, n_steps=200, step_size=0.01):
    """Local refinement using small random steps (stochastic hill climbing)."""
    current_bary = bary.copy()
    current_points = _bary_to_cart(current_bary)
    current_score = _min_triangle_area_fast(current_points)
    
    best_bary = current_bary.copy()
    best_score = current_score
    
    for _ in range(n_steps):
        # Try moving one point at a time
        idx = rng.integers(0, len(bary))
        delta = rng.normal(0, step_size, size=(3,))
        new_bary = current_bary.copy()
        new_bary[idx] += delta
        new_bary = np.maximum(new_bary, 0)
        new_bary = new_bary / new_bary.sum(axis=1, keepdims=True)
        
        new_points = _bary_to_cart(new_bary)
        new_score = _min_triangle_area_fast(new_points)
        
        if new_score > current_score:
            current_bary = new_bary
            current_points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_bary = current_bary.copy()
    
    return best_bary, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s
    
    rng = np.random.default_rng(42)
    
    best_bary = None
    best_score = -float('inf')
    
    # Get structured initial configurations
    inits = _structured_bary_inits()
    
    # Add random inits
    for _ in range(3):
        inits.append(_random_bary_init(rng, N))
    
    # Run simulated annealing from each init
    for start_idx, bary_init in enumerate(inits):
        if time.time() > deadline - 20:
            break
        
        # Run SA with different parameters
        try:
            sa_bary, sa_score = _simulated_annealing(
                bary_init, rng,
                max_iters=30000,
                t_start=0.08,
                t_end=0.0005,
                deadline=deadline - 15
            )
            
            if sa_score > best_score:
                best_score = sa_score
                best_bary = sa_bary.copy()
        except Exception:
            pass
    
    # If no valid result, use fallback
    if best_bary is None:
        best_bary = np.array([
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.05, 0.9],
            [0.333, 0.333, 0.334],
            [0.6, 0.2, 0.2],
            [0.2, 0.6, 0.2],
            [0.2, 0.2, 0.6],
            [0.5, 0.25, 0.25],
            [0.25, 0.5, 0.25],
            [0.25, 0.25, 0.5],
            [0.4, 0.35, 0.25],
        ])
        best_score = _min_triangle_area_fast(_bary_to_cart(best_bary))
    
    # Local refinement from best
    if time.time() < deadline - 10:
        try:
            ref_bary, ref_score = _local_refine_bary(best_bary, rng, n_steps=300, step_size=0.005)
            if ref_score > best_score:
                best_bary = ref_bary
                best_score = ref_score
        except Exception:
            pass
    
    # Another round of SA from the best
    if time.time() < deadline - 5:
        try:
            sa_bary, sa_score = _simulated_annealing(
                best_bary, rng,
                max_iters=20000,
                t_start=0.02,
                t_end=0.0002,
                deadline=deadline - 3
            )
            if sa_score > best_score:
                best_bary = sa_bary
                best_score = sa_score
        except Exception:
            pass
    
    # Convert to Cartesian
    best_points = _bary_to_cart(best_bary)
    
    # Ensure points are strictly inside (small margin)
    # Barycentric coords already guarantee this if all components > 0
    
    return best_points
# EVOLVE-BLOCK-END
