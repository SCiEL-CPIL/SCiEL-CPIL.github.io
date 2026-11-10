"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_score(pts):
    """Compute the maximum radius for given centers."""
    n = len(pts)
    # Min distance between any two centers
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    dmin = np.min(dists)
    
    # Min distance to walls
    wmin = np.min(0.5 - np.abs(pts))
    
    return min(dmin / 2.0, wmin)

def _generate_bcc_init():
    """
    Generate an initial configuration based on a BCC lattice.
    BCC is denser than simple cubic. 
    We create a grid of BCC unit cells and select the best 50 points 
    that fit within the cube, or just a dense BCC grid and jitter it.
    
    A BCC unit cell has points at corners (0,0,0) etc and center (0.5, 0.5, 0.5).
    In a lattice with spacing 'a', points are at integer coordinates and half-integer coordinates.
    To fit in [-0.5, 0.5], we need to scale.
    Volume of cube = 1. Volume per BCC point = a^3 / 2.
    We want 50 points. a^3 * 50 / 2 <= 1 => a^3 <= 0.04 => a <= 0.34.
    If a=0.34, a^3/2 = 0.0196. 1/0.0196 = 51 points. 
    So a BCC lattice with a~0.34 should hold ~50 points.
    """
    a = 0.34
    # Generate points in a grid larger than the cube, then filter
    # Range from -1 to 1 to be safe, then filter to [-0.5, 0.5]
    # BCC points: 
    # 1. Integer lattice points: (i*a, j*a, k*a)
    # 2. Centered lattice points: ((i+0.5)*a, (j+0.5)*a, (k+0.5)*a)
    
    # We need to center the lattice. 
    # Let's generate points in a box [-1, 1]^3 and filter.
    n_grid = 5 # -2 to 2
    
    pts_int = []
    pts_cen = []
    for i in range(-n_grid, n_grid + 1):
        for j in range(-n_grid, n_grid + 1):
            for k in range(-n_grid, n_grid + 1):
                pts_int.append([i*a, j*a, k*a])
                pts_cen.append([(i+0.5)*a, (j+0.5)*a, (k+0.5)*a])
                
    all_pts = np.array(pts_int + pts_cen)
    
    # Filter points strictly inside [-0.5, 0.5] with a small margin for jitter
    margin = 0.05
    mask = np.all(np.abs(all_pts) < (0.5 - margin), axis=1)
    pts = all_pts[mask]
    
    # If we have too many, we need to select 50. 
    # If we have too few, we need to add more or adjust 'a'.
    # For a=0.34, we expect ~50-60 points.
    if len(pts) > N:
        # Randomly select N points
        idx = np.random.choice(len(pts), N, replace=False)
        pts = pts[idx]
    elif len(pts) < N:
        # If too few, fill with random points or increase density
        # For robustness, let's just generate a simple grid if BCC fails to provide enough
        # But let's try a slightly smaller 'a' if needed? No, smaller 'a' means more points.
        # If len(pts) < N, it means the lattice is too sparse.
        # Let's just fallback to a jittered grid for initialization in this case
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        pts = grid[:N].copy()
        
    # Add small jitter to break symmetry
    pts += np.random.randn(len(pts), 3) * 0.01
    
    # Ensure inside cube
    pts = np.clip(pts, -0.49, 0.49)
    
    return pts

def _optimize_sa(pts, max_iter=20000, deadline=None):
    """
    Simulated Annealing optimization.
    Objective: Maximize min(d_ij/2, wall_dist).
    Energy: -min_dist (or a smooth approximation).
    To make SA effective, we use a smooth energy function that approximates the min-distance
    but is differentiable and responsive to small changes.
    E = -sum( exp(-alpha * d_ij) ) is not quite right for max-min.
    
    Better: 
    Score = min(dists, walls).
    We want to maximize this.
    Let's use a potential energy:
    U = sum_{i<j} 1/d_ij^p + sum_i 1/wall_i^p
    Minimizing U tends to maximize the minimum distances, especially the smallest ones,
    because 1/x^p grows very fast as x -> 0.
    """
    pts = pts.copy()
    n = len(pts)
    
    best_score = _compute_score(pts)
    best_pts = pts.copy()
    
    # SA Parameters
    T = 0.05  # Initial temperature
    T_min = 1e-6
    cooling_rate = 0.9995
    
    # Step size for moves
    step_scale = 0.02
    
    current_energy = None
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
            
        # Generate a new candidate by moving a random point
        idx = np.random.randint(n)
        old_pt = pts[idx].copy()
        
        # Gaussian perturbation
        new_pt = old_pt + np.random.randn(3) * step_scale
        pts[idx] = new_pt
        
        # Clamp to cube
        pts[idx] = np.clip(pts[idx], -0.499, 0.499)
        
        # Calculate new energy
        # Energy = sum 1/d^2 for pairs + sum 1/w^2 for walls
        # This is expensive to calculate fully every step.
        # Optimization: Only calculate change in energy.
        
        # But for simplicity and robustness within time limits, 
        # let's just calculate the score directly if it's cheap enough?
        # N=50, pairs=1225. 1225 ops per step. 20000 steps = 24.5M ops. 
        # Python loop overhead is high. Vectorized is better.
        
        # Let's try a hybrid: 
        # 1. Calculate full score periodically.
        # 2. Use a fast approximate energy for the Metropolis step.
        
        # Actually, let's just use the full score calculation but optimize it.
        # _compute_score is vectorized. 
        # 20000 iterations * (50x50 matrix ops) might be too slow in pure Python/NumPy overhead.
        # Let's reduce max_iter or make the step smarter.
        
        # Alternative: Use a local move that only affects neighbors.
        # For now, let's stick to global score but limit iterations if it's slow.
        
        new_score = _compute_score(pts)
        
        # Metropolis Criterion
        # We want to maximize score.
        # delta_score = new_score - best_score? No, new_score - current_score (if we tracked it).
        # We didn't track current_score efficiently. 
        # Let's track current_score.
        
        # Wait, I didn't define current_score in the loop properly above.
        # Let's refactor to track current_score.
        
        if current_energy is None:
            current_energy = new_score # Just using score as energy proxy (maximize)
            
        delta = new_score - current_energy
        
        if delta > 0:
            # Accept
            current_energy = new_score
            if new_score > best_score:
                best_score = new_score
                best_pts = pts.copy()
        else:
            # Reject or accept with probability exp(delta / T)
            # delta is negative.
            prob = np.exp(delta / T)
            if np.random.rand() < prob:
                current_energy = new_score
            else:
                # Revert
                pts[idx] = old_pt
                current_energy = current_energy # Unchanged
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    np.random.seed(42)
    
    best_pts = None
    best_score = -1.0
    
    # Initial configurations
    configs = []
    
    # 1. BCC-like initial config
    p1 = _generate_bcc_init()
    configs.append(p1)
    
    # 2. Jittered 4x4x4 grid (64 points, take 50)
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    p2 = grid[:N].copy()
    p2 += np.random.randn(N, 3) * 0.05
    configs.append(p2)
    
    # 3. Random points
    p3 = np.random.uniform(-0.4, 0.4, (N, 3))
    configs.append(p3)
    
    # Optimize each config
    for idx, init_pts in enumerate(configs):
        if time.time() > deadline:
            break
        
        # Run SA optimization
        # SA is stochastic, so multiple runs or longer runs are needed.
        # We allocate time based on remaining budget.
        remaining_configs = len(configs) - idx
        time_per_config = (deadline - time.time()) / remaining_configs
        
        # Determine max_iter based on time_per_config
        # Estimate time per iter: ~0.1ms? 
        # Let's just use a fixed reasonable number and rely on the deadline check inside.
        pts_opt, score = _optimize_sa(init_pts, max_iter=5000, deadline=deadline)
        
        if score > best_score:
            best_score = score
            best_pts = pts_opt
            
    # Fallback if all failed or score is bad
    if best_pts is None or best_score < 0.1:
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_pts = grid[:N]
        best_score = _compute_score(best_pts)
        
    return best_pts
# EVOLVE-BLOCK-END
