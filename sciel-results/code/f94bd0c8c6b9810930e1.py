"""
Sphere packing optimization using Multi-Start L-BFGS-B.
Strategy:
1. Use a Repulsive Energy surrogate (sum of inverse squared distances) for the main optimization.
   This function is smooth and provides strong gradients to push close pairs apart.
2. Use a Log-Sum-Exp (LSE) surrogate for a final polishing phase to precisely maximize
   the minimum radius (soft-min).
3. Multi-start with diverse initial configurations (Jittered Grid, FCC-like, Random).
"""

import numpy as np
import time

N = 50
CUBE_HALF = 0.5
DEADLINE = 115.0 # Safety margin for 120s limit

def _score(pts):
    """Calculate the minimum radius for a configuration."""
    # Vectorized min distance
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    min_wall = np.min(0.5 - np.abs(pts))
    return min(min_dist / 2.0, min_wall)

def _generate_grid_start():
    """Generate a 50-point starting configuration from a jittered 4x4x4 grid."""
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    np.random.seed(42)
    grid += np.random.randn(*grid.shape) * 0.02
    grid = np.clip(grid, -0.49, 0.49)
    return grid[:N].copy()

def _generate_fcc_start():
    """Generate starting configuration from FCC lattice points."""
    points = []
    for i in range(-1, 3):
        for j in range(-1, 3):
            for k in range(-1, 3):
                for offset in [(0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)]:
                    p = np.array([i + offset[0], j + offset[1], k + offset[2]])
                    if -0.5 <= p[0] <= 0.5 and -0.5 <= p[1] <= 0.5 and -0.5 <= p[2] <= 0.5:
                        points.append(p)
    points = np.array(points)
    if len(points) > N:
        idx = np.random.choice(len(points), N, replace=False)
        points = points[idx]
    elif len(points) < N:
        extra = np.random.randn(N - len(points), 3) * 0.3
        points = np.vstack([points, extra])
    return points[:N].copy()

def _generate_random_start(seed):
    np.random.seed(seed)
    pts = np.random.uniform(-0.45, 0.45, size=(N, 3))
    return pts

def _simulated_annealing(pts, time_limit, seed=0):
    """
    Simulated Annealing to maximize the minimum radius.
    Uses a 'move' strategy that targets the bottleneck constraints.
    """
    np.random.seed(seed)
    start_time = time.time()
    current_pts = pts.copy()
    best_pts = current_pts.copy()
    best_score = _score(best_pts)
    current_score = best_score
    
    # Initial temperature
    T = 0.01
    T_min = 1e-6
    alpha = 0.99 # Cooling rate
    
    while time.time() - start_time < time_limit:
        # Check for time
        if time.time() - start_time > time_limit:
            break
            
        # Identify bottleneck
        # 1. Find closest pair
        diff = current_pts[:, np.newaxis, :] - current_pts[np.newaxis, :, :]
        dists_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        min_dist_idx = np.unravel_index(np.argmin(dists_sq), dists_sq.shape)
        i, j = min_dist_idx[0], min_dist_idx[1]
        min_dist = np.sqrt(dists_sq[i, j])
        
        # 2. Find closest wall
        wall_dists = CUBE_HALF - np.abs(current_pts)
        min_wall_idx = np.unravel_index(np.argmin(wall_dists), wall_dists.shape)
        k, axis = min_wall_idx[0], min_wall_idx[1]
        min_wall = wall_dists[k, axis]
        
        # Determine the bottleneck radius
        current_r = min(min_dist / 2.0, min_wall)
        
        # Generate a move
        # Strategy: Push the most constrained spheres apart
        move_type = np.random.random()
        
        new_pts = current_pts.copy()
        
        if move_type < 0.5:
            # Push pair i, j apart
            direction = current_pts[i] - current_pts[j]
            norm = np.linalg.norm(direction)
            if norm < 1e-10:
                direction = np.random.randn(3)
                norm = np.linalg.norm(direction)
            direction = direction / norm
            
            step_size = np.random.uniform(0.01, 0.05) * T * 100 # Scale step with T
            # Move i in direction, j in opposite
            new_pts[i] += direction * step_size
            new_pts[j] -= direction * step_size
        else:
            # Push sphere k away from wall
            # Determine sign to move away from wall
            sign = -np.sign(current_pts[k, axis])
            if sign == 0:
                sign = np.random.choice([-1, 1])
                
            step_size = np.random.uniform(0.01, 0.05) * T * 100
            new_pts[k, axis] += sign * step_size
        
        # Clip to box
        new_pts = np.clip(new_pts, -CUBE_HALF + 1e-6, CUBE_HALF - 1e-6)
        
        # Calculate new score
        new_score = _score(new_pts)
        
        # Acceptance criterion
        if new_score >= current_score:
            current_pts = new_pts
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_pts = current_pts.copy()
        else:
            # Metropolis criterion
            delta = current_score - new_score
            # Probability of accepting worse move
            # We use a simple Boltzmann factor
            if T > T_min:
                p_accept = np.exp(-delta / T)
                if np.random.random() < p_accept:
                    current_pts = new_pts
                    current_score = new_score
        
        # Cool down
        T *= alpha
        
        # Random restarts within SA if stuck? 
        # If T is very low, we might do a random kick
        if T < 0.001:
             # Random kick to escape local minima
             if np.random.random() < 0.01:
                 idx = np.random.randint(N)
                 new_pts = current_pts.copy()
                 new_pts[idx] += np.random.randn(3) * 0.05
                 new_pts = np.clip(new_pts, -CUBE_HALF + 1e-6, CUBE_HALF - 1e-6)
                 if _score(new_pts) > current_score * 0.9: # Allow slight degradation
                     current_pts = new_pts
                     current_score = _score(current_pts)
                     T = 0.01 # Reheat slightly

    return best_pts

def construct():
    start_time = time.time()
    
    # Generate diverse initial points
    starts = []
    starts.append(_generate_grid_start())
    starts.append(_generate_fcc_start())
    
    # Add random starts
    for i in range(5):
        starts.append(_generate_random_start(100 + i))
        
    best_pts = None
    best_score = -np.inf
    
    # Score initial points
    scored_starts = [(s, _score(s)) for s in starts]
    scored_starts.sort(key=lambda x: x[1], reverse=True)
    
    # Allocate time budget
    # We will run SA on the top 3 starts, giving each ~35s.
    num_runs = 3
    time_per_run = (DEADLINE - (time.time() - start_time) - 5) / num_runs
    
    for i in range(num_runs):
        if time.time() - start_time > DEADLINE - 5:
            break
            
        init_pts, init_score = scored_starts[i]
        
        # Run Simulated Annealing
        optimized_pts = _simulated_annealing(init_pts, time_per_run, seed=42 + i)
        
        # Evaluate true score
        current_score = _score(optimized_pts)
        
        if current_score > best_score:
            best_score = current_score
            best_pts = optimized_pts
            
        # Check time
        if time.time() - start_time > DEADLINE - 10:
            break
            
    # Fallback
    if best_pts is None:
        best_pts = scored_starts[0][0]
        
    # Ensure validity
    best_pts = np.clip(best_pts, -0.4999999, 0.4999999)
    
    return best_pts
