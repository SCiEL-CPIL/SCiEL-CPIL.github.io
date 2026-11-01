import numpy as np
from scipy.spatial.distance import pdist

N = 100

# EVOLVE-BLOCK-START
def construct():
    def get_radius(pts):
        """Calculate the maximum valid radius for a set of points."""
        if len(pts) == 0:
            return 0.0
        dists = pdist(pts)
        min_pair_dist = np.min(dists) / 2.0 if len(dists) > 0 else np.inf
        
        wall_dists = 0.5 - np.abs(pts)
        min_wall_dist = np.min(wall_dists)
        
        return min(min_pair_dist, min_wall_dist)

    def compute_forces(pts, power=6):
        """
        Compute repulsive forces based on 1/d^power potential.
        Force on i from j is proportional to (r_i - r_j) / d^power.
        Wall forces push points inward.
        """
        n = len(pts)
        forces = np.zeros_like(pts)
        
        # Pairwise forces
        # Use pdist to get distances, but we need directions.
        # For N=100, O(N^2) loop is fine (10k ops).
        for i in range(n):
            for j in range(i + 1, n):
                diff = pts[i] - pts[j]
                d = np.linalg.norm(diff)
                if d < 1e-8:
                    # Random small push to break symmetry
                    diff = np.random.randn(3) * 1e-6
                    d = np.linalg.norm(diff)
                
                # Force magnitude: k / d^power
                # Gradient of 1/d^power is -power * d^(-power-1) * (r_i - r_j)/d
                # We want to move i away from j, so force on i is in direction of diff.
                # Magnitude of potential is 1/d^power.
                # Force = -grad(Potential) = power * d^(-power-1) * diff
                # Let's just use diff / d^power for simplicity, it's monotonic.
                f_mag = 1.0 / (d ** power)
                forces[i] += f_mag * diff
                forces[j] -= f_mag * diff

        # Wall forces
        # Potential: 1 / (0.5 - |x|)^power
        # Gradient w.r.t x: if x > 0, d/dx (0.5-x)^-power = power * (0.5-x)^(-power-1)
        # Force should push towards center (negative x direction if x>0).
        # So force_x = - power * (0.5 - |x|)^(-power-1) * sign(x)
        
        wall_dists = 0.5 - np.abs(pts)
        # Avoid division by zero
        wall_dists_safe = np.maximum(wall_dists, 1e-6)
        
        # Force component for each axis
        # f_wall = - power * wall_dists_safe^(-power-1) * sign(pts)
        # If pts is near 0.5, wall_dist is small, force is large negative (push left).
        # If pts is near -0.5, wall_dist is small, sign is -1, force is large positive (push right).
        
        forces -= power * (wall_dists_safe ** (-power - 1)) * np.sign(pts)
        
        return forces

    def run_descent(x0, time_limit=30.0):
        import time
        start_time = time.time()
        pts = x0.copy()
        best_radius = get_radius(pts)
        best_pts = pts.copy()
        
        # Adaptive step size
        step_size = 0.01
        min_step_size = 1e-6
        no_improve_count = 0
        max_no_improve = 100
        
        while (time.time() - start_time) < time_limit:
            forces = compute_forces(pts, power=6)
            
            # Normalize forces to avoid huge jumps
            max_force = np.max(np.linalg.norm(forces, axis=1))
            if max_force > 0:
                forces = forces / max_force
            
            # Update
            new_pts = pts + step_size * forces
            new_pts = np.clip(new_pts, -0.5 + 1e-9, 0.5 - 1e-9)
            
            new_radius = get_radius(new_pts)
            
            if new_radius > best_radius:
                best_radius = new_radius
                best_pts = new_pts.copy()
                pts = new_pts
                no_improve_count = 0
                # Optionally increase step size slightly if we are improving fast
                # step_size = min(step_size * 1.1, 0.05)
            else:
                pts = new_pts # Still move, but track no improvement
                no_improve_count += 1
                # Reduce step size
                step_size *= 0.95
                if step_size < min_step_size:
                    step_size = min_step_size
            
            if no_improve_count > max_no_improve:
                # Try a larger jump? Or restart?
                # For now, just continue with smaller steps
                pass
                
        return best_pts, best_radius

    # Initializations
    best_pts_overall = None
    best_radius_overall = -np.inf
    
    # Grid 1: 4x5x5
    nx, ny, nz = 4, 5, 5
    xs = np.linspace(-0.4, 0.4, nx)
    ys = np.linspace(-0.4, 0.4, ny)
    zs = np.linspace(-0.4, 0.4, nz)
    grid1 = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    
    # Grid 2: 5x5x4 (swap axes)
    nx, ny, nz = 5, 5, 4
    xs = np.linspace(-0.4, 0.4, nx)
    ys = np.linspace(-0.4, 0.4, ny)
    zs = np.linspace(-0.4, 0.4, nz)
    grid2 = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    
    # Grid 3: 5x4x5
    nx, ny, nz = 5, 4, 5
    xs = np.linspace(-0.4, 0.4, nx)
    ys = np.linspace(-0.4, 0.4, ny)
    zs = np.linspace(-0.4, 0.4, nz)
    grid3 = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    
    # Jitter each grid slightly to avoid symmetry locks
    rng = np.random.default_rng(42)
    
    for grid in [grid1, grid2, grid3]:
        pts_init = grid + rng.uniform(-0.01, 0.01, grid.shape)
        pts_result, r_result = run_descent(pts_init, time_limit=30.0)
        
        if r_result > best_radius_overall:
            best_radius_overall = r_result
            best_pts_overall = pts_result
            
        # If we already beat the best known (0.11138), we can stop early
        if best_radius_overall > 0.11138:
            break
            
    if best_pts_overall is None:
        # Fallback
        m = 5
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in xs]) # Wait, xs is from last loop
        # Just use a standard grid
        ticks = (np.arange(5) + 0.5) / 5 - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]
        
    return best_pts_overall
# EVOLVE-BLOCK-END
