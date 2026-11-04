# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms


def _compute_covering_radius_fast(pts):
    """
    Compute a fast estimate of the covering radius in degrees.
    Uses a dense grid.
    """
    # Higher resolution for better accuracy in scoring
    n_theta = 200
    n_phi = 400
    thetas = np.linspace(0, np.pi, n_theta)
    phis = np.linspace(0, 2 * np.pi, n_phi)
    TH, PH = np.meshgrid(thetas, phis, indexing='ij')
    Y = np.zeros((n_theta, n_phi, 3))
    Y[:, :, 0] = np.sin(TH) * np.cos(PH)
    Y[:, :, 1] = np.sin(TH) * np.sin(PH)
    Y[:, :, 2] = np.cos(TH)
    Y = Y.reshape(-1, 3)
    
    # Compute max dot product (min angular distance) for each grid point
    # Y is (M, 3), pts is (N, 3). Dots is (M, N)
    dots = Y @ pts.T
    min_dots = np.max(dots, axis=1)
    max_angular = np.arccos(np.clip(min_dots, -1, 1))
    return np.degrees(np.max(max_angular))


def _simulate(pts_init, steps, power, dt_init, dt_min, damping):
    """
    Run Velocity Verlet simulation for a fixed number of steps.
    """
    N = pts_init.shape[0]
    pts = pts_init.copy()
    vel = np.zeros((N, 3))
    
    for step in range(steps):
        t = step / steps
        
        # Adaptive step size: decay from dt_init to dt_min
        dt = dt_init * ((dt_min / dt_init) ** t)
        # Damping increases over time
        current_damping = damping[0] + (damping[1] - damping[0]) * t
        
        # --- Vectorized Force Calculation ---
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist2 = np.sum(diff ** 2, axis=2)
        
        # Avoid division by zero on diagonal
        np.fill_diagonal(dist2, 1.0)
        
        # Force magnitude: proportional to 1/dist^(power+1)
        # V = sum 1/dist^power, F = -grad(V) = sum power * 1/dist^(power+1) * r_vec
        inv_dist = 1.0 / np.sqrt(dist2)
        f_mag = inv_dist ** (power + 1)
        
        # Zero out self-forces
        np.fill_diagonal(f_mag, 0.0)
        
        # Total force on each point i: sum_j f_mag[i,j] * diff[i,j]
        forces = np.sum(f_mag[:, :, np.newaxis] * diff, axis=1)
        
        # Project forces onto the tangent plane of the sphere at each point
        # F_tangent = F - (F . p) p
        dot_fp = np.sum(forces * pts, axis=1, keepdims=True)
        forces -= dot_fp * pts
        
        # --- Velocity Verlet Update ---
        # v_half = v + 0.5 * dt * F
        vel_half = vel + 0.5 * dt * forces
        
        # Project vel_half to tangent plane
        dot_vp = np.sum(vel_half * pts, axis=1, keepdims=True)
        vel_half -= dot_vp * pts
        
        # x_new = x + dt * v_half
        pts_new = pts + dt * vel_half
        
        # Normalize to unit sphere
        pts_new = _normalize(pts_new)
        
        # Recalculate forces at new position
        diff_new = pts_new[:, np.newaxis, :] - pts_new[np.newaxis, :, :]
        dist2_new = np.sum(diff_new ** 2, axis=2)
        np.fill_diagonal(dist2_new, 1.0)
        inv_dist_new = 1.0 / np.sqrt(dist2_new)
        f_mag_new = inv_dist_new ** (power + 1)
        np.fill_diagonal(f_mag_new, 0.0)
        forces_new = np.sum(f_mag_new[:, :, np.newaxis] * diff_new, axis=1)
        
        # Project new forces to tangent plane
        dot_fp_new = np.sum(forces_new * pts_new, axis=1, keepdims=True)
        forces_new -= dot_fp_new * pts_new
        
        # v_new = v_half + 0.5 * dt * F_new
        vel_new = vel_half + 0.5 * dt * forces_new
        
        # Apply damping
        vel_new *= current_damping
        
        # Project vel_new to tangent plane at new position
        dot_vp_new = np.sum(vel_new * pts_new, axis=1, keepdims=True)
        vel_new -= dot_vp_new * pts_new
        
        # Update state
        pts = pts_new
        vel = vel_new
        
    return pts


def _fibonacci_sphere(N, seed=0):
    """Generate a Fibonacci sphere distribution."""
    rng = np.random.default_rng(seed)
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    
    # Add small random perturbation to break symmetry and help escape local minima
    # But keep it small to start near the ideal distribution
    noise = rng.normal(0, 0.01, (N, 3))
    
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts += noise
    return _normalize(pts)


def construct():
    """
    Minimize covering radius using multi-start annealing with Velocity Verlet.
    """
    N = 50
    start_time = time.time()
    time_limit = 110.0  # Leave 10s buffer
    
    best_pts = None
    best_score = float('inf')
    
    # Parameters for annealing
    # We will try a few different "power" values and restarts
    # Higher power approximates covering radius better
    # We start with low power to find basin, then increase
    
    num_restarts = 8
    steps_per_restart = 3000
    
    # Pre-allocate some arrays if needed, but simple loop is fine
    
    for r in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Generate initial points
        pts_init = _fibonacci_sphere(N, seed=r * 100)
        
        # Annealing schedule:
        # Phase 1: Low power (2-4) for broad convergence
        # Phase 2: Medium power (6-10)
        # Phase 3: High power (12-20) for fine tuning
        
        # We can simulate this by running the simulator with a changing power,
        # but the current _simulate takes a fixed power.
        # To keep it simple and fast, we will run sequential phases.
        
        current_pts = pts_init.copy()
        
        # Phase 1: Soft potential
        p1_steps = 1000
        p1_power = 4
        current_pts = _simulate(current_pts, p1_steps, p1_power, dt_init=0.05, dt_min=0.001, damping=(0.95, 0.99))
        
        # Phase 2: Medium potential
        p2_steps = 1000
        p2_power = 10
        current_pts = _simulate(current_pts, p2_steps, p2_power, dt_init=0.02, dt_min=0.001, damping=(0.98, 0.995))
        
        # Phase 3: Hard potential
        p3_steps = 1000
        p3_power = 20
        current_pts = _simulate(current_pts, p3_steps, p2_power, dt_init=0.01, dt_min=0.0005, damping=(0.99, 0.999))
        
        # Evaluate
        score = _compute_covering_radius_fast(current_pts)
        
        if score < best_score:
            best_score = score
            best_pts = current_pts.copy()
            
        # Optional: If we have time, do a quick local refinement with very high power
        # But 3000 steps per restart * 8 restarts = 24000 steps.
        # Each step is O(N^2) = 2500 ops. 24000 * 2500 = 60M ops. Should be fast in NumPy.
        
    if best_pts is None:
        # Fallback to a single run if something went wrong
        best_pts = _simulate(_fibonacci_sphere(N, seed=0), 3000, 10, 0.02, 0.001, (0.98, 0.995))
        
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
