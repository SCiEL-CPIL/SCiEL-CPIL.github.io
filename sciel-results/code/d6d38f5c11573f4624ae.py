"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import cKDTree

def _compute_radius(centers):
    """Compute the maximum valid radius for the given centers."""
    if centers is None or len(centers) == 0:
        return 0.0
    
    # Check for non-finite values
    if not np.all(np.isfinite(centers)):
        return 0.0
        
    # Check if centers are within the square [-0.5, 0.5]
    # If any center is outside, the radius is 0 (invalid)
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0

    # 1. Distance to walls
    min_x = np.min(centers[:, 0])
    max_x = np.max(centers[:, 0])
    min_y = np.min(centers[:, 1])
    max_y = np.max(centers[:, 1])
    
    r_wall = min(0.5 + min_x, 0.5 - max_x, 0.5 + min_y, 0.5 - max_y)
    
    # 2. Distance between centers
    # Using cKDTree for efficiency in finding minimum distance
    tree = cKDTree(centers)
    # query_pairs returns pairs of points closer than r
    # We want the minimum distance, so we can just find the nearest neighbors
    # For N=200, pairwise is fast, but let's stick to robust logic.
    # Actually, for just the score, pairwise is fine and simpler to verify correctness.
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)
    r_pair = min_pair_dist / 2.0
    
    return min(r_wall, r_pair)

def _generate_hex_grid(n):
    """Generate a hexagonal packing initialization."""
    points = []
    # Target radius approx 0.036
    r_init = 0.036
    dy = r_init * np.sqrt(3)
    dx = 2 * r_init
    
    y_start = -0.5 + r_init
    x_start = -0.5 + r_init
    
    n_rows = int(np.ceil(1.0 / dy)) + 2
    n_cols = int(np.ceil(1.0 / dx)) + 2
    
    for i in range(n_rows):
        y = y_start + i * dy
        if y > 0.5 - r_init:
            continue
        offset = 0 if i % 2 == 0 else r_init
        for j in range(n_cols):
            x = x_start + j * dx + offset
            if x <= 0.5 - r_init:
                points.append([x, y])
                
    points = np.array(points)
    if len(points) > n:
        return points[:n]
    else:
        while len(points) < n:
            x = np.random.uniform(-0.5+r_init, 0.5-r_init)
            y = np.random.uniform(-0.5+r_init, 0.5-r_init)
            points = np.vstack([points, [[x, y]]])
        return points

def _generate_random(n):
    """Generate random points inside the square, kept away from edges."""
    # Keep them slightly inside to allow for some radius
    margin = 0.02
    return np.random.uniform(-0.5 + margin, 0.5 - margin, size=(n, 2))

def _simulate_annealing(centers, time_limit):
    """
    Perform Simulated Annealing on the centers to maximize the minimum radius.
    Uses cKDTree for fast neighbor identification during moves.
    """
    start_time = time.time()
    # Initial temperature
    T = 0.01
    # Cooling rate
    alpha = 0.999
    # Step size for moves
    step_scale = 0.01
    
    current_centers = centers.copy()
    current_r = _compute_radius(current_centers)
    best_centers = current_centers.copy()
    best_r = current_r
    
    # Pre-compute tree? No, it changes every step.
    # But we can use it to check validity/energy quickly?
    # Energy function: We want to maximize r.
    # Let's define energy E = -r. Minimize E.
    
    # To make SA work well, we need to evaluate the "gain" of a move quickly.
    # A full _compute_radius is O(N^2) or O(N log N). With N=200, N^2=40000 ops.
    # This is very fast in NumPy. We can afford many iterations.
    
    # Number of iterations per second?
    # Let's just run until time limit.
    
    while time.time() - start_time < time_limit:
        # Generate a random move
        # Pick a random point and move it randomly
        idx = np.random.randint(0, N)
        # Gaussian step
        step = np.random.normal(0, step_scale, size=2)
        new_centers = current_centers.copy()
        new_centers[idx] += step
        
        # Clamp to boundaries to keep valid (or allow invalid and penalize?)
        # Standard packing: must be inside. If we clamp, we might get stuck.
        # Better: Allow moves outside? No, the problem requires them inside.
        # If a move goes outside, the radius becomes 0 (or invalid).
        # Let's clamp to the valid region [-0.5, 0.5] to ensure we always have a valid configuration.
        # This restricts the search space but avoids invalid states.
        new_centers[idx] = np.clip(new_centers[idx], -0.5, 0.5)
        
        # Calculate new radius
        new_r = _compute_radius(new_centers)
        
        # Calculate delta
        delta = new_r - current_r
        
        # Metropolis criterion
        if delta > 0:
            # Accept
            current_centers = new_centers
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_centers = current_centers.copy()
        else:
            # Reject or accept with probability exp(delta/T)
            # delta is negative
            if np.random.rand() < np.exp(delta / T):
                current_centers = new_centers
                current_r = new_r
                if current_r > best_r:
                    best_r = current_r
                    best_centers = current_centers.copy()
        
        # Cool down
        T *= alpha
        if T < 1e-6:
            T = 1e-6
            
    return best_centers, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # 5s margin for 120s limit (or 150s if that's the hard kill, but 120s is specified in text)
    # The prompt says "finish within 120 s". The budget says "KILLS ... after 150s".
    # I will use 115s to be safe.
    
    best_overall_centers = None
    best_overall_r = 0.0
    
    # Strategy: Multiple restarts with different initializations
    # 1. Hex grid
    # 2. Random
    # 3. Random
    # ...
    
    # Allocate time budget to each restart
    # Let's do 5 restarts of 20s each? Or adaptively?
    # Adaptive: Run SA until time is up, but keep best.
    # Since SA converges, restarting helps escape local minima.
    
    # Restart 1: Hex
    centers = _generate_hex_grid(N)
    r = _compute_radius(centers)
    if r > best_overall_r:
        best_overall_r = r
        best_overall_centers = centers.copy()
    
    # Run SA on Hex
    time_per_restart = 20.0
    # Adjust time per restart based on remaining time
    # We have ~115s. 5 restarts * 20s = 100s. Leaves 15s for final polish/eval.
    
    for restart in range(5):
        if time.time() > deadline:
            break
            
        # Choose initialization
        if restart == 0:
            # Already did hex, maybe perturb it? Or just run SA on it.
            # The previous block ran SA on hex? No, I put SA in a separate function.
            # Let's run SA on the current best (which is hex initially).
            init_centers = best_overall_centers.copy()
        else:
            # Random initialization
            init_centers = _generate_random(N)
        
        # Run SA for a specific time slice
        # We need to pass a time limit to SA.
        # Let's reserve the last 5 seconds for final evaluation and cleanup.
        current_deadline = deadline - 5.0
        sa_time_limit = min(20.0, max(0, current_deadline - time.time()))
        
        if sa_time_limit <= 0:
            break
            
        sa_centers, sa_r = _simulate_annealing(init_centers, sa_time_limit)
        
        if sa_r > best_overall_r:
            best_overall_r = sa_r
            best_overall_centers = sa_centers.copy()
            
        # Optional: Local polish on the best found so far
        # The SA already does local moves.
        
    # Final check and return
    if best_overall_centers is None:
        # Fallback
        best_overall_centers = _generate_hex_grid(N)
        
    # Ensure validity
    r_final = _compute_radius(best_overall_centers)
    if r_final <= 0:
        # Something went wrong, return a safe grid
        best_overall_centers = _generate_hex_grid(N)
        
    return best_overall_centers

# EVOLVE-BLOCK-END
