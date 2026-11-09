import numpy as np
import time

N = 50
TIME_LIMIT = 145.0

def _score(points):
    """Compute the score: largest common radius."""
    if len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    # Check all points inside unit circle
    norms = np.hypot(points[:, 0], points[:, 1])
    if np.any(norms > 1.0):
        return 0.0
    # Min distance from any point to boundary
    boundary_min = 1.0 - np.max(norms)
    # Min pairwise distance / 2
    if N > 1:
        d = np.sqrt(np.sum((points[:, np.newaxis, :] - points[np.newaxis, :, :])**2, axis=-1))
        np.fill_diagonal(d, np.inf)
        min_pair = np.min(d) / 2.0
    else:
        min_pair = boundary_min
    return min(min_pair, boundary_min)

def _repulsion_dynamics(points, time_limit, alpha=0.05, k=5, rng=None):
    """
    Directed repulsion dynamics:
    1. Calculate repulsive forces from k nearest neighbors.
    2. Calculate boundary forces.
    3. Update positions.
    4. Stochastic k-NN perturbation: pick a point, move it away from its k nearest neighbors.
    """
    if rng is None:
        rng = np.random.default_rng(42)
        
    current = points.copy()
    current_score = _score(current)
    best = current.copy()
    best_score = current_score
    
    # We use a fixed number of iterations or time-based
    # Since each step is fast (vectorized k-NN via cdist or similar), we can do many.
    # Let's use time-based to be safe.
    
    # Precompute indices for k-NN? No, positions change.
    # We will compute k-NN on the fly.
    
    # To speed up, we can approximate forces using all pairs if N is small, 
    # but N=50 is small enough that O(N^2) is fine.
    # Actually, "k-NN" force is specific. Let's just use all-pairs repulsion 
    # but weighted by inverse distance to emphasize close pairs, 
    # OR implement proper k-NN.
    
    # Proper k-NN force:
    # For each point i, find k nearest neighbors.
    # Force = sum_{j in kNN(i)} (r_i - r_j) / |r_i - r_j|^2
    
    # This is O(N^2) per step.
    
    steps = 0
    while time.time() < time_limit:
        # 1. Compute k-NN forces
        # Distances
        d = np.sqrt(np.sum((current[:, np.newaxis, :] - current[np.newaxis, :, :])**2, axis=-1))
        np.fill_diagonal(d, np.inf)
        
        # Find k nearest neighbors for each point
        # argsort along axis 1
        k_idx = np.argsort(d, axis=1)[:, :k]
        
        # Compute forces
        forces = np.zeros_like(current)
        for i in range(N):
            neighbors = k_idx[i]
            diffs = current[i] - current[neighbors]
            dists = d[i, neighbors]
            # Avoid division by zero
            dists = np.where(dists > 1e-8, dists, 1.0)
            # Force is proportional to diff / dist^2
            f = np.sum(diffs / (dists[:, np.newaxis]**2), axis=0)
            forces[i] = f
            
        # 2. Boundary forces
        norms = np.hypot(current[:, 0], current[:, 1])
        # If outside, push in. If close to boundary, push in slightly to keep inside.
        # Let's use a soft boundary force: if r > 0.9, push towards center.
        # Or hard constraint: project back.
        # Hard projection is better for "inside" constraint.
        # But dynamics might oscillate.
        # Let's use a force that increases as we approach boundary.
        # F_boundary = -grad(r^2) = -2*r * r_hat
        # Only if r > r_threshold?
        # Let's just project back if outside.
        
        # Update positions
        current = current + alpha * forces
        
        # Project back to unit disk
        norms = np.hypot(current[:, 0], current[:, 1])
        mask = norms > 1.0
        if np.any(mask):
            current[mask] = current[mask] / norms[mask, np.newaxis]
        
        # 3. Stochastic k-NN perturbation
        # Pick a random point
        i = rng.integers(0, N)
        # Find its k nearest neighbors
        d_i = d[i]
        k_idx_i = np.argsort(d_i)[:k]
        
        # Move it away from them
        diffs = current[i] - current[k_idx_i]
        dists = d_i[k_idx_i]
        dists = np.where(dists > 1e-8, dists, 1.0)
        push_dir = np.sum(diffs / (dists[:, np.newaxis]**2), axis=0)
        
        # Normalize push_dir
        push_norm = np.linalg.norm(push_dir)
        if push_norm > 1e-8:
            push_dir /= push_norm
        
        # Move it
        step_size = 0.05
        new_pos = current[i] + step_size * push_dir
        
        # Check if this improves score or is acceptable
        # We don't want to accept worse moves blindly in this phase, 
        # but we need to escape local minima.
        # Let's use a simple hill-climbing with occasional random jumps.
        
        # For now, let's just accept if it stays inside.
        # This is a "relaxation" step.
        
        # To escape local minima, we need a "kick".
        # If we get stuck (score doesn't improve for X steps), do a random kick.
        
        # Let's track score
        new_score = _score(current)
        
        # This is getting complex. Let's simplify:
        # Alternate between:
        # A) Relaxation (repulsion dynamics)
        # B) Kick (random point, random move)
        
        # Actually, the repulsion dynamics *is* the optimization.
        # The "kick" is to escape local minima.
        
        # Let's implement a simple SA-like acceptance on the kick.
        # But we are doing deterministic relaxation.
        
        # Let's restart the relaxation from a perturbed state if stuck.
        
        # For this implementation, let's just run the relaxation and see.
        # If it converges to a local min, we might need a different strategy.
        
        # Let's add a small random noise to the forces to help escape.
        # current = current + alpha * (forces + noise)
        
        # Let's try:
        # 1. Run relaxation for a few steps.
        # 2. Evaluate score.
        # 3. If score improves, keep.
        # 4. If not, try a random kick.
        
        # This loop is doing that implicitly? No.
        
        # Let's refactor:
        # The function should return the best point found during the process.
        
        # We already update best.
        
        # Let's add a "kick" every 100 steps.
        if steps % 100 == 0:
            i = rng.integers(0, N)
            noise = rng.normal(0, 0.1, 2)
            current[i] += noise
            # Project back
            norms = np.hypot(current[:, 0], current[:, 1])
            mask = norms > 1.0
            if np.any(mask):
                current[mask] = current[mask] / norms[mask, np.newaxis]
        
        steps += 1
        
        # Update best
        if new_score > best_score:
            best_score = new_score
            best = current.copy()
            
        # Check for convergence
        if steps % 1000 == 0:
            # If no improvement in last 1000 steps, we might be stuck.
            # But we don't track that here.
            pass

    return best, best_score

def _local_optimize(points):
    """
    Run the repulsion dynamics for a short time to polish.
    """
    # Use a tighter alpha and more steps
    # But time is limited.
    # Let's just use the same function with a shorter time limit.
    # Actually, we can call _repulsion_dynamics with a specific end time.
    # But we don't know the global start time here.
    
    # Let's create a dedicated polish function.
    # For now, just return points. The main loop will handle polishing.
    return points

def _hexagonal_initial(spacing):
    """Generate hexagonal lattice points inside unit disk."""
    points = []
    k = int(1.0 / spacing) + 2
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if x * x + y * y <= (1.0 - spacing / 2.0) ** 2:
                points.append([x, y])
    points = np.array(points)
    if len(points) < N:
        return None
    # Select N closest to origin
    norms = np.hypot(points[:, 0], points[:, 1])
    idx = np.argsort(norms, kind='stable')[:N]
    return points[idx]

def construct():
    start_time = time.time()
    
    best_points = None
    best_score = 0.0
    
    rng = np.random.default_rng(42)
    
    # Initializations
    initializations = []
    
    # Hexagonal
    for spacing in [0.20, 0.22, 0.24, 0.26, 0.28]:
        pts = _hexagonal_initial(spacing)
        if pts is not None:
            initializations.append(pts)
            
    # Ring
    def _ring_initial():
        points = []
        points.append([0.0, 0.0])
        radii = [0.3, 0.55, 0.75, 0.9]
        counts = [6, 12, 18, 13]
        for r, n in zip(radii, counts):
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            for a in angles:
                points.append([r * np.cos(a), r * np.sin(a)])
        return np.array(points[:N])
    initializations.append(_ring_initial())
    
    # Random
    for _ in range(3):
        points = []
        for _ in range(N):
            while True:
                x = rng.uniform(-0.9, 0.9)
                y = rng.uniform(-0.9, 0.9)
                if x * x + y * y <= 0.81:
                    points.append([x, y])
                    break
        initializations.append(np.array(points))
        
    # Strategy:
    # 1. For each init, run repulsion dynamics.
    # 2. Keep track of global best.
    
    # Time budget:
    # We have ~145s.
    # Let's allocate time per init.
    # If we have 9 inits, ~15s each.
    
    num_inits = len(initializations)
    time_per_init = (TIME_LIMIT - 10) / num_inits  # Reserve 10s for final polish
    
    for i, init_points in enumerate(initializations):
        if time.time() - start_time > TIME_LIMIT - 10:
            break
            
        end_time = time.time() + time_per_init
        # Ensure we don't exceed global limit
        if end_time > TIME_LIMIT - 10:
            end_time = TIME_LIMIT - 10
            
        dyn_points, dyn_score = _repulsion_dynamics(init_points, end_time, alpha=0.05, k=5, rng=rng)
        
        if dyn_score > best_score:
            best_score = dyn_score
            best_points = dyn_points.copy()
            
    # Final Polish Phase
    if best_points is not None:
        remaining_time = TIME_LIMIT - (time.time() - start_time)
        if remaining_time > 5:
            # Run a more aggressive local optimization
            # Use smaller alpha, more steps
            polished_points, polished_score = _repulsion_dynamics(
                best_points, 
                time.time() + remaining_time - 2, 
                alpha=0.01, 
                k=5, 
                rng=rng
            )
            if polished_score > best_score:
                best_score = polished_score
                best_points = polished_points.copy()
                
            # One more polish with even smaller alpha
            if time.time() - start_time < TIME_LIMIT - 2:
                polished_points2, polished_score2 = _repulsion_dynamics(
                    best_points, 
                    TIME_LIMIT - 2, 
                    alpha=0.001, 
                    k=5, 
                    rng=rng
                )
                if polished_score2 > best_score:
                    best_score = polished_score2
                    best_points = polished_points2.copy()

    # Fallback
    if best_points is None:
        best_points = _hexagonal_initial(0.25)
        if best_points is None:
            best_points = np.zeros((N, 2))

    return best_points
