import numpy as np
import time
from scipy.optimize import minimize

N = 50

def compute_radius(points):
    """
    Computes the maximum radius r such that circles of radius r centered at points
    are inside the unit circle and do not overlap.
    """
    if points is None or len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
        
    # Distance to boundary: min(1 - |c_i|)
    dists_to_boundary = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(dists_to_boundary)
    
    if min_boundary <= 0:
        return 0.0
    
    # Minimum distance between centers
    # Use numpy broadcasting for N=50 (fast enough)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Set diagonal to infinity to ignore self-distance
    np.fill_diagonal(dists, np.inf)
    
    min_dist = np.min(dists)
    
    # The radius is limited by half the minimum distance between centers
    # and the minimum distance to the boundary
    r = min(min_dist / 2.0, min_boundary)
    
    return r if r > 0 else 0.0

def objective(x):
    """
    Objective function for scipy.optimize.
    We want to MAXIMIZE the minimum radius, so we MINIMIZE the negative radius.
    To make the optimization smoother and avoid the kink at the minimum,
    we can use a soft-min approximation or just the negative of the exact radius.
    The exact radius is non-smooth (min of smooth functions), but L-BFGS-B
    can often handle it if we provide gradients or use a robust method.
    However, without gradients, Nelder-Mead or Powell might be better for
    small N, but N=50 is 100 dimensions.
    
    A better approach for packing problems is to maximize the minimum distance
    using a potential energy function or a direct maximization of the min distance.
    
    Let's stick to maximizing r.
    r(x) = min( min_{i<j} |x_i - x_j|/2, min_i (1 - |x_i|) )
    
    We want to minimize -r(x).
    """
    points = x.reshape(N, 2)
    
    # Check validity quickly
    # If any point is outside, penalty
    norms = np.linalg.norm(points, axis=1)
    if np.any(norms > 1.0):
        return 1.0 + np.max(norms) # Large penalty
    
    dists_to_boundary = 1.0 - norms
    min_boundary = np.min(dists_to_boundary)
    
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    r = min(min_dist / 2.0, min_boundary)
    
    return -r

def objective_grad(x):
    """
    Compute gradient of -r(x).
    This is tricky because r(x) is the minimum of several functions.
    The gradient of the minimum is the gradient of the function that achieves the minimum.
    If there are multiple functions achieving the minimum, we can take the average of their gradients
    or just pick one (subgradient).
    
    Let f_i(x) = 1 - |x_i| for boundary constraints.
    Let g_{ij}(x) = |x_i - x_j| / 2 for pairwise constraints.
    
    r = min( min_i f_i, min_{i<j} g_{ij} )
    
    Let k be the index of the constraint that is active (achieves the minimum).
    If it's a boundary constraint i:
        grad f_i = -x_i / |x_i| (for x_i != 0)
        grad (-r) = - grad f_i = x_i / |x_i|
        
    If it's a pairwise constraint (i,j):
        grad g_{ij} = 0.5 * (x_i - x_j) / |x_i - x_j| for x_i
                    = 0.5 * (x_j - x_i) / |x_i - x_j| for x_j
        grad (-r) = - grad g_{ij}
        
    We need to find the active constraint(s).
    """
    points = x.reshape(N, 2)
    
    norms = np.linalg.norm(points, axis=1)
    dists_to_boundary = 1.0 - norms
    min_boundary = np.min(dists_to_boundary)
    
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    r = min(min_dist / 2.0, min_boundary)
    
    grad = np.zeros_like(x)
    
    # Find active constraints
    # Boundary
    active_boundary_indices = np.where(np.isclose(dists_to_boundary, min_boundary, atol=1e-8))[0]
    # Pairwise
    active_pair_indices = np.where(np.isclose(dists, min_dist, atol=1e-8))
    
    num_active = 0
    active_grads = []
    
    if min_boundary <= min_dist / 2.0:
        # Boundary is the limiting factor
        for idx in active_boundary_indices:
            if norms[idx] > 1e-8:
                # grad f_i = -x_i / |x_i|
                # grad -r = - grad f_i = x_i / |x_i|
                g = points[idx] / norms[idx]
                active_grads.append(g)
                num_active += 1
            else:
                # At origin, gradient is undefined, pick 0 or random?
                active_grads.append(np.zeros(2))
                num_active += 1
    else:
        # Pairwise is the limiting factor
        # We need to handle the fact that dists is a matrix.
        # active_pair_indices gives arrays of row and col indices.
        rows, cols = active_pair_indices
        # Ensure i < j to avoid double counting and sign issues?
        # Actually, g_{ij} = g_{ji}. The gradient for x_i is 0.5 * (x_i - x_j)/d.
        # The gradient for x_j is 0.5 * (x_j - x_i)/d.
        # Since we are summing gradients for the objective -r, and r is the min,
        # we should sum the subgradients of the active constraints.
        
        # To avoid double counting (i,j) and (j,i), we only consider i < j.
        unique_pairs = set()
        for i, j in zip(rows, cols):
            if i < j:
                unique_pairs.add((i, j))
            elif j < i:
                unique_pairs.add((j, i))
                
        for i, j in unique_pairs:
            d = dists[i, j]
            if d > 1e-8:
                # grad g_{ij} wrt x_i: 0.5 * (x_i - x_j) / d
                # grad g_{ij} wrt x_j: 0.5 * (x_j - x_i) / d
                # grad (-r) = - grad g_{ij}
                # So for x_i: -0.5 * (x_i - x_j) / d = 0.5 * (x_j - x_i) / d
                # For x_j: -0.5 * (x_j - x_i) / d = 0.5 * (x_i - x_j) / d
                
                vec = points[i] - points[j]
                g_i = -0.5 * vec / d
                g_j = -0.5 * (-vec) / d # Wait.
                
                # Let's re-derive carefully.
                # g_{ij}(x) = ||x_i - x_j|| / 2
                # d/dx_i g_{ij} = (x_i - x_j) / (2 * ||x_i - x_j||)
                # d/dx_j g_{ij} = (x_j - x_i) / (2 * ||x_j - x_i||) = -(x_i - x_j) / (2 * d)
                
                # We are minimizing -r.
                # grad (-r) = - grad r.
                # If r = g_{ij}, then grad (-r) = - grad g_{ij}.
                
                g_i = - (points[i] - points[j]) / (2 * d)
                g_j = - (points[j] - points[i]) / (2 * d)
                
                active_grads.append(g_i)
                active_grads.append(g_j)
                num_active += 2 # Two variables involved
                
    if num_active > 0:
        # Average the subgradients
        grad_flat = np.zeros(2 * N)
        for k, g in enumerate(active_grads):
            # Determine which variable this gradient belongs to?
            # This implementation is messy because I mixed boundary (1 var) and pairwise (2 vars).
            # Let's rewrite this more cleanly.
            pass
            
    # This manual gradient is error-prone. Let's use a numerical gradient or a simpler method.
    # For N=50, numerical gradient is 100 evaluations per step.
    # L-BFGS-B with numerical gradient might be slow but robust.
    # Or just use Nelder-Mead which doesn't need gradients.
    
    # Let's return 0 for now and use a gradient-free method or numerical.
    return None

def initial_hexagonal():
    """
    Generates an initial set of points using a hexagonal lattice.
    """
    spacing = 0.22 # Slightly larger to ensure non-overlap initially?
    points = []
    
    max_k = 10
    for j in range(-max_k, max_k + 1):
        for i in range(-max_k, max_k + 1):
            x = i * spacing + (j % 2) * spacing * 0.5
            y = j * spacing * np.sqrt(3) / 2
            if x**2 + y**2 < 1.0:
                points.append([x, y])
                
    points = np.array(points)
    
    dists = np.linalg.norm(points, axis=1)
    indices = np.argsort(dists)
    
    if len(points) >= N:
        selected = points[indices[:N]]
    else:
        needed = N - len(points)
        random_pts = np.random.uniform(-0.5, 0.5, size=(needed, 2))
        selected = np.vstack([points, random_pts])
        
    return selected

def anneal(initial_points, time_limit=140.0, seed=0):
    """
    Simulated Annealing to maximize the packing radius.
    """
    np.random.seed(seed)
    
    points = initial_points.copy()
    current_r = compute_radius(points)
    best_r = current_r
    best_points = points.copy()
    
    # Annealing parameters
    T = 0.01
    T_min = 1e-6
    cooling_rate = 0.999
    step_size = 0.05
    
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        if T < T_min:
            break
            
        idx = np.random.randint(N)
        move = np.random.normal(0, step_size, size=2)
        
        new_points = points.copy()
        new_points[idx] += move
        
        new_r = compute_radius(new_points)
        
        if new_r > current_r:
            points = new_points
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_points = points.copy()
        else:
            delta_r = new_r - current_r
            if delta_r > -T * 10:
                prob = np.exp(delta_r / T)
                if np.random.random() < prob:
                    points = new_points
                    current_r = new_r
        
        T *= cooling_rate
        
    return best_points, best_r

def optimize_scipy(initial_points, time_limit=30.0):
    """
    Use scipy.optimize to refine the packing.
    """
    x0 = initial_points.flatten()
    
    # We want to maximize r, so minimize -r.
    # We will use Nelder-Mead as it is robust for non-smooth functions.
    # However, N=50 -> 100 dimensions. Nelder-Mead might be slow.
    # Let's try L-BFGS-B with numerical gradient.
    
    start_time = time.time()
    
    # Define a wrapper to check time
    def time_check(*args):
        if time.time() - start_time > time_limit:
            raise TimeoutError()
            
    # Nelder-Mead is often better for packing problems with min objectives.
    # But 100 dims is high.
    # Let's try a multi-start approach with a simpler local optimizer.
    
    # Alternative: Use a "push" algorithm.
    # If circles overlap, push them apart.
    # If a circle is outside, push it in.
    
    return initial_points, compute_radius(initial_points)

def push_algorithm(points, time_limit=20.0):
    """
    Simple iterative pushing algorithm.
    """
    points = points.copy()
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        moved = False
        # Sort by distance to center? Or just iterate.
        for i in range(N):
            # Check boundary
            norm = np.linalg.norm(points[i])
            if norm > 1.0:
                # Push in
                points[i] = points[i] / norm
                moved = True
                
            # Check overlaps
            for j in range(i + 1, N):
                diff = points[i] - points[j]
                dist = np.linalg.norm(diff)
                # We don't know the exact radius yet, but we can assume a target radius
                # or just push if they are very close.
                # A common heuristic: push if dist < 2 * r_est.
                # Let's estimate r from current best.
                # Or just push if dist < 0.2 (arbitrary threshold).
                # Better: push proportional to overlap.
                
                # Let's assume a target radius r_target = 0.12 (guess).
                # If dist < 2 * r_target, push apart.
                r_target = 0.12
                if dist < 2 * r_target and dist > 1e-8:
                    # Push vector
                    push_vec = diff / dist
                    # Push amount
                    # We want to increase dist to 2 * r_target.
                    # push = (2 * r_target - dist) / 2 for each?
                    # Let's just move them slightly.
                    step = 0.01
                    points[i] += push_vec * step
                    points[j] -= push_vec * step
                    moved = True
                elif dist < 1e-8:
                    # Coincident, push random
                    points[i] += np.random.normal(0, 0.01, 2)
                    points[j] -= np.random.normal(0, 0.01, 2)
                    moved = True
                    
        if not moved:
            break
            
    # Ensure all points are inside
    for i in range(N):
        norm = np.linalg.norm(points[i])
        if norm > 1.0:
            points[i] = points[i] / norm
            
    return points

def construct():
    deadline = time.time() + 145.0
    
    best_overall_r = 0.0
    best_overall_points = None
    
    # 1. Hexagonal start + Push + Anneal
    try:
        init_pts = initial_hexagonal()
        init_pts = push_algorithm(init_pts, time_limit=5.0)
        r_check = compute_radius(init_pts)
        
        if r_check > 0:
            p, r = anneal(init_pts, time_limit=40.0, seed=42)
            if r > best_overall_r:
                best_overall_r = r
                best_overall_points = p
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_overall_points if best_overall_points is not None else initial_hexagonal()
        
    # 2. Ring start + Push + Anneal
    try:
        points = np.zeros((N, 2))
        points[0] = [0, 0]
        count = 1
        ring_radius = 0.15
        ring_step = 0.15
        
        while count < N:
            n_in_ring = int(2 * np.pi * ring_radius / 0.25)
            if n_in_ring < 1:
                n_in_ring = 1
            
            for k in range(n_in_ring):
                if count >= N:
                    break
                angle = 2 * np.pi * k / n_in_ring
                points[count, 0] = ring_radius * np.cos(angle)
                points[count, 1] = ring_radius * np.sin(angle)
                count += 1
            
            ring_radius += ring_step
            
        points += np.random.normal(0, 0.01, size=(N, 2))
        
        points = push_algorithm(points, time_limit=5.0)
        r_check = compute_radius(points)
        
        if r_check > 0:
            p, r = anneal(points, time_limit=40.0, seed=123)
            if r > best_overall_r:
                best_overall_r = r
                best_overall_points = p
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_overall_points if best_overall_points is not None else initial_hexagonal()

    # 3. Random start + Push + Anneal
    try:
        if time.time() < deadline - 40:
            pts = []
            while len(pts) < N:
                x, y = np.random.uniform(-1, 1, 2)
                if x**2 + y**2 < 1.0:
                    pts.append([x, y])
                    
            points = np.array(pts)
            points = push_algorithm(points, time_limit=5.0)
            
            r_check = compute_radius(points)
            if r_check > 0:
                p, r = anneal(points, time_limit=30.0, seed=999)
                if r > best_overall_r:
                    best_overall_r = r
                    best_overall_points = p
    except Exception:
        pass

    # Final Polish
    if best_overall_points is not None and time.time() < deadline - 5:
        p, r = anneal(best_overall_points, time_limit=5.0, seed=0)
        if r > best_overall_r:
            best_overall_r = r
            best_overall_points = p
            
    return best_overall_points if best_overall_points is not None else initial_hexagonal()

if __name__ == "__main__":
    pts = construct()
    r = compute_radius(pts)
    print(f"Radius: {r}")
    print(f"Shape: {pts.shape}")
