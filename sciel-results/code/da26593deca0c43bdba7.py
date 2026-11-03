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
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    r = min(min_dist / 2.0, min_boundary)
    return r if r > 0 else 0.0

def hexagonal_init():
    """
    Generate a hexagonal packing initial configuration.
    This is a high-density packing for infinite plane, which is a good prior.
    """
    points = np.zeros((N, 2))
    # Hexagonal lattice with spacing d = 2r (approx).
    # We want to fit ~50 points in unit circle.
    # Area of unit circle = pi. Area of 50 circles = 50 * pi * r^2.
    # r^2 ~ 1/50 -> r ~ 0.14.
    # Let's start with r = 0.12, so d = 0.24.
    
    # Generate hex grid points
    # Hex grid coords:
    # x = d * (i + 0.5 * (j % 2))
    # y = d * (sqrt(3)/2 * j)
    
    d = 0.24
    pts = []
    # Generate enough points to cover the unit circle
    # Radius of coverage needs to be > 1
    max_j = int(2 / (d * np.sqrt(3)/2)) + 2
    max_i = int(2 / d) + 2
    
    for j in range(-max_j, max_j + 1):
        for i in range(-max_i, max_i + 1):
            x = d * (i + 0.5 * (j % 2))
            y = d * (np.sqrt(3)/2 * j)
            if np.sqrt(x*x + y*y) < 1.0:
                pts.append([x, y])
    
    pts = np.array(pts)
    
    # We might have more or less than N points.
    # If more, randomly select N.
    # If less, add random points inside.
    
    if len(pts) >= N:
        # Randomly select N points
        indices = np.random.choice(len(pts), N, replace=False)
        points = pts[indices]
    else:
        points = np.zeros((N, 2))
        points[:len(pts)] = pts
        # Fill rest with random points
        for k in range(len(pts), N):
            while True:
                p = np.random.uniform(-1, 1, 2)
                if np.linalg.norm(p) < 0.9: # Keep inside
                    points[k] = p
                    break
    
    # Add small noise to break symmetry and avoid degenerate states
    points += np.random.normal(0, 0.01, (N, 2))
    
    # Ensure all points are strictly inside
    norms = np.linalg.norm(points, axis=1)
    out_mask = norms > 0.95
    if np.any(out_mask):
        points[out_mask] = points[out_mask] / norms[out_mask, np.newaxis] * 0.95
        
    return points

def anneal(points, time_budget=60.0):
    """
    Simulated Annealing to maximize the minimum radius.
    Objective: Minimize -r(points).
    We use a potential energy E that penalizes overlaps and boundary violations.
    E = sum_{i<j} max(0, 2r - d_ij)^2 + sum_i max(0, r - (1-|p_i|))^2
    But r is variable.
    
    Alternative: 
    We want to maximize r.
    Let's fix r and check feasibility? No, r changes.
    
    Better: 
    Minimize Energy E(points, r) where r is a parameter we want to increase.
    Actually, standard SA for packing minimizes a potential energy of overlaps 
    assuming a fixed radius, or maximizes the minimum distance.
    
    Let's maximize the minimum distance D = min(min_dist, 2*min_boundary).
    This is equivalent to maximizing r = D/2.
    
    We can use a soft-min of the distances to make it smooth for SA? 
    Or just use the exact min and accept moves based on probability.
    
    Since N=50, computing exact min is O(N^2) which is fast.
    We will use exact min.
    
    Move: Randomly perturb one point (or a few) by a small Gaussian step.
    Accept if new_r > old_r (Hill Climbing).
    Accept if new_r < old_r with probability exp((new_r - old_r) / T).
    
    T (temperature) decays over time.
    """
    
    current_points = points.copy()
    current_r = compute_radius(current_points)
    
    best_points = current_points.copy()
    best_r = current_r
    
    # Initial temperature
    # Scale of r is ~0.1. Delta r is small.
    # Let's set T such that we accept some worse moves initially.
    T = 0.01
    T_min = 1e-6
    
    # Step size for perturbation
    step_size = 0.05
    
    start_time = time.time()
    
    while time.time() - start_time < time_budget:
        # Pick a random point to move
        idx = np.random.randint(N)
        
        # Generate a new position
        # Gaussian perturbation
        delta = np.random.normal(0, step_size, 2)
        new_point = current_points[idx] + delta
        
        # Keep inside unit circle (hard constraint for validity, but SA can handle soft)
        # For simplicity, if it goes outside, reject or project?
        # Projecting is better for SA to stay in feasible region of boundary.
        norm = np.linalg.norm(new_point)
        if norm > 1.0:
            new_point = new_point / norm * 1.0
            # Actually, if we project, we might be on the boundary.
            # Let's allow it to be slightly outside and let the penalty handle it?
            # No, compute_radius returns 0 if outside. That's a hard penalty.
            # So if it goes outside, r becomes 0. That's a huge drop.
            # Better to constrain the move to stay inside or handle it.
            # Let's just reject moves that go outside to keep search efficient.
            continue
            
        # Calculate new r
        old_point = current_points[idx].copy()
        current_points[idx] = new_point
        new_r = compute_radius(current_points)
        
        # Acceptance criterion
        if new_r >= current_r:
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best_points = current_points.copy()
        else:
            # Probabilistic acceptance
            # delta_r = new_r - current_r (negative)
            delta_r = new_r - current_r
            # Avoid overflow in exp
            if delta_r < -100 * T:
                # Reject
                current_points[idx] = old_point
            else:
                p = np.exp(delta_r / T)
                if np.random.rand() < p:
                    current_r = new_r
                else:
                    current_points[idx] = old_point
        
        # Update temperature
        # Exponential decay
        # Or linear decay?
        # Let's use a cooling schedule based on iterations or time.
        # Simple: T *= 0.999
        T *= 0.999
        
        # Adapt step size?
        # If we are stuck, increase step size?
        # If we are jumping around, decrease?
        # Simple heuristic: if best_r hasn't improved in last X steps, increase step size slightly.
        # For now, keep it simple.
        
    return best_points

def polish(points):
    """
    Final gradient-based polish using L-BFGS-B on a smooth objective.
    """
    if points is None:
        return None
        
    def obj(x):
        pts = x.reshape(N, 2)
        r = compute_radius(pts)
        # We want to maximize r, so minimize -r.
        # But -r is non-smooth.
        # Use soft-min for smoothness in polish.
        # Or just use -r and hope L-BFGS-B handles the kink?
        # L-BFGS-B is a quasi-Newton method, it might struggle with non-smoothness.
        # Let's use a smooth approximation: -log-sum-exp of constraints.
        # Actually, let's just try minimizing -r directly first. If it fails, use smooth.
        # For polish, a small number of iterations on -r might be enough to snap to the boundary.
        
        # Let's use a smooth objective for polish to be safe.
        beta = 100.0
        norms = np.linalg.norm(pts, axis=1)
        b = 1.0 - norms
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        d = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(d, np.inf)
        
        E_pair = np.exp(-beta * d / 2.0)
        np.fill_diagonal(E_pair, 0.0)
        E_bound = np.exp(-beta * b)
        S = np.sum(E_pair) + np.sum(E_bound)
        val = np.log(S) / beta
        return val

    def grad(x):
        pts = x.reshape(N, 2)
        beta = 100.0
        norms = np.linalg.norm(pts, axis=1)
        b = 1.0 - norms
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        d = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(d, np.inf)
        
        E_pair = np.exp(-beta * d / 2.0)
        np.fill_diagonal(E_pair, 0.0)
        E_bound = np.exp(-beta * b)
        S = np.sum(E_pair) + np.sum(E_bound)
        
        grad = np.zeros_like(x)
        
        # Boundary grad
        inv_norms = np.zeros_like(norms)
        mask_nonzero = norms > 1e-8
        inv_norms[mask_nonzero] = 1.0 / norms[mask_nonzero]
        grad_bound = (E_bound * beta)[:, np.newaxis] * pts * inv_norms[:, np.newaxis]
        
        # Pairwise grad
        d_safe = d.copy()
        np.fill_diagonal(d_safe, 1.0)
        factor = (E_pair * (-beta / 2.0)) / d_safe
        np.fill_diagonal(factor, 0.0)
        
        row_sums = np.sum(factor, axis=1)
        weighted_points = factor @ pts
        grad_pair = pts * row_sums[:, np.newaxis] - weighted_points
        
        return (grad_bound + grad_pair).flatten()

    x0 = points.flatten()
    try:
        res = minimize(obj, x0, jac=grad, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-12})
        new_pts = res.x.reshape(N, 2)
        if np.all(np.isfinite(new_pts)):
            new_r = compute_radius(new_pts)
            old_r = compute_radius(points)
            if new_r > old_r:
                return new_pts
    except Exception:
        pass
    return points

def construct():
    deadline = time.time() + 145.0
    
    best_overall_r = 0.0
    best_overall_points = None
    
    # 1. Hexagonal Init + Anneal
    pts = hexagonal_init()
    
    # Allocate time: 80s for Anneal, 20s for Polish, 15s for safety/multiple restarts if needed
    # Let's do 1 main anneal run, then polish.
    # If time permits, do a second anneal run from a different init.
    
    t_start = time.time()
    anneal_budget = 80.0
    if time.time() - t_start < anneal_budget:
        pts_annealed = anneal(pts, time_budget=anneal_budget)
        r = compute_radius(pts_annealed)
        if r > best_overall_r:
            best_overall_r = r
            best_overall_points = pts_annealed
    
    # 2. Polish
    if time.time() < deadline - 10 and best_overall_points is not None:
        pts_polished = polish(best_overall_points)
        r = compute_radius(pts_polished)
        if r > best_overall_r:
            best_overall_r = r
            best_overall_points = pts_polished
            
    # 3. If time remains, try a different init (e.g. rings) and anneal again?
    # The current program uses rings. Let's try rings + anneal as a second attempt.
    if time.time() < deadline - 30:
        # Simple ring init
        pts_ring = np.zeros((N, 2))
        pts_ring[0] = [0, 0]
        count = 1
        # 6, 12, 18, 14
        rings = [(0.25, 6), (0.50, 12), (0.75, 18), (0.95, 14)]
        for radius, capacity in rings:
            for k in range(capacity):
                if count >= N:
                    break
                angle = 2 * np.pi * k / capacity
                pts_ring[count] = [radius * np.cos(angle), radius * np.sin(angle)]
                count += 1
            if count >= N:
                break
        
        pts_ring += np.random.normal(0, 0.01, (N, 2))
        
        # Anneal with remaining time (minus safety margin for polish)
        remaining = deadline - time.time() - 15 # 15s for polish + safety
        if remaining > 20:
            pts_annealed2 = anneal(pts_ring, time_budget=remaining)
            r = compute_radius(pts_annealed2)
            if r > best_overall_r:
                best_overall_r = r
                best_overall_points = pts_annealed2
                
            # Polish the second result if better
            if time.time() < deadline - 5:
                pts_polished2 = polish(pts_annealed2)
                r = compute_radius(pts_polished2)
                if r > best_overall_r:
                    best_overall_r = r
                    best_overall_points = pts_polished2

    return best_overall_points if best_overall_points is not None else hexagonal_init()

if __name__ == "__main__":
    pts = construct()
    r = compute_radius(pts)
    print(f"Radius: {r}")
    print(f"Shape: {pts.shape}")
