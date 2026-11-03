"""Pack 50 equal circles in a unit square using Delaunay-based Contact Graph Optimization."""
import numpy as np
import time
from scipy.spatial import Delaunay
from scipy.optimize import minimize

N = 50
DEADLINE = 145.0  # seconds, safety margin before 150s kill

def _radius(centers):
    """Compute the max common radius for given centers (exact, for scoring)."""
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Wall radius
    wall_r = np.min(0.5 - np.abs(centers))
    if wall_r < 0:
        return 0.0
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    return min(wall_r, min_pair / 2.0)

def _make_hexagonal_grid(n):
    """Create a hexagonal grid of n points centered in the square."""
    dx = 1.0 / 7.0
    dy = dx * np.sqrt(3) / 2.0
    
    points = []
    for i in range(8):
        y = -0.5 + (i + 0.5) * dy
        for j in range(7):
            x = -0.5 + (j + 0.5) * dx
            if i % 2 == 1:
                x += dx / 2.0
            points.append([x, y])
            
    points = np.array(points)
    
    # Filter to square
    mask = (points[:, 0] >= -0.5) & (points[:, 0] <= 0.5) & (points[:, 1] >= -0.5) & (points[:, 1] <= 0.5)
    points = points[mask]
    
    if len(points) > N:
        # Sort by distance from center to keep a compact blob
        dists = np.sum(points**2, axis=1)
        idx = np.argsort(dists)
        points = points[idx[:N]]
    elif len(points) < N:
        # Fallback: add random points near edges if grid is too small (unlikely for 8x7)
        while len(points) < N:
            points = np.vstack([points, np.random.uniform(-0.4, 0.4, size=(1, 2))])
            
    return points[:N]

def _get_contact_graph(centers, r, margin=0.01):
    """
    Identify active contacts using Delaunay triangulation.
    A contact exists if distance < 2r + margin.
    We use Delaunay to restrict the search space to neighbors, 
    which is crucial for efficiency and correctness in dense packings.
    """
    try:
        tri = Delaunay(centers)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    idx_i, idx_j = simplex[i], simplex[j]
                    d = np.linalg.norm(centers[idx_i] - centers[idx_j])
                    if d < 2*r + margin:
                        edges.add((min(idx_i, idx_j), max(idx_i, idx_j)))
        return list(edges)
    except Exception:
        # Fallback to all pairs if Delaunay fails (e.g., coincident points)
        edges = []
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(centers[i] - centers[j])
                if d < 2*r + margin:
                    edges.append((i, j))
        return edges

def _solve_contact_system(centers, r, max_iter=50, tol=1e-6):
    """
    Solve for displacements to resolve overlaps for a fixed radius r.
    Uses a linearized least-squares approach on the contact graph.
    
    Constraints:
    1. Pairwise: |c_i - c_j| >= 2r
       Linearized: (c_i - c_j) . (delta_i - delta_j) >= 2r - |c_i - c_j|
    2. Wall: |c_k| <= 0.5 - r
       Linearized: c_k . delta_k <= (0.5 - r) - |c_k| (if c_k > 0) etc.
    
    We formulate this as a quadratic program: minimize ||delta||^2 subject to A delta <= b.
    Since scipy doesn't have a direct QP solver in standard imports without extras,
    we use an iterative projected gradient / Gauss-Seidel relaxation on the constraints,
    which is robust for small N.
    """
    centers = centers.copy()
    
    for it in range(max_iter):
        # Identify active constraints
        active_pairs = []
        active_walls = [] # (idx, axis, sign) where sign=1 means c>0, -1 means c<0
        
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(centers[i] - centers[j])
                if d < 2*r + 1e-4:
                    active_pairs.append((i, j, d))
            
            # Walls
            if centers[i, 0] > 0.5 - r:
                active_walls.append((i, 0, 1))
            elif centers[i, 0] < -0.5 + r:
                active_walls.append((i, 0, -1))
            if centers[i, 1] > 0.5 - r:
                active_walls.append((i, 1, 1))
            elif centers[i, 1] < -0.5 + r:
                active_walls.append((i, 1, -1))
        
        if not active_pairs and not active_walls:
            break
            
        # Construct linear system A delta = b
        # We want to find delta such that constraints are satisfied.
        # Minimize ||delta||^2 s.t. A delta >= b (where b is the violation amount)
        
        # Build A (m x 2N) and b (m x 1)
        m = len(active_pairs) + len(active_walls)
        A = np.zeros((m, 2*N))
        b = np.zeros(m)
        
        idx_m = 0
        
        # Pair constraints
        for i, j, d in active_pairs:
            if d < 1e-8:
                # Random nudge
                A[idx_m, 2*i] = 1
                A[idx_m, 2*j] = -1
                b[idx_m] = 2*r
            else:
                dir_vec = (centers[i] - centers[j]) / d
                # (c_i - c_j) . (delta_i - delta_j) >= 2r - d
                A[idx_m, 2*i] = dir_vec[0]
                A[idx_m, 2*i+1] = dir_vec[1]
                A[idx_m, 2*j] = -dir_vec[0]
                A[idx_m, 2*j+1] = -dir_vec[1]
                b[idx_m] = 2*r - d
            idx_m += 1
        
        # Wall constraints
        for i, axis, sign in active_walls:
            # sign * c_i >= sign * (0.5 - r)
            # sign * delta_i >= sign * (0.5 - r) - sign * c_i
            A[idx_m, 2*i + axis] = sign
            b[idx_m] = sign * (0.5 - r) - sign * centers[i, axis]
            idx_m += 1
            
        # Solve least squares: min ||delta|| s.t. A delta >= b
        # This is equivalent to finding the projection of 0 onto the polytope.
        # We can use scipy.optimize.lsq_linear for A delta >= b? No, lsq_linear is for equality/inequality bounds on x.
        # Let's use a simple iterative relaxation (Gauss-Seidel style) which is fast and stable for N=50.
        
        delta = np.zeros(2*N)
        
        # Relaxation steps
        for _ in range(10):
            # Update for pairs
            for i, j, d in active_pairs:
                if d < 1e-8: continue
                dir_vec = (centers[i] - centers[j]) / d
                current_overlap = 2*r - (centers[i] - centers[j]) @ dir_vec
                if current_overlap > 0:
                    # Distribute correction equally
                    corr = current_overlap / 2.0
                    delta[2*i] += corr * dir_vec[0]
                    delta[2*i+1] += corr * dir_vec[1]
                    delta[2*j] -= corr * dir_vec[0]
                    delta[2*j+1] -= corr * dir_vec[1]
            
            # Update for walls
            for i, axis, sign in active_walls:
                current_violation = sign * (0.5 - r) - sign * centers[i, axis]
                if current_violation > 0:
                    delta[2*i + axis] += current_violation
        
        # Apply delta with a damping factor to ensure convergence
        damping = 0.5
        centers += damping * delta.reshape(N, 2)
        
        # Check convergence
        # Re-evaluate violations
        max_violation = 0.0
        for i, j, _ in active_pairs:
            d = np.linalg.norm(centers[i] - centers[j])
            v = 2*r - d
            if v > max_violation: max_violation = v
        for i, axis, sign in active_walls:
            v = sign * (0.5 - r) - sign * centers[i, axis]
            if v > max_violation: max_violation = v
            
        if max_violation < tol:
            break
            
    return centers

def _expand_radius(centers, r_start, r_end, time_limit=10.0):
    """
    Incrementally expand radius and solve for new positions.
    """
    start_time = time.time()
    r = r_start
    step = 0.001
    centers = centers.copy()
    
    while r < r_end and time.time() - start_time < time_limit:
        # Try to expand
        r_new = min(r + step, r_end)
        
        # Solve for configuration that supports r_new
        new_centers = _solve_contact_system(centers, r_new, max_iter=20)
        
        # Verify
        r_actual = _radius(new_centers)
        
        if r_actual >= r_new - 1e-5:
            centers = new_centers
            r = r_new
            # Increase step slightly if we're doing well? Keep constant for stability.
        else:
            # Failed to support r_new, try smaller step or break
            # If we can't even support current r, we are stuck
            if r_actual < r - 1e-4:
                break
            step /= 2.0
            if step < 1e-5:
                break
                
    return centers, r

def _slsqp_polish(centers, time_limit=5.0):
    """
    Final polish using SLSQP to maximize radius exactly.
    Variables: centers (100) + r (1) -> 101 vars.
    Objective: maximize r.
    Constraints:
    1. |c_i - c_j| >= 2r
    2. |c_ik| <= 0.5 - r
    """
    x0 = np.concatenate([centers.ravel(), [_radius(centers)]])
    
    def objective(x):
        return -x[-1] # Minimize -r
        
    def constraints(x):
        c = x[:2*N].reshape(N, 2)
        r = x[-1]
        cons = []
        
        # Pairwise
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(c[i] - c[j])
                cons.append({'type': 'ineq', 'fun': lambda x, i=i, j=j: np.linalg.norm(x[:2*N].reshape(N, 2)[i] - x[:2*N].reshape(N, 2)[j]) - 2*x[-1]})
                
        # Walls
        for i in range(N):
            for k in range(2):
                cons.append({'type': 'ineq', 'fun': lambda x, i=i, k=k: (0.5 - x[-1]) - abs(x[:2*N].reshape(N, 2)[i, k])})
                
        return cons
        
    # SLSQP is slow for many constraints, but N=50 is manageable for a short polish.
    # We restrict maxiter to avoid timeout.
    try:
        res = minimize(objective, x0, method='SLSQP', options={'maxiter': 100, 'ftol': 1e-6})
        if res.success or res.x is not None:
            return res.x[:2*N].reshape(N, 2)
    except Exception:
        pass
    return centers

def construct():
    start_time = time.time()
    best_centers = None
    best_r = 0.0
    
    np.random.seed(42)
    
    # 1. Generate Initial Candidates
    candidates = []
    
    # Hexagonal Grid
    hex_c = _make_hexagonal_grid(N)
    candidates.append(hex_c)
    
    # Square Grid
    ticks_x = (np.arange(7) + 0.5) / 7 - 0.5
    ticks_y = (np.arange(8) + 0.5) / 8 - 0.5
    sq_c = np.array([(x, y) for y in ticks_y for x in ticks_x])[:N]
    candidates.append(sq_c)
    
    # Random
    for _ in range(2):
        candidates.append(np.random.uniform(-0.4, 0.4, size=(N, 2)))
        
    # 2. Process Candidates
    for idx, c_init in enumerate(candidates):
        if time.time() - start_time > DEADLINE - 20:
            break
            
        c_curr = c_init.copy()
        
        # Ensure valid start
        r_start = _radius(c_curr)
        if r_start < 0.01:
             # If start is bad, try to fix it with a small radius first
             c_curr = _solve_contact_system(c_curr, 0.05, max_iter=50)
             r_start = _radius(c_curr)
             
        if r_start <= 0:
            continue
            
        # 3. Radius Expansion
        # Target slightly above known best
        r_target = 0.0720
        
        # Use time budget for expansion
        time_left_exp = min(15.0, DEADLINE - 25 - (time.time() - start_time))
        if time_left_exp > 0:
            c_exp, r_exp = _expand_radius(c_curr, r_start, r_target, time_limit=time_left_exp)
            r_exp = _radius(c_exp)
            
            if r_exp > best_r:
                best_r = r_exp
                best_centers = c_exp.copy()
            
            c_curr = c_exp
            
        # 4. Polish
        time_left_pol = DEADLINE - 5 - (time.time() - start_time)
        if time_left_pol > 1.0:
            c_pol = _slsqp_polish(c_curr, time_limit=time_left_pol)
            r_pol = _radius(c_pol)
            
            if r_pol > best_r:
                best_r = r_pol
                best_centers = c_pol.copy()
                
    # Fallback
    if best_centers is None:
        best_centers = _make_hexagonal_grid(N)
        
    return best_centers

if __name__ == "__main__":
    t0 = time.time()
    res = construct()
    t1 = time.time()
    r = _radius(res)
    print(f"Time: {t1-t0:.2f}s, Radius: {r}")
