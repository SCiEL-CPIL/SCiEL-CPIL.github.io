"""Pack 50 equal circles in a unit square, maximize common radius.

Strategy: Rigid-graph / contact-constraint optimization.
Instead of gradient-based search over all coordinates with a soft-min objective,
we first run a short gradient-based phase to get a decent configuration, then
identify the contact graph (which pairs of circles are touching and which circles
touch the walls), and finally solve the resulting system of nonlinear equations
(exact distance constraints for contacts + wall constraints) using scipy's least_squares.
This treats the packing as a mechanism design problem where the topology is fixed.
"""
import numpy as np
import time
from scipy.optimize import minimize, least_squares

N = 50
DEADLINE = 140.0  # seconds, safety margin before 150s kill

def _radius(centers):
    """Compute the max common radius for given centers."""
    wall_r = np.min(0.5 - np.abs(centers))
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    return min(wall_r, min_pair / 2.0)

def _soft_min_radius(centers, beta=50.0):
    """Smooth approximation of the minimum radius using log-sum-exp."""
    d_left = centers[:, 0] + 0.5
    d_right = 0.5 - centers[:, 0]
    d_bottom = centers[:, 1] + 0.5
    d_top = 0.5 - centers[:, 1]
    wall_dists = np.concatenate([d_left, d_right, d_bottom, d_top])
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    idx = np.triu_indices(N, 1)
    pair_dists = dist[idx] / 2.0
    
    all_dists = np.concatenate([wall_dists, pair_dists])
    u = -beta * all_dists
    max_u = np.max(u)
    sum_exp = np.sum(np.exp(u - max_u))
    log_sum = max_u + np.log(sum_exp)
    return -log_sum / beta

def _objective_smooth(x, beta=50.0):
    centers = x.reshape(N, 2)
    r = _soft_min_radius(centers, beta)
    return -r

def _make_grid(n_per_side, N):
    ticks = (np.arange(n_per_side) + 0.5) / n_per_side - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()

def _jitter(centers, scale):
    return centers + np.random.normal(0, scale, size=centers.shape)

def _local_refine_smooth(centers, beta=50.0, time_limit=5.0):
    x0 = centers.ravel()
    bounds = [(-0.5, 0.5)] * (2 * N)
    start_time = time.time()
    
    def callback(xk):
        if time.time() - start_time > time_limit:
            raise RuntimeError("Time limit exceeded")
        return False
    
    try:
        res = minimize(
            _objective_smooth, x0, args=(beta,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False}
        )
        if res.x is not None:
            return res.x.reshape(N, 2)
    except RuntimeError:
        pass
    except Exception:
        pass
    return centers

def _identify_contact_graph(centers, r, tol=0.001):
    """
    Identify which pairs of circles are in contact (distance ≈ 2r)
    and which circles touch the walls (distance to wall ≈ r).
    Returns:
        pair_contacts: list of (i, j) indices
        wall_contacts: list of (i, wall_idx) where wall_idx is 0=left,1=right,2=bottom,3=top
    """
    r = _radius(centers)
    # Use a slightly tighter tolerance for identifying "active" constraints
    # We want constraints that are nearly binding.
    contact_tol = max(0.001, r * 0.01)
    
    pair_contacts = []
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    # Find pairs where distance is close to 2r
    for i in range(N):
        for j in range(i+1, N):
            if dist[i, j] <= 2 * r + contact_tol:
                pair_contacts.append((i, j))
    
    wall_contacts = []
    for i in range(N):
        x, y = centers[i]
        if x + 0.5 <= r + contact_tol:
            wall_contacts.append((i, 0))  # left
        if 0.5 - x <= r + contact_tol:
            wall_contacts.append((i, 1))  # right
        if y + 0.5 <= r + contact_tol:
            wall_contacts.append((i, 2))  # bottom
        if 0.5 - y <= r + contact_tol:
            wall_contacts.append((i, 3))  # top
    
    return pair_contacts, wall_contacts, r

def _solve_contact_system(centers0, pair_contacts, wall_contacts, r_target, max_iter=1000, time_limit=5.0):
    """
    Solve the system of equations where:
    - For each pair (i,j) in pair_contacts: ||c_i - c_j|| = 2 * r_target
    - For each (i, wall) in wall_contacts: distance from c_i to wall = r_target
    
    We use least_squares to minimize the residual of these constraints.
    The unknowns are the 2*N coordinates.
    """
    start_time = time.time()
    
    def residuals(x):
        centers = x.reshape(N, 2)
        res = []
        
        # Pair constraints: ||c_i - c_j|| - 2*r_target = 0
        for i, j in pair_contacts:
            d = np.linalg.norm(centers[i] - centers[j]) - 2 * r_target
            res.append(d)
        
        # Wall constraints: distance to wall - r_target = 0
        for i, wall in wall_contacts:
            x_c, y_c = centers[i]
            if wall == 0:  # left: x + 0.5 - r = 0
                res.append(x_c + 0.5 - r_target)
            elif wall == 1:  # right: 0.5 - x - r = 0
                res.append(0.5 - x_c - r_target)
            elif wall == 2:  # bottom: y + 0.5 - r = 0
                res.append(y_c + 0.5 - r_target)
            elif wall == 3:  # top: 0.5 - y - r = 0
                res.append(0.5 - y_c - r_target)
        
        return np.array(res)
    
    x0 = centers0.ravel()
    
    def callback(xk, res, cost, g):
        if time.time() - start_time > time_limit:
            raise RuntimeError("Time limit exceeded")
    
    try:
        sol = least_squares(
            residuals, x0,
            method='trf',  # Trust Region Reflective
            bounds=([-0.5] * (2*N), [0.5] * (2*N)),
            max_nfev=max_iter,
            xtol=1e-10, ftol=1e-10, gtol=1e-10
        )
        if sol.success or sol.x is not None:
            return sol.x.reshape(N, 2)
    except RuntimeError:
        pass
    except Exception:
        pass
    
    return centers0

def _refine_with_contact_graph(centers, time_limit=10.0):
    """
    Refine centers by identifying the contact graph and solving the constraint system.
    """
    start_time = time.time()
    r = _radius(centers)
    
    # Identify contacts with a reasonable tolerance
    pair_contacts, wall_contacts, r_active = _identify_contact_graph(centers, r, tol=0.005)
    
    # If we have too few or too many contacts, the graph might be wrong.
    # A typical 50-circle packing has about 2-3 contacts per circle on average, so ~50-75 pair contacts.
    # Let's just proceed.
    
    if len(pair_contacts) == 0 and len(wall_contacts) == 0:
        return centers
    
    # Solve the system
    time_left = time_limit - (time.time() - start_time)
    if time_left < 1.0:
        return centers
    
    refined = _solve_contact_system(centers, pair_contacts, wall_contacts, r_active, 
                                     max_iter=500, time_limit=time_left * 0.8)
    
    return refined

def construct():
    global DEADLINE
    start_time = time.time()
    
    best_centers = None
    best_r = 0.0
    
    # Generate initial configurations
    inits = []
    
    # 1. 7x7 Grid (49 pts) + 1 center, jittered
    c1 = _make_grid(7, 49)
    c1 = np.vstack([c1, np.array([0.0, 0.0])])
    c1 = _jitter(c1, 0.03)
    inits.append(c1)
    
    # 2. 8x8 Grid (64 pts) -> take 50, jittered
    c2 = _make_grid(8, 50)
    c2 = _jitter(c2, 0.03)
    inits.append(c2)
    
    # 3. Random points
    np.random.seed(42)
    c3 = np.random.uniform(-0.45, 0.45, size=(N, 2))
    inits.append(c3)
    
    # 4. Another random seed
    np.random.seed(123)
    c4 = np.random.uniform(-0.45, 0.45, size=(N, 2))
    inits.append(c4)
    
    # 5. 5x10 Grid
    ticks_x = (np.arange(5) + 0.5) / 5 - 0.5
    ticks_y = (np.arange(10) + 0.5) / 10 - 0.5
    grid_5x10 = np.array([(x, y) for y in ticks_y for x in ticks_x])
    c5 = _jitter(grid_5x10, 0.03)
    inits.append(c5)
    
    # 6. 10x5 Grid
    ticks_x = (np.arange(10) + 0.5) / 10 - 0.5
    ticks_y = (np.arange(5) + 0.5) / 5 - 0.5
    grid_10x5 = np.array([(x, y) for y in ticks_y for x in ticks_x])
    c6 = _jitter(grid_10x5, 0.03)
    inits.append(c6)
    
    # Refine each initial configuration
    for i, init in enumerate(inits):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        time_left = DEADLINE - 10 - (time.time() - start_time)
        alloc_time = min(15.0, time_left / max(1, len(inits) - i))
        
        # Stage 1: Gradient-based refinement to get a good starting point
        refined1 = _local_refine_smooth(init, beta=50.0, time_limit=alloc_time * 0.4)
        r1 = _radius(refined1)
        
        if r1 > best_r:
            best_r = r1
            best_centers = refined1.copy()
        
        # Stage 2: Contact graph refinement
        time_left2 = alloc_time - (time.time() - start_time)
        if time_left2 > 2.0:
            refined2 = _refine_with_contact_graph(refined1, time_limit=time_left2 * 0.8)
            r2 = _radius(refined2)
            
            if r2 > best_r:
                best_r = r2
                best_centers = refined2.copy()
        
        # Stage 3: Another gradient pass from the contact-refined solution
        time_left3 = alloc_time - (time.time() - start_time)
        if time_left3 > 2.0:
            refined3 = _local_refine_smooth(refined2, beta=100.0, time_limit=time_left3 * 0.8)
            r3 = _radius(refined3)
            
            if r3 > best_r:
                best_r = r3
                best_centers = refined3.copy()
    
    # If we have time left, do more random restarts with contact graph refinement
    while time.time() - start_time < DEADLINE - 5:
        np.random.seed(int(time.time() * 1000) % 10000)
        c_rand = np.random.uniform(-0.45, 0.45, size=(N, 2))
        
        time_left = DEADLINE - 5 - (time.time() - start_time)
        if time_left < 3: break
        
        # Quick gradient refine
        refined = _local_refine_smooth(c_rand, beta=50.0, time_limit=time_left * 0.3)
        r = _radius(refined)
        
        # Contact graph refine
        time_left2 = time_left - (time.time() - start_time)
        if time_left2 > 2:
            refined2 = _refine_with_contact_graph(refined, time_limit=time_left2 * 0.7)
            r2 = _radius(refined2)
            r = max(r, r2)
            if r2 > r:
                refined = refined2
        
        if r > best_r:
            best_r = r
            best_centers = refined.copy()
            
    return best_centers
