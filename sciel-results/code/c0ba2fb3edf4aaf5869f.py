"""
Pack 50 equal circles in a unit square using contact-graph enumeration and
constrained least-squares resolution of the isostatic system.

Strategy:
1. Generate diverse initial configurations (grid, hex, random).
2. For each configuration, compute the Delaunay triangulation to identify
   potential contact pairs.
3. Use a "shrink-wrap" heuristic to identify the active contact graph:
   iteratively remove pairs that are "too far" apart given a target radius,
   or use the current geometry to guess active contacts.
4. For a fixed contact graph (set of pairs that must be at distance 2r,
   and pairs against walls that must be at distance r), solve for the
   optimal coordinates and radius using a constrained least-squares approach.
   This exploits the local rigidity of the packing.
5. Use the solution as a starting point for a final SLSQP polish to handle
   any non-rigid modes or boundary effects.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr

N = 50
DEADLINE = 140.0  # Safety margin before 150s kill

def _true_radius(centers):
    """Compute the true max common radius for given centers."""
    wall_r = np.min(0.5 - np.abs(centers))
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    return min(wall_r, min_pair / 2.0)

def _make_grid_init(n_x, n_y):
    """Create a rectangular grid."""
    ticks_x = (np.arange(n_x) + 0.5) / n_x - 0.5
    ticks_y = (np.arange(n_y) + 0.5) / n_y - 0.5
    grid = np.array([(x, y) for y in ticks_y for x in ticks_x])
    return grid[:N].copy()

def _make_hexagonal_init():
    """Create a hexagonal lattice initialization."""
    spacing = 0.14
    dy = spacing * np.sqrt(3) / 2
    points = []
    for j in range(10):
        for i in range(10):
            x = i * spacing - 0.5 + spacing / 2
            y = j * dy - 0.5
            if j % 2 != 1:
                x += spacing / 2
            points.append((x, y))
    points = np.array(points)
    points = points[:N]
    points = points - np.mean(points, axis=0)
    max_ext = np.max(np.abs(points))
    if max_ext > 0.45:
        points = points * (0.45 / max_ext)
    return points

def _identify_active_contacts(centers, r_guess, tol=0.02):
    """
    Identify pairs of circles that are likely in contact.
    A pair (i, j) is active if dist(i, j) < 2*r_guess + tol.
    Also identify wall contacts.
    """
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    contact_pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            if dist[i, j] < 2 * r_guess + tol:
                contact_pairs.append((i, j))
                
    wall_contacts = []
    for i in range(N):
        # Check x boundaries
        if 0.5 - centers[i, 0] < r_guess + tol:
            wall_contacts.append((i, 'x+'))
        if 0.5 + centers[i, 0] < r_guess + tol:
            wall_contacts.append((i, 'x-'))
        # Check y boundaries
        if 0.5 - centers[i, 1] < r_guess + tol:
            wall_contacts.append((i, 'y+'))
        if 0.5 + centers[i, 1] < r_guess + tol:
            wall_contacts.append((i, 'y-'))
            
    return contact_pairs, wall_contacts

def _solve_contact_system(centers_init, contact_pairs, wall_contacts):
    """
    Solve for the optimal positions and radius given a fixed contact graph.
    
    Variables: x_i, y_i for i in 0..N-1, and r.
    Total variables: 2*N + 1.
    
    Constraints:
    For each contact pair (i, j):
        (x_i - x_j)^2 + (y_i - y_j)^2 = (2r)^2
    For each wall contact (i, side):
        e.g., x_i + r = 0.5  => x_i + r - 0.5 = 0
    
    We linearize these constraints around the initial guess and solve the
    least-squares problem.
    """
    centers = centers_init.copy()
    r_guess = _true_radius(centers)
    
    # Variables: [x_0, y_0, ..., x_{N-1}, y_{N-1}, r]
    x0 = np.zeros(2 * N + 1)
    for i in range(N):
        x0[2*i] = centers[i, 0]
        x0[2*i+1] = centers[i, 1]
    x0[-1] = r_guess
    
    # Build Jacobian J and residual vector b for linearized system J * dx = b
    # We want to minimize ||J * dx - b||^2
    # Where the constraints are g_k(x) = 0
    
    num_constraints = len(contact_pairs) + len(wall_contacts)
    if num_constraints == 0:
        return centers, r_guess
    
    J = np.zeros((num_constraints, 2 * N + 1))
    b = np.zeros(num_constraints)
    
    row = 0
    # Circle-circle contacts
    for i, j in contact_pairs:
        dx = centers[i, 0] - centers[j, 0]
        dy = centers[i, 1] - centers[j, 1]
        dist = np.sqrt(dx*dx + dy*dy)
        if dist < 1e-8:
            continue
            
        # Constraint: (x_i - x_j)^2 + (y_i - y_j)^2 - (2r)^2 = 0
        # Linearization:
        # 2(x_i - x_j) * dx_i - 2(x_i - x_j) * dx_j + 2(y_i - y_j) * dy_i - 2(y_i - y_j) * dy_j - 4r * dr = 0
        # Let g = dist^2 - (2r)^2
        # dg/dx_i = 2(dx), dg/dx_j = -2(dx), dg/dy_i = 2(dy), dg/dy_j = -2(dy), dg/dr = -4r
        
        # We want J * dx = -g (Newton step)
        g = dist*dist - (2*r_guess)**2
        
        J[row, 2*i] = 2 * dx
        J[row, 2*i+1] = 2 * dy
        J[row, 2*j] = -2 * dx
        J[row, 2*j+1] = -2 * dy
        J[row, -1] = -4 * r_guess
        
        b[row] = -g
        row += 1
        
    # Wall contacts
    for i, side in wall_contacts:
        # Constraint depends on side
        # x+ : x_i + r - 0.5 = 0
        # x- : -x_i + r + 0.5 = 0
        # y+ : y_i + r - 0.5 = 0
        # y- : -y_i + r + 0.5 = 0
        
        if side == 'x+':
            J[row, 2*i] = 1
            J[row, -1] = 1
            b[row] = -(centers[i, 0] + r_guess - 0.5)
        elif side == 'x-':
            J[row, 2*i] = -1
            J[row, -1] = 1
            b[row] = -(-centers[i, 0] + r_guess + 0.5)
        elif side == 'y+':
            J[row, 2*i+1] = 1
            J[row, -1] = 1
            b[row] = -(centers[i, 1] + r_guess - 0.5)
        elif side == 'y-':
            J[row, 2*i+1] = -1
            J[row, -1] = 1
            b[row] = -(-centers[i, 1] + r_guess + 0.5)
            
        row += 1

    # Solve J * dx = b
    try:
        dx, sres, rank, itn, istop, nrmr, conda, normu, normv = lsqr(J, b)
    except Exception:
        return centers, r_guess
        
    # Update variables
    x_new = x0 + dx
    centers_new = np.zeros((N, 2))
    for i in range(N):
        centers_new[i, 0] = x_new[2*i]
        centers_new[i, 1] = x_new[2*i+1]
    r_new = x_new[-1]
    
    # Clip to valid bounds
    centers_new = np.clip(centers_new, -0.5 + 1e-6, 0.5 - 1e-6)
    r_new = max(r_new, 1e-6)
    
    return centers_new, r_new

def _refine_with_contact_graph(centers, max_iter=20, time_limit=10.0):
    """
    Iteratively identify contact graph and solve the isostatic system.
    """
    start_time = time.time()
    centers = centers.copy()
    r_current = _true_radius(centers)
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        # Identify active contacts based on current radius
        # Use a tight tolerance to ensure we only pick up true contacts
        tol = 0.005 
        contact_pairs, wall_contacts = _identify_active_contacts(centers, r_current, tol=tol)
        
        # Solve for new positions
        centers_new, r_new = _solve_contact_system(centers, contact_pairs, wall_contacts)
        
        # Calculate new true radius
        r_true_new = _true_radius(centers_new)
        
        # Check for improvement
        if r_true_new > r_current + 1e-6:
            centers = centers_new
            r_current = r_true_new
        else:
            # If no improvement, try a slightly looser tolerance to find more contacts?
            # Or break if stuck.
            # For now, if we don't improve, we might be in a local min of the graph.
            # Try to perturb slightly? No, let's just break if no improvement after a few tries.
            # To avoid infinite loops on exact stagnation, add a small check.
            if np.allclose(centers, centers_new, atol=1e-8):
                break
            # If we moved but radius didn't improve, revert?
            # Actually, the LSQR solution minimizes the constraint violation.
            # If the graph is wrong, it might push us into a worse spot.
            # Let's accept the move if it doesn't violate constraints too badly?
            # Simpler: only accept if radius improves.
            # If radius doesn't improve, we might be stuck.
            # Let's try one more iteration with a slightly larger tolerance to see if we missed a contact.
            contact_pairs, wall_contacts = _identify_active_contacts(centers, r_current, tol=tol*2)
            centers_new2, r_new2 = _solve_contact_system(centers, contact_pairs, wall_contacts)
            r_true_new2 = _true_radius(centers_new2)
            if r_true_new2 > r_current + 1e-6:
                centers = centers_new2
                r_current = r_true_new2
            else:
                break
                
    return centers, r_current

def _slsqp_polish(centers, time_limit=20.0):
    """
    Final polish with SLSQP to handle non-rigid modes and ensure feasibility.
    """
    x0 = centers.ravel()
    start_time = time.time()
    
    def objective(x):
        c = x.reshape(N, 2)
        wall_dists = 0.5 - np.abs(c)
        min_wall = np.min(wall_dists)
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
        min_pair = np.min(dist)
        r = min(min_wall, min_pair / 2.0)
        return -r
        
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    def callback(xk):
        if time.time() - start_time > time_limit:
            raise RuntimeError("Time limit exceeded")
        return False
        
    try:
        res = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-10, 'disp': False},
            callback=callback
        )
        if res.x is not None:
            c_res = res.x.reshape(N, 2)
            # Ensure validity
            if np.all(np.isfinite(c_res)) and np.all(np.abs(c_res) <= 0.5):
                return c_res
    except Exception:
        pass
        
    return centers

def construct():
    global DEADLINE
    start_time = time.time()
    
    best_centers = None
    best_r = 0.0
    
    # Generate initial configurations
    inits = []
    
    # 1. Grid 7x8 (approx 50)
    inits.append(_make_grid_init(7, 8))
    # 2. Grid 8x7
    inits.append(_make_grid_init(8, 7))
    # 3. Hexagonal
    inits.append(_make_hexagonal_init())
    
    # 4. Random with jitter
    np.random.seed(42)
    for _ in range(3):
        c_rand = np.random.uniform(-0.45, 0.45, size=(N, 2))
        inits.append(c_rand)
        
    # Process each initial configuration
    for i, init in enumerate(inits):
        if time.time() - start_time > DEADLINE - 20:
            break
            
        time_left = DEADLINE - 20 - (time.time() - start_time)
        
        # Step 1: Initial relaxation to remove obvious overlaps
        # Simple push-apart
        centers = init.copy()
        for _ in range(50):
            if time.time() - start_time > DEADLINE - 20:
                break
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff ** 2, axis=2))
            np.fill_diagonal(dist, np.inf)
            min_dist = np.min(dist)
            if min_dist > 0.1: # Roughly 2*0.05
                break
            # Push apart
            for i in range(N):
                for j in range(i+1, N):
                    if dist[i, j] < 0.1:
                        d = centers[j] - centers[i]
                        norm = np.linalg.norm(d)
                        if norm > 1e-6:
                            d = d / norm
                            overlap = 0.1 - dist[i, j]
                            centers[i] -= 0.5 * overlap * d
                            centers[j] += 0.5 * overlap * d
            centers = np.clip(centers, -0.5, 0.5)
            
        # Step 2: Contact Graph Refinement
        refined, r_refined = _refine_with_contact_graph(centers, max_iter=30, time_limit=min(15.0, time_left * 0.6))
        
        # Step 3: SLSQP Polish
        final_centers = _slsqp_polish(refined, time_limit=min(10.0, time_left * 0.4))
        
        r_final = _true_radius(final_centers)
        
        if r_final > best_r:
            best_r = r_final
            best_centers = final_centers.copy()
            
    # If time remains, try more random restarts
    while time.time() - start_time < DEADLINE - 10:
        seed = int(time.time() * 1000) % 100000
        np.random.seed(seed)
        c_rand = np.random.uniform(-0.45, 0.45, size=(N, 2))
        
        time_left = DEADLINE - 10 - (time.time() - start_time)
        if time_left < 15:
            break
        
        # Quick relaxation
        centers = c_rand.copy()
        for _ in range(20):
            if time.time() - start_time > DEADLINE - 10:
                break
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff ** 2, axis=2))
            np.fill_diagonal(dist, np.inf)
            min_dist = np.min(dist)
            if min_dist > 0.1:
                break
            for i in range(N):
                for j in range(i+1, N):
                    if dist[i, j] < 0.1:
                        d = centers[j] - centers[i]
                        norm = np.linalg.norm(d)
                        if norm > 1e-6:
                            d = d / norm
                            overlap = 0.1 - dist[i, j]
                            centers[i] -= 0.5 * overlap * d
                            centers[j] += 0.5 * overlap * d
            centers = np.clip(centers, -0.5, 0.5)
            
        # Contact Graph Refinement
        refined, r_refined = _refine_with_contact_graph(centers, max_iter=20, time_limit=min(10.0, time_left * 0.5))
        
        # SLSQP Polish
        final_centers = _slsqp_polish(refined, time_limit=min(5.0, time_left * 0.3))
        
        r_final = _true_radius(final_centers)
        
        if r_final > best_r:
            best_r = r_final
            best_centers = final_centers.copy()
            
    if best_centers is None:
        # Fallback
        best_centers = _make_grid_init(7, 8)
        
    return best_centers
