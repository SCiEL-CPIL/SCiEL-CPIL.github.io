"""
Circle packing in a unit square: maximize common radius for 50 equal circles.
Strategy: Physics-Based Force-Directed Relaxation (Molecular Dynamics) + Sparse SLSQP Polish.
1. Initialize with a hexagonal grid.
2. Use Velocity Verlet integration with repulsive forces between circles and walls.
   - Forces are based on a soft penalty for overlap (distance < 2r) and wall violation.
   - Temperature (random noise) is annealed over time to escape local minima.
3. Once the system settles (low kinetic energy), extract the best radius found.
4. Run a final SLSQP polish on the active constraints (Delaunay neighbors + walls) 
   to refine the radius to machine precision.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 50
DEADLINE = 115.0  # Safety margin before 150s kill

def _score(pos):
    """Compute the maximum valid radius for given positions."""
    pos = np.asarray(pos, dtype=np.float64)
    if pos.shape != (N, 2):
        return -1.0
    if not np.all(np.isfinite(pos)):
        return -1.0
    # Wall constraint
    wall = 0.5 - np.abs(pos).max(axis=1)
    r_wall = wall.min()
    if r_wall < 0:
        return -1.0
    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    np.fill_diagonal(dist, np.inf)
    r_pair = (dist.min() / 2.0)
    return min(r_wall, r_pair)

def _init_hex():
    """
    Generate a hexagonal lattice initialization.
    """
    positions = []
    # Target ~50 circles. Hex packing is dense.
    # Let's generate a grid that covers the area and pick the best 50 or scale.
    # Spacing d. Area per circle ~ d^2 * sqrt(3)/2.
    # 50 circles in area 1 -> area/circle ~ 0.02.
    # d^2 * 0.866 ~ 0.02 -> d^2 ~ 0.023 -> d ~ 0.15.
    spacing = 0.15
    row_spacing = spacing * np.sqrt(3) / 2
    
    # Generate a large enough grid
    rows = 10
    cols = 10
    for row in range(rows):
        for col in range(cols):
            x = col * spacing + (0.5 * spacing if row % 2 else 0)
            y = row * row_spacing
            positions.append([x, y])
            
    positions = np.array(positions[:N])
    
    # Center and scale
    min_x, min_y = positions.min(axis=0)
    max_x, max_y = positions.max(axis=0)
    
    # Shift to center
    positions[:, 0] -= (min_x + max_x) / 2
    positions[:, 1] -= (min_y + max_y) / 2
    
    # Scale to fit within [-0.45, 0.45] to leave room
    max_abs = np.abs(positions).max()
    if max_abs > 0:
        positions *= (0.45 / max_abs)
        
    return positions

def _physics_relax(pos0, t0, deadline):
    """
    Physics-based relaxation using Velocity Verlet.
    """
    pos = pos0.copy()
    vel = np.zeros_like(pos)
    force = np.zeros_like(pos)
    
    # Parameters
    dt = 0.005
    k_rep = 10.0  # Repulsion strength
    k_wall = 50.0 # Wall strength
    damping = 0.99
    temp_start = 0.05
    temp_end = 0.001
    max_steps = 5000
    
    # Estimate initial radius
    r_est = _score(pos)
    if r_est <= 0:
        r_est = 0.05
    
    best_pos = pos.copy()
    best_r = r_est
    
    # Precompute pair indices for efficiency? No, N=50 is small enough for O(N^2) per step if vectorized.
    # But let's use a simple loop or vectorized pairwise diff.
    
    for step in range(max_steps):
        if time.time() - t0 > deadline - 5:
            break
        
        # Annealing temperature
        progress = step / max_steps
        temp = temp_start * (1 - progress) + temp_end * progress
        
        # 1. Compute Forces
        # Pairwise repulsion: F = k_rep * (2r - d) * (d_vec / d) if d < 2r
        # We want to maximize r, so we treat current r_est as the target distance 2r_est?
        # Actually, in packing, we usually fix r and push apart, or fix positions and grow r.
        # Here, let's use a soft constraint: minimize sum of max(0, 2r_est - d)^2.
        # This effectively pushes circles apart to increase the minimum distance.
        
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=-1))
        np.fill_diagonal(dist, 1e-9)
        
        # Overlap amount
        target_dist = 2 * r_est
        overlap = np.maximum(0, target_dist - dist)
        
        # Force magnitude
        # F_ij = k_rep * overlap * (diff_ij / dist_ij)
        # Only where overlap > 0
        force_mag = k_rep * overlap
        # Normalize direction
        dir_vec = diff / dist[:, :, np.newaxis]
        
        # Sum forces
        force = np.sum(force_mag[:, :, np.newaxis] * dir_vec, axis=1)
        
        # Wall forces
        # Push away from walls if too close
        # x < -0.5 + r_est -> push right
        # x > 0.5 - r_est -> push left
        wall_margin = r_est
        
        # x-direction
        force_x = np.zeros(N)
        # Left wall
        dist_left = pos[:, 0] - (-0.5 + wall_margin)
        force_x -= k_wall * np.maximum(0, -dist_left) # If dist_left < 0, force is positive (right)
        # Wait, if pos is at -0.49 and margin is 0.07, limit is -0.43.
        # dist_left = -0.49 - (-0.43) = -0.06.
        # We want force to +x.
        # force_x += k_wall * max(0, -dist_left)
        force_x += k_wall * np.maximum(0, -dist_left)
        
        # Right wall
        dist_right = (0.5 - wall_margin) - pos[:, 0]
        force_x -= k_wall * np.maximum(0, -dist_right)
        
        # y-direction
        force_y = np.zeros(N)
        # Bottom wall
        dist_bottom = pos[:, 1] - (-0.5 + wall_margin)
        force_y += k_wall * np.maximum(0, -dist_bottom)
        
        # Top wall
        dist_top = (0.5 - wall_margin) - pos[:, 1]
        force_y -= k_wall * np.maximum(0, -dist_top)
        
        force[:, 0] += force_x
        force[:, 1] += force_y
        
        # 2. Velocity Verlet Integration
        # v(t+dt/2) = v(t) + 0.5 * dt * F(t)
        vel += 0.5 * dt * force
        # Apply damping and temperature
        vel *= damping
        vel += temp * np.random.randn(*pos.shape)
        
        # x(t+dt) = x(t) + dt * v(t+dt/2)
        pos += dt * vel
        
        # v(t+dt) = v(t+dt/2) + 0.5 * dt * F(t+dt)
        # Recompute force for second half step? 
        # For simplicity and speed, we can use the same force or recompute.
        # Recomputing is more accurate. Let's recompute force quickly.
        # Actually, for N=50, O(N^2) is ~2500 ops. 5000 steps -> 12.5M ops. Fast in numpy.
        
        # Recompute force
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=-1))
        np.fill_diagonal(dist, 1e-9)
        overlap = np.maximum(0, target_dist - dist)
        force_mag = k_rep * overlap
        dir_vec = diff / dist[:, :, np.newaxis]
        force_new = np.sum(force_mag[:, :, np.newaxis] * dir_vec, axis=1)
        
        # Wall forces again
        force_x = np.zeros(N)
        dist_left = pos[:, 0] - (-0.5 + wall_margin)
        force_x += k_wall * np.maximum(0, -dist_left)
        dist_right = (0.5 - wall_margin) - pos[:, 0]
        force_x -= k_wall * np.maximum(0, -dist_right)
        
        force_y = np.zeros(N)
        dist_bottom = pos[:, 1] - (-0.5 + wall_margin)
        force_y += k_wall * np.maximum(0, -dist_bottom)
        dist_top = (0.5 - wall_margin) - pos[:, 1]
        force_y -= k_wall * np.maximum(0, -dist_top)
        
        force_new[:, 0] += force_x
        force_new[:, 1] += force_y
        
        vel += 0.5 * dt * force_new
        
        # Update best radius
        # We can slightly increase r_est if the system is stable?
        # Or just check the actual score periodically.
        if step % 10 == 0:
            r_current = _score(pos)
            if r_current > best_r:
                best_r = r_current
                best_pos = pos.copy()
                # Optionally increase target distance slightly to encourage expansion
                # r_est = r_current * 1.01
                # But this might destabilize. Let's stick to fixed target for stability,
                # but the forces will naturally push them apart if they can.
                # Actually, if we fix target_dist = 2 * r_est, and r_est is initial,
                # the system will just try to achieve that distance.
                # To maximize r, we should slowly increase r_est.
                if r_current > 0:
                    r_est = r_current * 1.005
    
    return best_pos, best_r

def _slsqp_polish_sparse(centers, r, t0, deadline, maxiter=200):
    """
    SLSQP polish using only active constraints (Delaunay neighbors + walls).
    """
    # Compute Delaunay triangulation
    try:
        tri = Delaunay(centers)
        edges = set()
        for simplex in tri.simplices:
            edges.add(tuple(sorted((simplex[0], simplex[1]))))
            edges.add(tuple(sorted((simplex[1], simplex[2]))))
            edges.add(tuple(sorted((simplex[0], simplex[2]))))
        # Convert to list of pairs
        active_pairs = [list(e) for e in edges]
    except Exception:
        # Fallback to all pairs if Delaunay fails
        active_pairs = [[i, j] for i in range(N) for j in range(i+1, N)]
    
    n_vars = 2 * N + 1
    r_idx = 2 * N
    
    def objective(x):
        return -x[r_idx]

    def objective_grad(x):
        g = np.zeros(n_vars)
        g[r_idx] = -1.0
        return g

    # Wall constraints (all circles, all 4 walls)
    def wall_constraints(x):
        c = x[:2*N].reshape(N, 2)
        r = x[r_idx]
        cons = np.empty(4 * N)
        for i in range(N):
            cons[4*i]   = 0.5 + r - c[i, 0]
            cons[4*i+1] = -0.5 + r + c[i, 0]
            cons[4*i+2] = 0.5 + r - c[i, 1]
            cons[4*i+3] = -0.5 + r + c[i, 1]
        return cons

    def wall_constraints_grad(x):
        J = np.zeros((4 * N, n_vars))
        for i in range(N):
            J[4*i, 2*i] = -1.0
            J[4*i, r_idx] = 1.0
            
            J[4*i+1, 2*i] = 1.0
            J[4*i+1, r_idx] = 1.0
            
            J[4*i+2, 2*i+1] = -1.0
            J[4*i+2, r_idx] = 1.0
            
            J[4*i+3, 2*i+1] = 1.0
            J[4*i+3, r_idx] = 1.0
        return J

    # Pairwise constraints (only active pairs)
    n_active_pairs = len(active_pairs)
    
    def pair_constraints(x):
        c = x[:2*N].reshape(N, 2)
        r = x[r_idx]
        cons = np.empty(n_active_pairs)
        for idx, (i, j) in enumerate(active_pairs):
            diff = c[i] - c[j]
            cons[idx] = np.dot(diff, diff) - 4 * r * r
        return cons

    def pair_constraints_grad(x):
        c = x[:2*N].reshape(N, 2)
        r = x[r_idx]
        J = np.zeros((n_active_pairs, n_vars))
        for idx, (i, j) in enumerate(active_pairs):
            diff = c[i] - c[j]
            J[idx, 2*i] = 2 * diff[0]
            J[idx, 2*i+1] = 2 * diff[1]
            J[idx, 2*j] = -2 * diff[0]
            J[idx, 2*j+1] = -2 * diff[1]
            J[idx, r_idx] = -8 * r
        return J

    constraints = [
        {'type': 'ineq', 'fun': wall_constraints, 'jac': wall_constraints_grad},
        {'type': 'ineq', 'fun': pair_constraints, 'jac': pair_constraints_grad}
    ]
    
    bounds = [(-0.5, 0.5)] * (2 * N) + [(0, 0.5)]
    
    x0 = np.zeros(n_vars)
    x0[:2*N] = centers.flatten()
    x0[r_idx] = r
    
    try:
        result = minimize(
            objective, 
            x0, 
            method='SLSQP', 
            jac=objective_grad, 
            bounds=bounds, 
            constraints=constraints,
            options={'maxiter': maxiter, 'ftol': 1e-12, 'disp': False}
        )
        if result.success:
            c_res = result.x[:2*N].reshape(N, 2)
            r_res = result.x[r_idx]
            r_check = _score(c_res)
            if r_check >= 0:
                return c_res, r_check
    except Exception:
        pass
    
    return centers, r

def construct():
    t0 = time.time()
    rng = np.random.default_rng(42)
    
    # 1. Initialize
    c_hex = _init_hex()
    r_hex = _score(c_hex)
    
    best_c = c_hex.copy()
    best_r = r_hex
    
    # 2. Physics Relaxation
    # Run multiple rounds with different initial temperatures or restarts
    num_physics_rounds = 3
    
    for i in range(num_physics_rounds):
        remaining = DEADLINE - (time.time() - t0)
        if remaining < 20:
            break
            
        # Perturb initial position slightly for diversity
        if i > 0:
            c_start = best_c + rng.normal(0, 0.01, best_c.shape)
            c_start = np.clip(c_start, -0.49, 0.49)
        else:
            c_start = best_c
            
        c_new, r_new = _physics_relax(c_start, t0, DEADLINE - 15)
        
        if r_new > best_r:
            best_c = c_new
            best_r = r_new
    
    # 3. SLSQP Polish
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 10:
        c_new, r_new = _slsqp_polish_sparse(best_c, best_r, t0, DEADLINE - 5, maxiter=200)
        if r_new > best_r:
            best_c = c_new
            best_r = r_new
            
    # 4. Final Check
    if not np.all(np.isfinite(best_c)) or best_c.shape != (N, 2):
        best_c = c_hex
        best_r = r_hex
        
    return best_c
