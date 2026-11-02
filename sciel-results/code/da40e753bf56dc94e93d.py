import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform

N = 200

def _compute_radius(c):
    """Compute the largest common radius for given centres."""
    n = c.shape[0]
    if n == 0:
        return 0.0
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    if n < 2:
        return min_boundary
    dists = pdist(c, metric='euclidean')
    min_pair_dist = np.min(dists)
    return min(min_pair_dist / 2.0, min_boundary)

def _initialize_hexagonal():
    """
    Initialize with a hexagonal pattern.
    We try a range of spacings to find one that fits N points and maximizes initial radius.
    """
    best_c = None
    best_r = 0.0
    
    # Range of spacings d such that we can fit ~200 points in [-0.5, 0.5]^2
    # Area per point ~ d^2 * sqrt(3)/2. Total area 1.
    # d^2 ~ 2 / (200 * sqrt(3)) ~ 0.0057 -> d ~ 0.075
    # We search around this value.
    for d in np.linspace(0.065, 0.085, 40):
        points = []
        row_spacing = d * np.sqrt(3) / 2
        x_offset = d / 2
        
        y = -0.5 + d / 2
        row = 0
        while y <= 0.5 - d / 2 + 1e-9:
            x = -0.5 + d / 2
            if row % 2 == 1:
                x += x_offset
            while x <= 0.5 - d / 2 + 1e-9:
                # Check if point is inside
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    points.append([x, y])
                x += d
            y += row_spacing
            row += 1
        
        if len(points) >= N:
            # Take the first N points? Or the ones closest to center?
            # Taking first N might bias to one corner. 
            # Better: sort by distance from center and take N closest?
            # Or just take a subset that looks uniform.
            # For now, take first N, but ensure we don't cut off a partial row awkwardly if possible.
            # Actually, just taking the first N is fine if we iterate d.
            c = np.array(points[:N])
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
    
    if best_c is None:
        # Fallback grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_c = grid[:N].copy()
        best_r = _compute_radius(best_c)
        
    return best_c, best_r

def _dpo_step(c, r, max_iter=100, tol=1e-10):
    """
    Direct Physical Optimization step.
    Tries to increase radius by solving a linear system for displacements.
    """
    n = c.shape[0]
    c = c.copy()
    current_r = r
    
    for _ in range(max_iter):
        # 1. Identify active constraints
        # Pairs with distance <= 2*r + margin
        # Boundaries with distance <= r + margin
        margin = 1e-5
        tree = cKDTree(c)
        
        # Find close pairs
        # Using query_pairs is efficient
        pairs = list(tree.query_pairs(2 * current_r + margin))
        
        active_pairs = []
        for i, j in pairs:
            d = np.linalg.norm(c[i] - c[j])
            if d < 2 * current_r + margin:
                active_pairs.append((i, j))
                
        # Find boundary contacts
        # Left: x + r <= 0.5 -> x <= 0.5 - r
        # Right: -x + r <= 0.5 -> x >= -0.5 + r
        # Top: y + r <= 0.5 -> y <= 0.5 - r
        # Bottom: -y + r <= 0.5 -> y >= -0.5 + r
        
        active_bounds = []
        for i in range(n):
            # Left
            if c[i, 0] < 0.5 - current_r + margin:
                # Constraint: c[i,0] >= -0.5 + r  (Wait, boundary is -0.5)
                # Distance to left wall: c[i,0] - (-0.5) = c[i,0] + 0.5
                # We want c[i,0] + 0.5 >= r
                # Normal vector pointing inward: (1, 0)
                # If c[i,0] + 0.5 < r + margin, it's active
                if c[i, 0] + 0.5 < current_r + margin:
                    active_bounds.append((i, 0, 1.0)) # idx, coord, sign (inward)
            # Right
            if 0.5 - c[i, 0] < current_r + margin:
                active_bounds.append((i, 0, -1.0))
            # Bottom
            if c[i, 1] + 0.5 < current_r + margin:
                active_bounds.append((i, 1, 1.0))
            # Top
            if 0.5 - c[i, 1] < current_r + margin:
                active_bounds.append((i, 1, -1.0))

        if not active_pairs and not active_bounds:
            break
            
        # 2. Build system A * delta = b
        # We want to increase radius by dr.
        # For a pair (i,j) at distance d_ij:
        # New dist should be d_ij + 2*dr
        # (c_i + d_i - c_j - d_j) . u_ij = 2*dr
        # u_ij is unit vector from j to i.
        
        # For a boundary (i, coord, sign):
        # dist = sign * c_i[coord] + 0.5
        # New dist = dist + dr
        # sign * d_i[coord] = dr
        
        # We solve min ||delta||^2 s.t. A delta = b
        # Using least squares: delta = A^T (A A^T)^{-1} b  (if A A^T invertible)
        # Or simply use np.linalg.lstsq
        
        n_vars = 2 * n
        # Number of constraints
        n_cons = len(active_pairs) + len(active_bounds)
        
        if n_cons == 0:
            break
            
        A = np.zeros((n_cons, n_vars))
        b = np.zeros(n_cons)
        
        idx = 0
        # Pairs
        for i, j in active_pairs:
            vec = c[i] - c[j]
            d = np.linalg.norm(vec)
            if d < 1e-12:
                u = np.array([1.0, 0.0])
            else:
                u = vec / d
            # Equation: u . (d_i - d_j) = 2 * dr
            # We set dr = 1.0 for now, then scale the result.
            # Actually, we can just solve for delta that increases dist by 2*dr_target.
            # Let's aim for a small increase, e.g., 0.01 * current_r or just 1.0 and normalize.
            # Standard DPO: aim for increase of 1.0 in the "gap" variable?
            # Let's aim for increase of dr = 0.01.
            target_dr = 0.01
            b[idx] = 2 * target_dr
            A[idx, 2*i:2*i+2] = u
            A[idx, 2*j:2*j+2] = -u
            idx += 1
            
        # Bounds
        for i, coord, sign in active_bounds:
            # Equation: sign * d_i[coord] = dr
            b[idx] = target_dr
            col = 2*i + coord
            A[idx, col] = sign
            idx += 1

        # Solve least squares
        # delta = A^T (A A^T)^+ b
        # Using np.linalg.lstsq is robust
        try:
            delta, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            break
            
        # Apply delta
        # Clip to stay in bounds
        new_c = c + delta.reshape(-1, 2)
        new_c = np.clip(new_c, -0.5 + 1e-12, 0.5 - 1e-12)
        
        # Calculate new radius
        new_r = _compute_radius(new_c)
        
        if new_r > current_r + 1e-12:
            c = new_c
            current_r = new_r
        else:
            # If it didn't improve, try a smaller step
            # This can happen if the linear approximation is bad or constraints are conflicting
            # Break to avoid oscillation
            break
            
    return c, current_r

def _slsqp_polish(c0, r0, max_time=10.0):
    """
    Final polish using SLSQP to handle the non-linear constraints precisely.
    """
    start_time = time.time()
    best_c = c0.copy()
    best_r = r0
    n = N

    def _get_active_pairs(c, r_target, margin=0.002):
        tree = cKDTree(c)
        pairs = list(tree.query_pairs(2 * r_target + margin))
        return np.array(pairs, dtype=np.int32) if pairs else np.empty((0, 2), dtype=np.int32)

    # Increase r slightly and try to satisfy constraints
    # We will iterate a few times, increasing r by a small amount each time
    # and solving the feasibility problem.
    
    step = 1e-5
    for _ in range(50):
        if time.time() > start_time + max_time:
            break
            
        r_target = best_r + step
        pairs = _get_active_pairs(best_c, r_target, margin=0.005)
        npairs = len(pairs)

        def constraints_func(x):
            c = x.reshape(-1, 2)
            bd = 0.5 - np.abs(c) - r_target
            bd_flat = bd.ravel()
            if npairs == 0:
                return bd_flat
            ci = c[pairs[:, 0]]
            cj = c[pairs[:, 1]]
            diffs = ci - cj
            dists = np.sqrt(np.sum(diffs**2, axis=1))
            pd_vals = dists - 2 * r_target
            return np.concatenate([bd_flat, pd_vals])

        def jac_func(x):
            c = x.reshape(-1, 2)
            total_constraints = 2 * n + npairs
            jac = np.zeros((total_constraints, 2 * n))
            signs = -np.sign(c)
            for i in range(n):
                jac[2*i, 2*i] = signs[i, 0]
                jac[2*i+1, 2*i+1] = signs[i, 1]
            if npairs == 0:
                return jac
            ci = c[pairs[:, 0]]
            cj = c[pairs[:, 1]]
            diffs = ci - cj
            dists = np.sqrt(np.sum(diffs**2, axis=1))
            dists_safe = np.where(dists < 1e-12, 1e-12, dists)
            unit_vecs = diffs / dists_safe[:, np.newaxis]
            row_indices = 2 * n + np.arange(npairs)
            col_i = pairs[:, 0] * 2
            col_j = pairs[:, 1] * 2
            np.add.at(jac, (row_indices, col_i), unit_vecs[:, 0])
            np.add.at(jac, (row_indices, col_i + 1), unit_vecs[:, 1])
            np.add.at(jac, (row_indices, col_j), -unit_vecs[:, 0])
            np.add.at(jac, (row_indices, col_j + 1), -unit_vecs[:, 1])
            return jac

        try:
            result = minimize(
                lambda x: 0.0,
                best_c.flatten(),
                method='SLSQP',
                bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * n),
                constraints=[
                    {'type': 'ineq', 'fun': constraints_func, 'jac': jac_func}
                ],
                options={'maxiter': 50, 'ftol': 1e-12, 'disp': False}
            )
            c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)
            if r_new > best_r + 1e-14:
                best_c = c_new
                best_r = r_new
                step = min(step * 1.1, 5e-5)
            else:
                step *= 0.5
                if step < 1e-8:
                    break
        except Exception:
            break
            
    return best_c, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin
    
    # 1. Initialize
    c, r = _initialize_hexagonal()
    
    # 2. DPO Optimization
    # Run DPO for most of the time
    # DPO is fast per step but requires many steps
    # We will run it in a loop until time is up or no improvement
    
    last_improvement_time = start_time
    best_r = r
    best_c = c.copy()
    
    # Main DPO loop
    while time.time() < deadline - 15.0:
        c, r = _dpo_step(c, r, max_iter=50)
        if r > best_r + 1e-12:
            best_r = r
            best_c = c.copy()
            last_improvement_time = time.time()
        else:
            # If no improvement, we might be stuck.
            # Add small noise to escape?
            # Or break if stuck for too long.
            if time.time() - last_improvement_time > 10.0:
                # Perturb
                c += np.random.normal(0, 1e-4, c.shape)
                c = np.clip(c, -0.5 + 1e-12, 0.5 - 1e-12)
                r = _compute_radius(c)
                last_improvement_time = time.time()
                
    # 3. Final Polish
    if time.time() < deadline - 5.0:
        c, r = _slsqp_polish(best_c, best_r, max_time=10.0)
        if r > best_r:
            best_r = r
            best_c = c.copy()
            
    return best_c

# Keep dummy functions if they were part of the interface, though construct is the main one
# The prompt says "Keep the function names and signatures of the seed program".
# The seed program had: construct()
# It also had helper functions. I will keep the helpers I used and remove unused ones or keep them if they don't hurt.
# Actually, the prompt says "Keep the program's entry-point function name(s) and their return interface EXACTLY".
# It implies I can rewrite the internals. I will just provide the complete code.

if __name__ == "__main__":
    # Simple test
    c = construct()
    r = _compute_radius(c)
    print(f"Radius: {r}")
