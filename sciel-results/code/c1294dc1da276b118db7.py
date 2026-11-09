"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 115s safety margin before 150s kill
    
    best_result = None
    best_score = -1.0
    
    def is_valid(positions, radii, w, h, tol=1e-9):
        for i in range(N):
            if positions[i, 0] - radii[i] < -tol or positions[i, 0] + radii[i] > w + tol:
                return False
            if positions[i, 1] - radii[i] < -tol or positions[i, 1] + radii[i] > h + tol:
                return False
        for i in range(N):
            for j in range(i + 1, N):
                dx = positions[j, 0] - positions[i, 0]
                dy = positions[j, 1] - positions[i, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < radii[i] + radii[j] - tol:
                    return False
        return True
    
    def compute_score(positions, radii, w, h):
        if not is_valid(positions, radii, w, h):
            return -1.0
        return np.sum(radii)
    
    def random_initialization(w, h, n_circles=N):
        """Generate random non-overlapping circle positions with small radii."""
        positions = np.zeros((n_circles, 2))
        radii = np.zeros(n_circles)
        
        # Place circles one by one
        for i in range(n_circles):
            attempts = 0
            while attempts < 2000:
                r = 0.01 + np.random.random() * 0.02
                x = r + np.random.random() * (w - 2*r)
                y = r + np.random.random() * (h - 2*r)
                
                valid = True
                for j in range(i):
                    dx = x - positions[j, 0]
                    dy = y - positions[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < r + radii[j] + 0.01:
                        valid = False
                        break
                
                if valid:
                    positions[i] = [x, y]
                    radii[i] = r
                    break
                attempts += 1
            else:
                # Fallback: place in a grid-like position
                positions[i] = [0.05 + (i % 5) * 0.15, 0.05 + (i // 5) * 0.15]
                radii[i] = 0.01
        
        return positions, radii
    
    def grid_positions(w, h, cols, rows):
        """Generate grid positions."""
        positions = []
        dx = w / cols
        dy = h / rows
        for row in range(rows):
            for col in range(cols):
                x = (col + 0.5) * dx
                y = (row + 0.5) * dy
                positions.append([x, y])
        return np.array(positions[:N])
    
    def hexagonal_positions(w, h, cols, rows):
        """Generate hexagonal packing positions."""
        positions = []
        dx = w / cols
        dy = h / rows
        for row in range(rows):
            for col in range(cols):
                x = (col + 0.5) * dx
                y = (row + 0.5) * dy
                if row % 2 == 1:
                    x += dx * 0.25
                positions.append([x, y])
        return np.array(positions[:N])
    
    def initial_radii(positions, w, h):
        """Compute initial radii based on local spacing and boundaries."""
        radii = np.zeros(N)
        for i in range(N):
            r_max = min(positions[i, 0], w - positions[i, 0],
                        positions[i, 1], h - positions[i, 1])
            for j in range(N):
                if i != j:
                    dx = positions[j, 0] - positions[i, 0]
                    dy = positions[j, 1] - positions[i, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    if dist > 1e-10:
                        r_max = min(r_max, dist * 0.45)
            radii[i] = max(r_max * 0.9, 0.005)
        return radii
    
    def lj_relax(positions, radii, w, h, max_iter=2000, sigma_scale=1.0):
        """
        Lennard-Jones-like soft-body simulation for circle packing.
        Uses repulsive potentials between circles and boundary constraints.
        """
        positions = positions.copy()
        radii = radii.copy()
        
        # LJ parameters
        sigma = 0.05  # interaction range scale
        epsilon = 50.0  # interaction strength
        boundary_stiffness = 100.0
        radius_growth_rate = 0.02
        damping = 0.85
        
        # Velocity tracking for momentum
        velocities = np.zeros_like(positions)
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            forces = np.zeros_like(positions)
            
            # Boundary forces (smooth spring)
            for i in range(N):
                bl = positions[i, 0] - radii[i]
                br = w - positions[i, 0] - radii[i]
                bb = positions[i, 1] - radii[i]
                bt = h - positions[i, 1] - radii[i]
                
                if bl < 0:
                    forces[i, 0] += boundary_stiffness * bl
                if br < 0:
                    forces[i, 0] -= boundary_stiffness * br
                if bb < 0:
                    forces[i, 1] += boundary_stiffness * bb
                if bt < 0:
                    forces[i, 1] -= boundary_stiffness * bt
            
            # Pairwise LJ-like repulsive forces
            for i in range(N):
                for j in range(i + 1, N):
                    dx = positions[j, 0] - positions[i, 0]
                    dy = positions[j, 1] - positions[i, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    
                    if dist < 1e-10:
                        dx = np.random.randn() * 0.01
                        dy = np.random.randn() * 0.01
                        dist = np.sqrt(dx*dx + dy*dy)
                    
                    # Effective "diameter" for interaction
                    eff_dist = dist / (radii[i] + radii[j])
                    
                    if eff_dist < 2.0:
                        # LJ potential: (sigma/r)^12 - 2*(sigma/r)^6
                        # We use a modified version focused on repulsion
                        ratio = sigma_scale * (radii[i] + radii[j]) / dist
                        if ratio > 1.0:
                            # Strong repulsion when overlapping
                            force_mag = epsilon * (ratio ** 12 - ratio ** 6)
                        else:
                            # Weak attraction at moderate distances (helps clustering)
                            force_mag = epsilon * (ratio ** 12 - ratio ** 6) * 0.1
                        
                        fx = force_mag * dx / dist
                        fy = force_mag * dy / dist
                        forces[i, 0] -= fx
                        forces[i, 1] -= fy
                        forces[j, 0] += fx
                        forces[j, 1] += fy
            
            # Update velocities with damping
            velocities = velocities * damping + forces * 0.01
            positions += velocities
            
            # Grow radii toward maximum possible
            for i in range(N):
                r_max = min(positions[i, 0], w - positions[i, 0],
                            positions[i, 1], h - positions[i, 1])
                for j in range(N):
                    if i != j:
                        dx = positions[j, 0] - positions[i, 0]
                        dy = positions[j, 1] - positions[i, 1]
                        dist = np.sqrt(dx * dx + dy * dy)
                        if dist > 1e-10:
                            r_max = min(r_max, (dist - radii[j]) * 0.95)
                r_max = max(r_max, 0.001)
                radii[i] += (r_max - radii[i]) * radius_growth_rate
            
            # Clamp positions
            radii = np.maximum(radii, 0.001)
            for i in range(N):
                positions[i, 0] = np.clip(positions[i, 0], radii[i], w - radii[i])
                positions[i, 1] = np.clip(positions[i, 1], radii[i], h - radii[i])
        
        return positions, radii
    
    def gradient_optimize(positions, radii, w, h, max_iter=3000):
        """
        Gradient-based optimization with careful step size control.
        Uses analytic gradients for the sum of radii objective with constraints.
        """
        positions = positions.copy()
        radii = radii.copy()
        
        step_size = 0.01
        patience = 0
        best_local = np.sum(radii)
        best_positions = positions.copy()
        best_radii = radii.copy()
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            # Compute gradient of sum(radii) with respect to positions and radii
            grad_pos = np.zeros_like(positions)
            grad_rad = np.ones(N)  # d(sum)/dr_i = 1
            
            # Boundary constraints: if circle touches boundary, gradient pushes away
            for i in range(N):
                # Left wall
                if positions[i, 0] - radii[i] < 1e-6:
                    grad_pos[i, 0] -= 10.0
                # Right wall
                if positions[i, 0] + radii[i] > w - 1e-6:
                    grad_pos[i, 0] += 10.0
                # Bottom wall
                if positions[i, 1] - radii[i] < 1e-6:
                    grad_pos[i, 1] -= 10.0
                # Top wall
                if positions[i, 1] + radii[i] > h - 1e-6:
                    grad_pos[i, 1] += 10.0
                
                # Pairwise overlap constraints
                for j in range(i + 1, N):
                    dx = positions[j, 0] - positions[i, 0]
                    dy = positions[j, 1] - positions[i, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    min_dist = radii[i] + radii[j]
                    
                    if dist < min_dist + 0.01 and dist > 1e-10:
                        # Overlap or near-overlap: push apart
                        overlap = min_dist - dist
                        if overlap > 0:
                            grad_pos[i, 0] -= 50.0 * dx / dist
                            grad_pos[i, 1] -= 50.0 * dy / dist
                            grad_pos[j, 0] += 50.0 * dx / dist
                            grad_pos[j, 1] += 50.0 * dy / dist
                            # Also penalize radius growth
                            grad_rad[i] -= 20.0 * overlap
                            grad_rad[j] -= 20.0 * overlap
            
            # Compute max possible radius for each circle
            for i in range(N):
                r_max = min(positions[i, 0], w - positions[i, 0],
                            positions[i, 1], h - positions[i, 1])
                for j in range(N):
                    if i != j:
                        dx = positions[j, 0] - positions[i, 0]
                        dy = positions[j, 1] - positions[i, 1]
                        dist = np.sqrt(dx * dx + dy * dy)
                        if dist > 1e-10:
                            r_max = min(r_max, dist - radii[j])
                r_max = max(r_max, 0.001)
                
                # If current radius is much less than max, encourage growth
                if radii[i] < r_max * 0.9:
                    grad_rad[i] += 5.0 * (r_max - radii[i])
                
                # If radius is at max, no more growth
                if radii[i] >= r_max * 0.99:
                    grad_rad[i] = max(grad_rad[i], 0.0)
            
            # Adaptive step size
            grad_norm = np.sqrt(np.sum(grad_pos**2) + np.sum(grad_rad**2))
            if grad_norm > 1e-10:
                step = step_size / grad_norm
            else:
                step = step_size
            
            # Update
            positions += grad_pos * step
            radii += grad_rad * step
            
            # Clamp
            radii = np.maximum(radii, 0.001)
            for i in range(N):
                positions[i, 0] = np.clip(positions[i, 0], radii[i], w - radii[i])
                positions[i, 1] = np.clip(positions[i, 1], radii[i], h - radii[i])
            
            # Check validity and track best
            if is_valid(positions, radii, w, h):
                current_score = np.sum(radii)
                if current_score > best_local:
                    best_local = current_score
                    best_positions = positions.copy()
                    best_radii = radii.copy()
                    patience = 0
                else:
                    patience += 1
            else:
                # Revert to last valid state
                positions = best_positions.copy()
                radii = best_radii.copy()
                patience += 1
                step_size *= 0.95
            
            # Reduce step size if no improvement
            if patience > 100:
                step_size *= 0.9
                if step_size < 1e-6:
                    break
            elif patience == 0:
                step_size = min(step_size * 1.05, 0.05)
        
        return best_positions, best_radii
    
    def scipy_optimize(positions, radii, w, h):
        """Use scipy L-BFGS-B with penalty-based objective."""
        n = N
        x0 = np.zeros(3 * n)
        for i in range(n):
            x0[3*i] = positions[i, 0]
            x0[3*i+1] = positions[i, 1]
            x0[3*i+2] = radii[i]
        
        def objective(x):
            pos = np.array([[x[3*i], x[3*i+1]] for i in range(n)])
            rad = np.array([x[3*i+2] for i in range(n)])
            
            penalty = 0.0
            # Boundary penalties
            for i in range(n):
                bl = max(0.0, rad[i] - pos[i, 0])
                br = max(0.0, pos[i, 0] + rad[i] - w)
                bb = max(0.0, rad[i] - pos[i, 1])
                bt = max(0.0, pos[i, 1] + rad[i] - h)
                penalty += 1000.0 * (bl**2 + br**2 + bb**2 + bt**2)
            
            # Overlap penalties
            for i in range(n):
                for j in range(i + 1, n):
                    dx = pos[j, 0] - pos[i, 0]
                    dy = pos[j, 1] - pos[i, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    min_dist = rad[i] + rad[j]
                    overlap = max(0.0, min_dist - dist)
                    penalty += 1000.0 * overlap**2
            
            return -np.sum(rad) + penalty
        
        bounds = []
        for i in range(n):
            bounds.append((0.001, w - 0.001))
            bounds.append((0.001, h - 0.001))
            bounds.append((0.001, min(w, h)/2))
        
        try:
            result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 800, 'ftol': 1e-12, 'gtol': 1e-10})
            
            pos = np.array([[result.x[3*i], result.x[3*i+1]] for i in range(n)])
            rad = np.array([result.x[3*i+2] for i in range(n)])
            
            for i in range(n):
                pos[i, 0] = np.clip(pos[i, 0], rad[i], w - rad[i])
                pos[i, 1] = np.clip(pos[i, 1], rad[i], h - rad[i])
                rad[i] = max(rad[i], 0.001)
            
            if is_valid(pos, rad, w, h):
                return pos, rad
        except:
            pass
        
        return None, None
    
    def polish(positions, radii, w, h, max_iter=1000):
        """Final high-precision polish with very small steps."""
        positions = positions.copy()
        radii = radii.copy()
        step_size = 0.001
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            grad_pos = np.zeros_like(positions)
            grad_rad = np.ones(N)
            
            for i in range(N):
                if positions[i, 0] - radii[i] < 1e-8:
                    grad_pos[i, 0] -= 100.0
                if positions[i, 0] + radii[i] > w - 1e-8:
                    grad_pos[i, 0] += 100.0
                if positions[i, 1] - radii[i] < 1e-8:
                    grad_pos[i, 1] -= 100.0
                if positions[i, 1] + radii[i] > h - 1e-8:
                    grad_pos[i, 1] += 100.0
                
                for j in range(i + 1, N):
                    dx = positions[j, 0] - positions[i, 0]
                    dy = positions[j, 1] - positions[i, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    min_dist = radii[i] + radii[j]
                    
                    if dist < min_dist + 0.001 and dist > 1e-10:
                        overlap = min_dist - dist
                        if overlap > 0:
                            grad_pos[i, 0] -= 200.0 * dx / dist
                            grad_pos[i, 1] -= 200.0 * dy / dist
                            grad_pos[j, 0] += 200.0 * dx / dist
                            grad_pos[j, 1] += 200.0 * dy / dist
                            grad_rad[i] -= 50.0 * overlap
                            grad_rad[j] -= 50.0 * overlap
            
            for i in range(N):
                r_max = min(positions[i, 0], w - positions[i, 0],
                            positions[i, 1], h - positions[i, 1])
                for j in range(N):
                    if i != j:
                        dx = positions[j, 0] - positions[i, 0]
                        dy = positions[j, 1] - positions[i, 1]
                        dist = np.sqrt(dx * dx + dy * dy)
                        if dist > 1e-10:
                            r_max = min(r_max, dist - radii[j])
                r_max = max(r_max, 0.001)
                if radii[i] < r_max * 0.99:
                    grad_rad[i] += 10.0 * (r_max - radii[i])
                else:
                    grad_rad[i] = max(grad_rad[i], 0.0)
            
            grad_norm = np.sqrt(np.sum(grad_pos**2) + np.sum(grad_rad**2))
            if grad_norm > 1e-10:
                step = step_size / grad_norm
            else:
                step = step_size
            
            positions += grad_pos * step
            radii += grad_rad * step
            
            radii = np.maximum(radii, 0.001)
            for i in range(N):
                positions[i, 0] = np.clip(positions[i, 0], radii[i], w - radii[i])
                positions[i, 1] = np.clip(positions[i, 1], radii[i], h - radii[i])
        
        return positions, radii
    
    # Main search loop over aspect ratios
    aspect_ratios = [1.0, 1.2, 0.8, 1.5, 0.67, 2.0, 0.5, 1.4, 0.7, 1.3, 1.1, 0.9, 1.6, 0.63, 1.8, 0.55]
    
    for ar in aspect_ratios:
        if time.time() > deadline - 30:
            break
        
        w = ar * (2.0 / (1.0 + ar))
        h = 2.0 - w
        
        if w < 0.1 or h < 0.1:
            continue
        
        # Choose grid dimensions
        if ar >= 1.0:
            cols = int(np.round(np.sqrt(N * ar)))
            rows = int(np.ceil(N / cols))
        else:
            rows = int(np.round(np.sqrt(N / ar)))
            cols = int(np.ceil(N / rows))
        
        while cols * rows < N:
            if ar >= 1.0:
                cols += 1
            else:
                rows += 1
        
        # Multiple initialization strategies
        inits = []
        
        # Grid
        pos_grid = grid_positions(w, h, cols, rows)
        if len(pos_grid) >= N:
            inits.append(('grid', pos_grid[:N].copy(), initial_radii(pos_grid[:N], w, h)))
        
        # Hexagonal
        pos_hex = hexagonal_positions(w, h, cols, rows)
        if len(pos_hex) >= N:
            inits.append(('hex', pos_hex[:N].copy(), initial_radii(pos_hex[:N], w, h)))
        
        # Random
        pos_rand, rad_rand = random_initialization(w, h)
        inits.append(('random', pos_rand, rad_rand))
        
        # Another random with different seed
        np.random.seed(int(time.time() * 1000) % 10000)
        pos_rand2, rad_rand2 = random_initialization(w, h)
        inits.append(('random2', pos_rand2, rad_rand2))
        
        for init_name, pos, rad in inits:
            if time.time() > deadline - 10:
                break
            
            # Phase 1: LJ relaxation
            pos1, rad1 = lj_relax(pos, rad, w, h, max_iter=1500)
            score1 = compute_score(pos1, rad1, w, h)
            if score1 > best_score:
                best_score = score1
                best_result = np.column_stack([pos1, rad1]).copy()
            
            # Phase 2: Gradient optimization
            if time.time() < deadline - 15:
                pos2, rad2 = gradient_optimize(pos1, rad1, w, h, max_iter=2000)
                score2 = compute_score(pos2, rad2, w, h)
                if score2 > best_score:
                    best_score = score2
                    best_result = np.column_stack([pos2, rad2]).copy()
            
            # Phase 3: Scipy optimization
            if time.time() < deadline - 8:
                opt_pos, opt_rad = scipy_optimize(pos2, rad2, w, h)
                if opt_pos is not None:
                    score3 = compute_score(opt_pos, opt_rad, w, h)
                    if score3 > best_score:
                        best_score = score3
                        best_result = np.column_stack([opt_pos, opt_rad]).copy()
            
            # Phase 4: Polish
            if time.time() < deadline - 2 and best_result is not None:
                pos4 = best_result[:, :2].copy()
                rad4 = best_result[:, 2].copy()
                pos4, rad4 = polish(pos4, rad4, w, h, max_iter=500)
                score4 = compute_score(pos4, rad4, w, h)
                if score4 > best_score:
                    best_score = score4
                    best_result = np.column_stack([pos4, rad4]).copy()
    
    # Fallback
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result
# EVOLVE-BLOCK-END
