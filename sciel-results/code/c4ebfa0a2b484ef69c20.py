import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    best_result = None
    best_score = -1.0

    i_idx, j_idx = np.triu_indices(N, k=1)

    def is_valid(positions, radii, w, h, tol=1e-9):
        if np.any(positions[:, 0] - radii < -tol) or np.any(positions[:, 0] + radii > w + tol):
            return False
        if np.any(positions[:, 1] - radii < -tol) or np.any(positions[:, 1] + radii > h + tol):
            return False
        dx = positions[j_idx, 0] - positions[i_idx, 0]
        dy = positions[j_idx, 1] - positions[i_idx, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        min_dist = radii[i_idx] + radii[j_idx]
        if np.any(dist < min_dist - tol):
            return False
        return True

    def compute_score(positions, radii, w, h):
        if not is_valid(positions, radii, w, h):
            return -1.0
        return np.sum(radii)

    def relax_vectorized(positions, radii, w, h, max_iter=500):
        # Fast force-based relaxation
        lr = 0.05
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            dx = positions[j_idx, 0] - positions[i_idx, 0]
            dy = positions[j_idx, 1] - positions[i_idx, 1]
            dist_sq = dx * dx + dy * dy
            dist = np.sqrt(dist_sq)
            dist[dist < 1e-8] = 1e-8
            
            min_dist = radii[i_idx] + radii[j_idx]
            overlap = min_dist - dist
            
            forces = np.zeros((N, 2))
            mask = overlap > 0
            if np.any(mask):
                dir_x = dx[mask] / dist[mask]
                dir_y = dy[mask] / dist[mask]
                force_mag = overlap[mask] * 2.0
                np.add.at(forces, i_idx[mask], -force_mag[:, np.newaxis] * np.stack([dir_x, dir_y], axis=-1))
                np.add.at(forces, j_idx[mask], force_mag[:, np.newaxis] * np.stack([dir_x, dir_y], axis=-1))
            
            left_viol = np.maximum(0, radii - positions[:, 0])
            right_viol = np.maximum(0, positions[:, 0] + radii - w)
            bottom_viol = np.maximum(0, radii - positions[:, 1])
            top_viol = np.maximum(0, positions[:, 1] + radii - h)
            
            forces[:, 0] += (left_viol - right_viol) * 5.0
            forces[:, 1] += (bottom_viol - top_viol) * 5.0
            
            r_boundary = np.minimum(np.minimum(positions[:, 0], w - positions[:, 0]), 
                                    np.minimum(positions[:, 1], h - positions[:, 1]))
            
            r_neighbor_max = np.full(N, np.inf)
            space_for_i = dist - radii[j_idx]
            space_for_j = dist - radii[i_idx]
            np.minimum.at(r_neighbor_max, i_idx, space_for_i)
            np.minimum.at(r_neighbor_max, j_idx, space_for_j)
            
            r_max = np.minimum(r_boundary, r_neighbor_max)
            r_max = np.maximum(r_max, 0.001)
            
            dr = (r_max - radii) * 0.1
            positions += forces * lr
            radii += dr
            
            radii = np.maximum(radii, 0.001)
            positions[:, 0] = np.clip(positions[:, 0], radii, w - radii)
            positions[:, 1] = np.clip(positions[:, 1], radii, h - radii)
            
        return positions, radii

    def newton_polish(positions, radii, w, h, max_iter=20):
        # Newton-Raphson polish for active constraints
        if time.time() > deadline - 2:
            return positions, radii
            
        tol_active = 1e-4
        dx = positions[j_idx, 0] - positions[i_idx, 0]
        dy = positions[j_idx, 1] - positions[i_idx, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        
        active_pairs = dist < (radii[i_idx] + radii[j_idx]) + tol_active
        active_i = i_idx[active_pairs]
        active_j = j_idx[active_pairs]
        
        active_walls = []
        for i in range(N):
            if positions[i, 0] - radii[i] < tol_active: active_walls.append((i, 'left'))
            if w - positions[i, 0] - radii[i] < tol_active: active_walls.append((i, 'right'))
            if positions[i, 1] - radii[i] < tol_active: active_walls.append((i, 'bottom'))
            if h - positions[i, 1] - radii[i] < tol_active: active_walls.append((i, 'top'))
                
        if len(active_pairs) == 0 and len(active_walls) == 0:
            return positions, radii
            
        n_vars = 3 * N
        J = np.zeros((len(active_pairs) + len(active_walls), n_vars))
        R = np.zeros(len(active_pairs) + len(active_walls))
        
        row = 0
        for k in range(len(active_i)):
            i = active_i[k]
            j = active_j[k]
            xi, yi = positions[i]
            xj, yj = positions[j]
            ri, rj = radii[i], radii[j]
            dx_ij = xi - xj
            dy_ij = yi - yj
            r_sum = ri + rj
            R[row] = dx_ij**2 + dy_ij**2 - r_sum**2
            J[row, 3*i] = 2 * dx_ij
            J[row, 3*i+1] = 2 * dy_ij
            J[row, 3*i+2] = -2 * r_sum
            J[row, 3*j] = -2 * dx_ij
            J[row, 3*j+1] = -2 * dy_ij
            J[row, 3*j+2] = -2 * r_sum
            row += 1
            
        for i, wall in active_walls:
            if wall == 'left':
                R[row] = positions[i, 0] - radii[i]
                J[row, 3*i] = 1.0
                J[row, 3*i+2] = -1.0
            elif wall == 'right':
                R[row] = -positions[i, 0] - radii[i] + w
                J[row, 3*i] = -1.0
                J[row, 3*i+2] = -1.0
            elif wall == 'bottom':
                R[row] = positions[i, 1] - radii[i]
                J[row, 3*i+1] = 1.0
                J[row, 3*i+2] = -1.0
            elif wall == 'top':
                R[row] = -positions[i, 1] - radii[i] + h
                J[row, 3*i+1] = -1.0
                J[row, 3*i+2] = -1.0
            row += 1
            
        try:
            dx_sol, residuals, rank, s = np.linalg.lstsq(J, -R, rcond=None)
        except:
            return positions, radii
            
        # Try a few step sizes
        for alpha in [0.5, 0.2, 0.1, 0.05]:
            x_old = np.zeros(n_vars)
            for i in range(N):
                x_old[3*i] = positions[i, 0]
                x_old[3*i+1] = positions[i, 1]
                x_old[3*i+2] = radii[i]
                
            x_new = x_old + alpha * dx_sol
            new_pos = np.zeros((N, 2))
            new_rad = np.zeros(N)
            for i in range(N):
                new_pos[i, 0] = x_new[3*i]
                new_pos[i, 1] = x_new[3*i+1]
                new_rad[i] = x_new[3*i+2]
                
            new_rad = np.maximum(new_rad, 0.001)
            new_pos[:, 0] = np.clip(new_pos[:, 0], new_rad, w - new_rad)
            new_pos[:, 1] = np.clip(new_pos[:, 1], new_rad, h - new_rad)
            
            if is_valid(new_pos, new_rad, w, h):
                return new_pos, new_rad
        return positions, radii

    def perturb(positions, radii, w, h, strength=0.1):
        # Randomly perturb positions
        new_pos = positions.copy()
        # Move a random subset of circles
        n_perturb = np.random.randint(3, 8)
        idx = np.random.choice(N, n_perturb, replace=False)
        noise_x = np.random.normal(0, strength, n_perturb)
        noise_y = np.random.normal(0, strength, n_perturb)
        new_pos[idx, 0] += noise_x
        new_pos[idx, 1] += noise_y
        
        # Clamp to bounds
        new_pos[:, 0] = np.clip(new_pos[:, 0], radii, w - radii)
        new_pos[:, 1] = np.clip(new_pos[:, 1], radii, h - radii)
        
        # Optionally shrink radii slightly to allow re-relaxation
        new_rad = radii * 0.9
        return new_pos, new_rad

    # Main search loop
    aspect_ratios = [1.0, 1.2, 0.8, 1.4, 0.7, 1.6, 0.6, 1.8, 0.5, 2.0, 1.1, 0.9, 1.3, 0.75, 1.25]
    
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
        
        w = ar * (2.0 / (1.0 + ar))
        h = 2.0 - w
        
        # Initialize with a grid
        cols = int(np.round(np.sqrt(N * ar)))
        rows = int(np.ceil(N / cols))
        while cols * rows < N:
            cols += 1
            
        positions = np.zeros((N, 2))
        for i in range(N):
            row = i // cols
            col = i % cols
            positions[i, 0] = (col + 0.5) * w / cols
            positions[i, 1] = (row + 0.5) * h / rows
        
        # Initial radii
        dx = w / cols
        dy = h / rows
        r_init = 0.5 * min(dx, dy) * 0.95
        radii = np.full(N, r_init)
        for i in range(N):
            r_max = min(positions[i, 0], w - positions[i, 0],
                        positions[i, 1], h - positions[i, 1])
            radii[i] = min(r_init, r_max)
        
        # Relax and Polish
        positions, radii = relax_vectorized(positions, radii, w, h)
        positions, radii = newton_polish(positions, radii, w, h)
        
        score = compute_score(positions, radii, w, h)
        if score > best_score:
            best_score = score
            best_result = np.column_stack([positions, radii]).copy()
            
        # Perturbation loop
        # We use a simple hill-climbing with occasional random restarts of the perturbation strength
        current_pos, current_rad = positions.copy(), radii.copy()
        current_score = score
        
        # Run a few perturbation steps
        for _ in range(50):
            if time.time() > deadline - 1:
                break
                
            # Perturb
            p_pos, p_rad = perturb(current_pos, current_rad, w, h, strength=0.05)
            
            # Relax and Polish
            p_pos, p_rad = relax_vectorized(p_pos, p_rad, w, h, max_iter=200)
            p_pos, p_rad = newton_polish(p_pos, p_rad, w, h)
            
            p_score = compute_score(p_pos, p_rad, w, h)
            
            if p_score > current_score:
                current_pos, current_rad = p_pos, p_rad
                current_score = p_score
                if current_score > best_score:
                    best_score = current_score
                    best_result = np.column_stack([current_pos, current_rad]).copy()
            else:
                # If we don't improve, we might want to try a larger perturbation or reset
                # For simplicity, just try another small perturbation from the current best
                pass

    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result
