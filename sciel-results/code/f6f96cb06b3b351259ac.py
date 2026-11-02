"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    """
    Pack 26 circles in the unit square, maximizing the sum of radii.
    Uses a multi-stage approach:
    1. Vectorized pairwise relaxation with position + radius adjustment
    2. Gradient-based refinement using penalty method
    3. Multiple restarts with different initial configurations
    """
    start_time = time.time()
    deadline = 115.0  # safety margin before 120s limit
    n = N

    def _grid_initial(seed=0):
        """Create a grid-based initial configuration."""
        rng = np.random.RandomState(seed)
        side = 5
        positions = []
        for i in range(side):
            for j in range(side):
                x = (i + 0.5) / side + rng.uniform(-0.02, 0.02)
                y = (j + 0.5) / side + rng.uniform(-0.02, 0.02)
                positions.append([x, y])
        # Add 26th circle
        positions.append([0.5 + rng.uniform(-0.05, 0.05), 0.5 + rng.uniform(-0.05, 0.05)])
        positions = np.array(positions, dtype=np.float64)
        positions[:, 0] = np.clip(positions[:, 0], 0.05, 0.95)
        positions[:, 1] = np.clip(positions[:, 1], 0.05, 0.95)
        radii = np.ones(n, dtype=np.float64) * 0.08
        return positions, radii

    def _hex_initial(seed=0):
        """Create a hexagonal-grid initial configuration."""
        rng = np.random.RandomState(seed)
        positions = []
        # 5 rows, alternating 5 and 5 with offset
        for row in range(5):
            y = (row + 0.5) / 5.0
            for col in range(5):
                x = (col + 0.5) / 5.0
                if row % 2 == 1:
                    x += 0.1 / 5.0
                x = np.clip(x, 0.05, 0.95)
                positions.append([x, y])
        # Add 26th in a gap
        positions.append([0.5 + rng.uniform(-0.03, 0.03), 0.5 + rng.uniform(-0.03, 0.03)])
        positions = np.array(positions, dtype=np.float64)
        positions[:, 0] = np.clip(positions[:, 0], 0.05, 0.95)
        positions[:, 1] = np.clip(positions[:, 1], 0.05, 0.95)
        radii = np.ones(n, dtype=np.float64) * 0.08
        return positions, radii

    def _random_initial(seed=0):
        """Create a random initial configuration."""
        rng = np.random.RandomState(seed)
        positions = rng.uniform(0.1, 0.9, size=(n, 2))
        radii = np.ones(n, dtype=np.float64) * 0.05
        return positions, radii

    def _enforce_validity(pos, rad):
        """Ensure all circles are inside the unit square and non-overlapping."""
        pos = pos.copy()
        rad = rad.copy()
        
        # Clamp positions
        pos[:, 0] = np.clip(pos[:, 0], 0, 1)
        pos[:, 1] = np.clip(pos[:, 1], 0, 1)
        
        # Enforce boundary: radius <= min distance to boundary
        for i in range(n):
            max_r = min(pos[i, 0], pos[i, 1], 1 - pos[i, 0], 1 - pos[i, 1])
            rad[i] = min(rad[i], max_r)
        
        # Enforce non-overlap by reducing radii
        for _ in range(50):
            changed = False
            for i in range(n):
                for j in range(i + 1, n):
                    dx = pos[j, 0] - pos[i, 0]
                    dy = pos[j, 1] - pos[i, 1]
                    d = np.sqrt(dx * dx + dy * dy)
                    if d < 1e-12:
                        # Push apart randomly
                        pos[i, 0] += rng.uniform(-0.01, 0.01)
                        pos[i, 1] += rng.uniform(-0.01, 0.01)
                        pos[j, 0] += rng.uniform(-0.01, 0.01)
                        pos[j, 1] += rng.uniform(-0.01, 0.01)
                        continue
                    overlap = rad[i] + rad[j] - d
                    if overlap > 1e-10:
                        # Reduce radii proportionally
                        total = rad[i] + rad[j]
                        if total > 0:
                            rad[i] -= overlap * (rad[i] / total) * 0.5
                            rad[j] -= overlap * (rad[j] / total) * 0.5
                            rad[i] = max(rad[i], 0.001)
                            rad[j] = max(rad[j], 0.001)
                            changed = True
            if not changed:
                break
        
        # Final boundary check
        for i in range(n):
            max_r = min(pos[i, 0], pos[i, 1], 1 - pos[i, 0], 1 - pos[i, 1])
            rad[i] = min(rad[i], max_r)
        
        return pos, rad

    def _score(pos, rad):
        """Compute the sum of radii (the objective)."""
        return np.sum(rad)

    def _is_valid(pos, rad, tol=1e-9):
        """Check if configuration is valid."""
        # Boundary check
        for i in range(n):
            if pos[i, 0] < -tol or pos[i, 0] > 1 + tol:
                return False
            if pos[i, 1] < -tol or pos[i, 1] > 1 + tol:
                return False
            if rad[i] < -tol:
                return False
            if rad[i] > min(pos[i, 0], pos[i, 1], 1 - pos[i, 0], 1 - pos[i, 1]) + tol:
                return False
        # Non-overlap check
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                d = np.sqrt(dx * dx + dy * dy)
                if d < rad[i] + rad[j] - tol:
                    return False
        return True

    def _vectorized_relax(pos, rad, n_iters=500):
        """Vectorized pairwise relaxation for faster convergence."""
        pos = pos.copy()
        rad = rad.copy()
        
        for it in range(n_iters):
            if time.time() - start_time > deadline - 5.0:
                break
            
            # Compute all pairwise distances
            dx = pos[:, 0][None, :] - pos[:, 0][:, None]
            dy = pos[:, 1][None, :] - pos[:, 1][:, None]
            dist = np.sqrt(dx * dx + dy * dy)
            np.fill_diagonal(dist, np.inf)
            
            # Compute overlaps
            rad_sum = rad[None, :] + rad[:, None]
            overlap = rad_sum - dist
            overlap = np.maximum(overlap, 0)
            
            # Push apart: compute forces
            # For each pair, if overlapping, push in direction of separation
            force_x = np.zeros((n, n))
            force_y = np.zeros((n, n))
            
            for i in range(n):
                for j in range(i + 1, n):
                    if overlap[i, j] > 1e-12:
                        d = dist[i, j]
                        if d > 1e-12:
                            ux = (pos[j, 0] - pos[i, 0]) / d
                            uy = (pos[j, 1] - pos[i, 1]) / d
                            push = overlap[i, j] * 0.5
                            force_x[i, j] -= ux * push
                            force_y[i, j] -= uy * push
                            force_x[j, i] += ux * push
                            force_y[j, i] += uy * push
                        else:
                            # Random push
                            force_x[i, j] += random.uniform(-0.01, 0.01)
                            force_y[i, j] += random.uniform(-0.01, 0.01)
                            force_x[j, i] -= force_x[i, j]
                            force_y[j, i] -= force_y[i, j]
            
            # Apply forces
            pos[:, 0] += np.sum(force_x, axis=1)
            pos[:, 1] += np.sum(force_y, axis=1)
            
            # Clamp positions
            pos[:, 0] = np.clip(pos[:, 0], 0, 1)
            pos[:, 1] = np.clip(pos[:, 1], 0, 1)
            
            # Reduce radii for overlapping pairs
            for i in range(n):
                for j in range(i + 1, n):
                    if overlap[i, j] > 1e-10:
                        total = rad[i] + rad[j]
                        if total > 0:
                            reduce_i = overlap[i, j] * (rad[i] / total) * 0.3
                            reduce_j = overlap[i, j] * (rad[j] / total) * 0.3
                            rad[i] = max(rad[i] - reduce_i, 0.001)
                            rad[j] = max(rad[j] - reduce_j, 0.001)
            
            # Enforce boundary
            for i in range(n):
                max_r = min(pos[i, 0], pos[i, 1], 1 - pos[i, 0], 1 - pos[i, 1])
                rad[i] = min(rad[i], max_r)
            
            # Try to grow radii
            for i in range(n):
                max_r = min(pos[i, 0], pos[i, 1], 1 - pos[i, 0], 1 - pos[i, 1])
                for j in range(n):
                    if i != j:
                        d = dist[i, j]
                        max_r = min(max_r, d - rad[j])
                rad[i] = max(max_r, 0.001)
        
        return pos, rad

    def _gradient_refine(pos, rad, n_iters=300):
        """Gradient-based refinement using penalty method."""
        pos = pos.copy()
        rad = rad.copy()
        
        penalty = 100.0
        lr = 0.01
        
        for it in range(n_iters):
            if time.time() - start_time > deadline - 5.0:
                break
            
            # Compute gradient of penalty objective
            grad_pos = np.zeros((n, 2))
            grad_rad = np.zeros(n)
            
            # Boundary penalty
            for i in range(n):
                # x constraint
                if pos[i, 0] < rad[i]:
                    grad_pos[i, 0] += penalty * (pos[i, 0] - rad[i])
                    grad_rad[i] -= penalty * (pos[i, 0] - rad[i])
                if pos[i, 0] > 1 - rad[i]:
                    grad_pos[i, 0] += penalty * (pos[i, 0] - (1 - rad[i]))
                    grad_rad[i] -= penalty * (pos[i, 0] - (1 - rad[i]))
                if pos[i, 1] < rad[i]:
                    grad_pos[i, 1] += penalty * (pos[i, 1] - rad[i])
                    grad_rad[i] -= penalty * (pos[i, 1] - rad[i])
                if pos[i, 1] > 1 - rad[i]:
                    grad_pos[i, 1] += penalty * (pos[i, 1] - (1 - rad[i]))
                    grad_rad[i] -= penalty * (pos[i, 1] - (1 - rad[i]))
            
            # Non-overlap penalty
            for i in range(n):
                for j in range(i + 1, n):
                    dx = pos[j, 0] - pos[i, 0]
                    dy = pos[j, 1] - pos[i, 1]
                    d = np.sqrt(dx * dx + dy * dy)
                    if d < 1e-12:
                        continue
                    overlap = rad[i] + rad[j] - d
                    if overlap > 0:
                        # Gradient with respect to positions
                        gx = penalty * overlap * dx / d
                        gy = penalty * overlap * dy / d
                        grad_pos[i, 0] += gx
                        grad_pos[i, 1] += gy
                        grad_pos[j, 0] -= gx
                        grad_pos[j, 1] -= gy
                        # Gradient with respect to radii
                        grad_rad[i] += penalty * overlap
                        grad_rad[j] += penalty * overlap
            
            # Objective gradient (maximize sum of radii)
            grad_rad += 1.0
            
            # Update
            pos[:, 0] += lr * grad_pos[:, 0]
            pos[:, 1] += lr * grad_pos[:, 1]
            rad += lr * grad_rad
            
            # Clamp
            pos[:, 0] = np.clip(pos[:, 0], 0, 1)
            pos[:, 1] = np.clip(pos[:, 1], 0, 1)
            rad = np.clip(rad, 0.001, 0.5)
            
            # Decay learning rate, increase penalty
            lr *= 0.995
            penalty *= 1.002
            
            # Periodic validity enforcement
            if it % 50 == 0:
                pos, rad = _enforce_validity(pos, rad)
        
        return pos, rad

    def _local_search(pos, rad, n_iters=200):
        """Local search with small random perturbations."""
        pos = pos.copy()
        rad = rad.copy()
        best_score = _score(pos, rad)
        best_pos = pos.copy()
        best_rad = rad.copy()
        step_size = 0.01
        
        for it in range(n_iters):
            if time.time() - start_time > deadline - 5.0:
                break
            
            # Random perturbation
            trial_pos = pos.copy()
            trial_rad = rad.copy()
            
            # Perturb a random subset
            idx = np.random.choice(n, size=max(1, n // 4), replace=False)
            trial_pos[idx, 0] += np.random.uniform(-step_size, step_size, size=len(idx))
            trial_pos[idx, 1] += np.random.uniform(-step_size, step_size, size=len(idx))
            trial_rad[idx] += np.random.uniform(-step_size * 0.5, step_size * 0.5, size=len(idx))
            
            # Enforce validity
            trial_pos, trial_rad = _enforce_validity(trial_pos, trial_rad)
            
            score = _score(trial_pos, trial_rad)
            if score > best_score:
                best_score = score
                best_pos = trial_pos.copy()
                best_rad = trial_rad.copy()
                pos = trial_pos.copy()
                rad = trial_rad.copy()
                step_size *= 0.95
            else:
                step_size *= 0.98
        
        return best_pos, best_rad

    # Main optimization loop with multiple restarts
    best_pos = None
    best_rad = None
    best_score = -np.inf
    
    # Try different initial configurations
    initializers = [
        lambda: _grid_initial(0),
        lambda: _grid_initial(42),
        lambda: _hex_initial(0),
        lambda: _hex_initial(123),
        lambda: _random_initial(7),
        lambda: _random_initial(99),
    ]
    
    for init_fn in initializers:
        if time.time() - start_time > deadline - 10.0:
            break
        
        pos, rad = init_fn()
        pos, rad = _enforce_validity(pos, rad)
        
        # Stage 1: Vectorized relaxation
        pos, rad = _vectorized_relax(pos, rad, n_iters=300)
        pos, rad = _enforce_validity(pos, rad)
        
        # Stage 2: Gradient refinement
        pos, rad = _gradient_refine(pos, rad, n_iters=200)
        pos, rad = _enforce_validity(pos, rad)
        
        # Stage 3: Local search
        pos, rad = _local_search(pos, rad, n_iters=100)
        pos, rad = _enforce_validity(pos, rad)
        
        score = _score(pos, rad)
        if score > best_score:
            best_score = score
            best_pos = pos.copy()
            best_rad = rad.copy()
    
    # Final enforcement
    if best_pos is not None:
        best_pos, best_rad = _enforce_validity(best_pos, best_rad)
    else:
        best_pos, best_rad = _grid_initial(0)
        best_pos, best_rad = _enforce_validity(best_pos, best_rad)
    
    # Build result
    result = np.zeros((n, 3))
    result[:, 0] = best_pos[:, 0]
    result[:, 1] = best_pos[:, 1]
    result[:, 2] = best_rad
    
    return result
# EVOLVE-BLOCK-END
