"""
Circle packing in unit square using Simulated Annealing.
Centers are optimized; radii are derived from the local geometry
(half the minimum distance to any other center or to a boundary).
This enforces tight packing and allows escaping local minima via stochastic search.
"""
import numpy as np
import time

N = 26
DEADLINE = 115.0  # Safety margin from 120s limit


def construct():
    start_time = time.time()

    def compute_radii(x, y):
        """Compute exact maximum radii for given centers."""
        # Boundary distances
        r_bound = np.minimum(
            np.minimum(x, 1.0 - x),
            np.minimum(y, 1.0 - y)
        )

        # Pairwise distances
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist, np.inf)

        # Minimum distance to any other circle
        min_dist = np.min(dist, axis=1)

        # Radius is limited by boundary and half the distance to the nearest neighbor
        r = np.minimum(r_bound, 0.5 * min_dist)
        # Ensure positive
        r = np.maximum(r, 1e-6)
        return r

    def check_validity(x, y, r):
        # Check boundary
        if np.any(x - r < -1e-9) or np.any(x + r > 1 + 1e-9):
            return False
        if np.any(y - r < -1e-9) or np.any(y + r > 1 + 1e-9):
            return False
        if np.any(r < -1e-9):
            return False

        # Check non-overlap
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        np.fill_diagonal(dist, np.inf)
        np.fill_diagonal(min_dist, 0)

        if np.any(dist < min_dist - 1e-9):
            return False
        return True

    def make_initial_guess(seed):
        rng = np.random.RandomState(seed)
        # Start with a grid and perturb
        side = int(np.ceil(np.sqrt(N)))
        x = np.zeros(N)
        y = np.zeros(N)
        idx = 0
        for i in range(side):
            for j in range(side):
                if idx >= N:
                    break
                x[idx] = (i + 0.5) / side
                y[idx] = (j + 0.5) / side
                idx += 1

        # Perturb
        x += rng.uniform(-0.05, 0.05, N)
        y += rng.uniform(-0.05, 0.05, N)

        # Clip to valid range (keep away from boundaries to allow growth)
        x = np.clip(x, 0.05, 0.95)
        y = np.clip(y, 0.05, 0.95)

        return x, y

    # Initial best: grid configuration
    side = int(np.ceil(np.sqrt(N)))
    best_x = np.zeros(N)
    best_y = np.zeros(N)
    idx = 0
    for i in range(side):
        for j in range(side):
            if idx >= N:
                break
            best_x[idx] = (i + 0.5) / side
            best_y[idx] = (j + 0.5) / side
            idx += 1
    best_r = compute_radii(best_x, best_y)
    best_val = np.sum(best_r)

    # Simulated Annealing Parameters
    # We will run multiple restarts with different seeds and SA schedules
    num_restarts = 5
    
    for restart in range(num_restarts):
        if time.time() > DEADLINE:
            break
            
        x, y = make_initial_guess(restart * 100)
        r = compute_radii(x, y)
        current_val = np.sum(r)
        
        # If the initial guess is already better, update
        if current_val > best_val:
            best_val = current_val
            best_x = x.copy()
            best_y = y.copy()
            best_r = r.copy()

        # SA Schedule
        T_start = 0.05
        T_end = 0.001
        max_steps = 50000 
        step_size = 0.05
        
        rng = np.random.RandomState(restart * 100 + 42)
        
        for step in range(max_steps):
            if time.time() > DEADLINE:
                break
                
            # Cool temperature
            progress = step / max_steps
            T = T_start * (T_end / T_start) ** progress
            
            # Pick a random circle to move
            i = rng.randint(N)
            
            # Save old position and radius
            old_x = x[i]
            old_y = y[i]
            old_r = r.copy()
            
            # Generate new position
            new_x = old_x + rng.normal(0, step_size)
            new_y = old_y + rng.normal(0, step_size)
            
            # Clip to boundaries
            new_x = np.clip(new_x, 0.0, 1.0)
            new_y = np.clip(new_y, 0.0, 1.0)
            
            # Update position
            x[i] = new_x
            y[i] = new_y
            
            # Compute new radii
            new_r = compute_radii(x, y)
            new_val = np.sum(new_r)
            
            # Calculate delta
            delta = new_val - current_val
            
            # Metropolis criterion
            if delta > 0:
                # Accept better solution
                current_val = new_val
                r = new_r
            else:
                # Accept worse solution with probability exp(delta / T)
                prob = np.exp(delta / T)
                if rng.random() < prob:
                    current_val = new_val
                    r = new_r
                else:
                    # Revert
                    x[i] = old_x
                    y[i] = old_y
                    r = old_r
            
            # Update best
            if current_val > best_val:
                best_val = current_val
                best_x = x.copy()
                best_y = y.copy()
                best_r = r.copy()

            # Adaptive step size: decay slightly
            if step % 10000 == 0:
                step_size *= 0.8

    # Final validity check
    if not check_validity(best_x, best_y, best_r):
        for scale in np.linspace(0.99, 0.90, 10):
            r_test = best_r * scale
            if check_validity(best_x, best_y, r_test):
                best_r = r_test
                break
        else:
            # Fallback to grid
            best_r = compute_radii(best_x, best_y)

    return np.column_stack((best_x, best_y, best_r))
