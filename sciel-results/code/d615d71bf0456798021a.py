import numpy as np
import time
import random

N = 50

def construct():
    start_time = time.time()
    max_time = 115.0

    def compute_dists(centers):
        """Compute full pairwise distance matrix."""
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        return dists

    def compute_score(centers, dists):
        """Compute score from centers and precomputed distance matrix."""
        min_pair = np.min(dists) / 2.0
        norms = np.linalg.norm(centers, axis=1)
        min_boundary = np.min(1.0 - norms)
        return min(min_pair, min_boundary)

    def update_dists_row(dists, centers, idx):
        """Update row and column idx of distance matrix after moving center[idx]."""
        diff = centers[idx] - centers
        new_dists = np.sqrt(np.sum(diff**2, axis=1))
        new_dists[idx] = np.inf
        dists[idx, :] = new_dists
        dists[:, idx] = new_dists

    def local_search(centers, time_budget, step_sizes):
        """
        Structural change: Use a multi-scale local search that directly
        addresses the binding constraints. Instead of random SA moves, we
        iteratively identify the most violated constraint and apply a
        targeted perturbation, with adaptive step sizes.
        """
        centers = centers.copy()
        dists = compute_dists(centers)
        current_score = compute_score(centers, dists)
        best_score = current_score
        best_centers = centers.copy()
        best_dists = dists.copy()

        end_time = start_time + time_budget
        no_improve = 0
        max_no_improve = 200

        # Multi-scale step sizes: start coarse, refine
        base_steps = step_sizes
        current_step_idx = 0
        scale_no_improve = 0

        while time.time() < end_time:
            if time.time() > start_time + max_time:
                break

            step = base_steps[current_step_idx]

            # Find the most violated constraint
            min_pair_dist = np.min(dists)
            i_min, j_min = np.unravel_index(np.argmin(dists), dists.shape)

            norms = np.linalg.norm(centers, axis=1)
            min_boundary_val = np.min(1.0 - norms)
            min_boundary_idx = np.argmin(1.0 - norms)

            # Determine which constraint is more binding
            # Score = min(min_pair/2, min_boundary)
            pair_contribution = min_pair_dist / 2.0
            boundary_contribution = min_boundary_val

            if pair_contribution <= boundary_contribution:
                # Pair constraint is binding: push i and j apart
                vec = centers[j_min] - centers[i_min]
                norm_vec = np.linalg.norm(vec)
                if norm_vec > 1e-10:
                    direction = vec / norm_vec
                    # Try several perturbation directions around the direct separation
                    angles = np.linspace(-np.pi/3, np.pi/3, 7)
                    best_new_score = current_score
                    best_new_centers = None
                    best_new_dists = None

                    for angle in angles:
                        cos_a = np.cos(angle)
                        sin_a = np.sin(angle)
                        # Rotate the separation direction slightly
                        rot_dir = np.array([
                            direction[0] * cos_a - direction[1] * sin_a,
                            direction[0] * sin_a + direction[1] * cos_a
                        ])

                        new_i = centers[i_min] - rot_dir * step * 0.5
                        new_j = centers[j_min] + rot_dir * step * 0.5

                        # Also try perpendicular component
                        perp_dir = np.array([-rot_dir[1], rot_dir[0]])
                        for perp_frac in [0.0, 0.3, -0.3]:
                            trial_i = new_i + perp_dir * step * 0.3 * perp_frac
                            trial_j = new_j - perp_dir * step * 0.3 * perp_frac

                            # Check boundary
                            if np.linalg.norm(trial_i) > 1.0 or np.linalg.norm(trial_j) > 1.0:
                                continue

                            # Trial move
                            old_i = centers[i_min].copy()
                            old_j = centers[j_min].copy()
                            centers[i_min] = trial_i
                            centers[j_min] = trial_j

                            # Update distances incrementally
                            update_dists_row(dists, centers, i_min)
                            update_dists_row(dists, centers, j_min)

                            new_score = compute_score(centers, dists)

                            if new_score > best_new_score:
                                best_new_score = new_score
                                best_new_centers = (centers.copy(), trial_i.copy(), trial_j.copy())

                            # Revert for next trial
                            centers[i_min] = old_i
                            centers[j_min] = old_j
                            update_dists_row(dists, centers, i_min)
                            update_dists_row(dists, centers, j_min)

                    if best_new_centers is not None and best_new_score > current_score:
                        centers[i_min] = best_new_centers[1]
                        centers[j_min] = best_new_centers[2]
                        update_dists_row(dists, centers, i_min)
                        update_dists_row(dists, centers, j_min)
                        current_score = best_new_score
                        no_improve = 0
                        scale_no_improve = 0

                        if current_score > best_score:
                            best_score = current_score
                            best_centers = centers.copy()
                            best_dists = dists.copy()
                    else:
                        no_improve += 1
                        scale_no_improve += 1
                else:
                    no_improve += 1
                    scale_no_improve += 1
            else:
                # Boundary constraint is binding: move that point inward
                idx = min_boundary_idx
                vec = centers[idx]
                norm_vec = np.linalg.norm(vec)
                if norm_vec > 1e-10:
                    direction = vec / norm_vec
                    # Try moving inward with slight angular variations
                    angles = np.linspace(-np.pi/4, np.pi/4, 5)
                    best_new_score = current_score
                    best_new_center = None

                    for angle in angles:
                        cos_a = np.cos(angle)
                        sin_a = np.sin(angle)
                        rot_dir = np.array([
                            direction[0] * cos_a - direction[1] * sin_a,
                            direction[0] * sin_a + direction[1] * cos_a
                        ])

                        trial = centers[idx] - rot_dir * step
                        if np.linalg.norm(trial) > 1.0:
                            continue

                        old = centers[idx].copy()
                        centers[idx] = trial
                        update_dists_row(dists, centers, idx)

                        new_score = compute_score(centers, dists)

                        if new_score > best_new_score:
                            best_new_score = new_score
                            best_new_center = trial.copy()

                        centers[idx] = old
                        update_dists_row(dists, centers, idx)

                    if best_new_center is not None and best_new_score > current_score:
                        centers[idx] = best_new_center
                        update_dists_row(dists, centers, idx)
                        current_score = best_new_score
                        no_improve = 0
                        scale_no_improve = 0

                        if current_score > best_score:
                            best_score = current_score
                            best_centers = centers.copy()
                            best_dists = dists.copy()
                    else:
                        no_improve += 1
                        scale_no_improve += 1
                else:
                    no_improve += 1
                    scale_no_improve += 1

            # Adapt step size scale
            if scale_no_improve > max_no_improve:
                # Move to finer scale
                if current_step_idx < len(base_steps) - 1:
                    current_step_idx += 1
                    scale_no_improve = 0
                    no_improve = 0
                else:
                    # Try random perturbation to escape local minimum
                    if no_improve > 500:
                        # Random restart-like perturbation
                        idx = np.random.randint(N)
                        old = centers[idx].copy()
                        noise = np.random.normal(0, 0.05, 2)
                        new_pos = centers[idx] + noise
                        if np.linalg.norm(new_pos) < 1.0:
                            centers[idx] = new_pos
                            update_dists_row(dists, centers, idx)
                            new_score = compute_score(centers, dists)
                            if new_score > current_score:
                                current_score = new_score
                                no_improve = 0
                                scale_no_improve = 0
                            else:
                                centers[idx] = old
                                update_dists_row(dists, centers, idx)
                                no_improve += 1
                        else:
                            no_improve += 1

            # Early termination if stuck at finest scale
            if current_step_idx == len(base_steps) - 1 and no_improve > 3000:
                break

        return best_centers, best_score

    def build_grid(centers, cell_size):
        """Build a spatial hash grid for O(1) neighbor lookups."""
        grid = {}
        for i, c in enumerate(centers):
            cx = int(np.floor(c[0] / cell_size))
            cy = int(np.floor(c[1] / cell_size))
            grid[(cx, cy)] = grid.get((cx, cy), [])
            grid[(cx, cy)].append(i)
        return grid

    def get_neighbors(grid, idx, centers, cell_size, search_radius):
        """Get neighbors within search_radius using grid."""
        c = centers[idx]
        cx = int(np.floor(c[0] / cell_size))
        cy = int(np.floor(c[1] / cell_size))
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                cell_key = (cx + dx, cy + dy)
                if cell_key in grid:
                    for j in grid[cell_key]:
                        if j != idx:
                            # Double check distance to be safe against cell size
                            dist_sq = np.sum((centers[idx] - centers[j])**2)
                            if dist_sq < search_radius**2:
                                neighbors.append(j)
        return neighbors

    def physics_relaxation(centers, time_budget):
        """
        Physics-inspired relaxation using spatial hashing for fast neighbor lookup.
        Uses a soft repulsive potential to maximize the minimum distance.
        """
        centers = centers.copy()
        current_score = compute_score(centers, compute_dists(centers))
        best_score = current_score
        best_centers = centers.copy()
        
        end_time = start_time + time_budget
        step_size = 0.02
        temperature = 0.01
        no_improve = 0
        
        # Estimate initial radius for search radius in grid
        # We want to find points that are "close" so we use a generous radius
        # Since max r ~ 0.125, diameter ~ 0.25. Search radius 0.4 is safe.
        search_radius = 0.4
        cell_size = search_radius / 2.0 # Ensures neighbors are in adjacent cells

        while time.time() < end_time:
            if time.time() > start_time + max_time:
                break

            # Build grid once per iteration (or update incrementally, but N=50 is small enough)
            grid = build_grid(centers, cell_size)
            
            # Calculate forces
            forces = np.zeros_like(centers)
            
            for i in range(N):
                neighbors = get_neighbors(grid, i, centers, cell_size, search_radius)
                for j in neighbors:
                    vec = centers[i] - centers[j]
                    dist_sq = np.sum(vec**2)
                    dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-12
                    
                    # Repulsive force: stronger when closer than target radius
                    # Target radius is roughly 0.125, so distance 0.25
                    # Use a smooth potential like 1/d^2 or exp(-d)
                    # Let's use a linear repulsion for simplicity and stability
                    if dist < 0.3: # Cutoff
                        # Force magnitude proportional to (cutoff - dist)
                        force_mag = (0.3 - dist) / 0.3
                        # Add some noise/jitter to escape local minima
                        jitter = np.random.normal(0, temperature, 2)
                        forces[i] += (vec / dist) * force_mag + jitter
            
            # Boundary force: push inward if close to boundary
            norms = np.linalg.norm(centers, axis=1)
            for i in range(N):
                if norms[i] > 0.8:
                    # Push towards center
                    force_mag = (norms[i] - 0.8) / 0.2
                    forces[i] -= (centers[i] / norms[i]) * force_mag

            # Move points
            # Adaptive step size
            for i in range(N):
                move = forces[i] * step_size
                new_pos = centers[i] + move
                
                # Clamp to boundary
                new_norm = np.linalg.norm(new_pos)
                if new_norm > 1.0:
                    new_pos = new_pos / new_norm * 0.999

                centers[i] = new_pos

            # Evaluate score
            new_score = compute_score(centers, compute_dists(centers))
            
            if new_score > best_score:
                best_score = new_score
                best_centers = centers.copy()
                no_improve = 0
            else:
                no_improve += 1
                # If stuck, increase temperature and step size slightly
                if no_improve > 50:
                    temperature = min(0.05, temperature * 1.1)
                    step_size = min(0.05, step_size * 1.05)
                    no_improve = 0
                elif no_improve > 20:
                    temperature = max(0.001, temperature * 0.9)
                    step_size = max(0.005, step_size * 0.95)

            # Decay temperature
            temperature *= 0.995

        return best_centers, best_score

    # Initialization: Multiple strategies
    def hex_init(spacing):
        k = int(1.0 / spacing) + 3
        points = []
        for j in range(-k, k+1):
            for i in range(-k, k+1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x*x + y*y <= 0.95:
                    points.append([x, y])
        points = np.array(points)
        if len(points) >= N:
            dists = np.linalg.norm(points, axis=1)
            sorted_indices = np.argsort(dists)
            return points[sorted_indices[:N]]
        return np.random.uniform(-0.8, 0.8, (N, 2))

    def random_init():
        # Random initialization with rejection sampling to avoid overlaps
        centers = np.zeros((N, 2))
        centers[0] = np.array([0.0, 0.0])
        min_r = 0.12
        for i in range(1, N):
            for attempt in range(1000):
                r = np.random.uniform(0, 0.9)
                theta = np.random.uniform(0, 2*np.pi)
                pos = np.array([r * np.cos(theta), r * np.sin(theta)])
                if np.all(np.linalg.norm(centers[:i] - pos, axis=1) > min_r * 2):
                    centers[i] = pos
                    break
            else:
                centers[i] = pos
        return centers

    # Try multiple initializations and keep the best
    inits = []
    inits.append(hex_init(0.22))
    inits.append(hex_init(0.20))
    inits.append(hex_init(0.25))
    inits.append(random_init())

    # Score each initialization
    best_init = None
    best_init_score = -1
    for init in inits:
        init_dists = compute_dists(init)
        init_score = compute_score(init, init_dists)
        if init_score > best_init_score:
            best_init_score = init_score
            best_init = init

    # Phase 1: Physics Relaxation for exploration and structure
    relax_time = 60.0
    relax_end = start_time + relax_time
    remaining_for_relax = min(relax_time, max_time - (time.time() - start_time))
    if remaining_for_relax > 5.0:
        best_centers, best_score = physics_relaxation(best_init, remaining_for_relax)
    else:
        best_centers = best_init
        best_score = best_init_score

    # Phase 2: Multi-scale local search for refinement
    remaining_time = max_time - (time.time() - start_time)
    if remaining_time > 5.0:
        step_sizes = [0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
        polished_centers, polished_score = local_search(best_centers, remaining_time - 2.0, step_sizes)
        if polished_score >= best_score:
            best_centers = polished_centers
            best_score = polished_score

    # Final safety check
    if best_centers is None or len(best_centers) != N:
        best_centers = np.random.uniform(-0.9, 0.9, (N, 2))

    # Ensure validity
    norms = np.linalg.norm(best_centers, axis=1)
    for i in range(N):
        if norms[i] > 1.0:
            best_centers[i] = best_centers[i] / norms[i] * 0.9999

    return best_centers
