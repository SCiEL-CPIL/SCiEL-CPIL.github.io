"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill

    def compute_radius(centers):
        """Compute the max common radius for given centers."""
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        dist_to_boundary = np.min(0.5 - np.abs(centers), axis=1)
        min_boundary_dist = np.min(dist_to_boundary)
        r = min(min_pair_dist / 2.0, min_boundary_dist)
        return r

    def force_directed_step(centers, r_target, dt, temp=0.0):
        """One step of force-directed relaxation.
        Forces: repulsion between circles (LJ-like), repulsion from walls.
        """
        n = len(centers)
        forces = np.zeros_like(centers)

        # Pairwise repulsion: push apart if closer than 2*r_target
        for i in range(n):
            for j in range(i + 1, n):
                d = centers[i] - centers[j]
                dist = np.sqrt(np.sum(d**2))
                if dist < 1e-12:
                    # Random nudge
                    d = np.array([np.random.randn(), np.random.randn()])
                    dist = np.sqrt(np.sum(d**2))
                # LJ-like: strong repulsion when close
                if dist < 2.5 * r_target:
                    # Force magnitude ~ 1/dist^2 (stronger than linear)
                    f_mag = 1.0 / (dist * dist + 1e-10)
                    forces[i] += f_mag * d / dist
                    forces[j] -= f_mag * d / dist

        # Wall repulsion
        for i in range(n):
            for axis in range(2):
                c = centers[i, axis]
                # Distance to nearest wall
                d_left = c + 0.5
                d_right = 0.5 - c
                if d_left < r_target:
                    f = (r_target - d_left) ** 2 * 10.0
                    forces[i, axis] += f  # push right (positive x)
                if d_right < r_target:
                    f = (r_target - d_right) ** 2 * 10.0
                    forces[i, axis] -= f  # push left (negative x)

        # Add thermal noise for annealing
        if temp > 0:
            forces += temp * np.random.randn(n, 2)

        # Update positions with velocity damping
        new_centers = centers + dt * forces
        # Keep within bounds (soft constraint - just clip)
        new_centers = np.clip(new_centers, -0.4999, 0.4999)
        return new_centers

    def relax(centers, r_target, max_iters=2000, dt_init=0.001, dt_decay=0.999):
        """Iterative force-directed relaxation to find a valid packing at r_target."""
        dt = dt_init
        prev_r = -np.inf
        stall_count = 0
        for it in range(max_iters):
            centers = force_directed_step(centers, r_target, dt)
            r = compute_radius(centers)
            if r > r_target + 1e-10:
                return centers, r
            if r > prev_r + 1e-12:
                stall_count = 0
            else:
                stall_count += 1
            prev_r = max(prev_r, r)
            if stall_count > 100:
                break
            dt *= dt_decay
        return centers, compute_radius(centers)

    def anneal_and_relax(centers_init, r_target, max_outer=50, max_inner=1000):
        """Simulated annealing: try to reach r_target with thermal restarts."""
        centers = centers_init.copy()
        best_centers = centers.copy()
        best_r = compute_radius(centers)
        for outer in range(max_outer):
            temp = 0.005 * (0.95 ** outer)
            for inner in range(max_inner):
                centers = force_directed_step(centers, r_target, 0.001, temp)
                r = compute_radius(centers)
                if r > best_r:
                    best_r = r
                    best_centers = centers.copy()
                if r > r_target + 1e-10:
                    return best_centers, best_r
            if time.time() > deadline - 10:
                break
        return best_centers, best_r

    # --- Initialization: 7x7 grid + guest ---
    ticks7 = (np.arange(7) + 0.5) / 7 - 0.5
    grid_7x7 = np.array([(x, y) for y in ticks7 for x in ticks7])
    guest_center = np.array([[0.0, 0.0]])
    init_base = np.vstack([grid_7x7, guest_center])

    # Also try 5x10 grid
    ticks_x10 = (np.arange(10) + 0.5) / 10 - 0.5
    ticks_y5 = (np.arange(5) + 0.5) / 5 - 0.5
    grid_5x10 = np.array([(x, y) for y in ticks_y5 for x in ticks_x10])

    # Evaluate initial configs
    r_7x7 = compute_radius(init_base)
    r_5x10 = compute_radius(grid_5x10)

    best_r = max(r_7x7, r_5x10)
    best_centers = init_base.copy() if r_7x7 >= r_5x10 else grid_5x10.copy()

    # Binary search on radius using force-directed relaxation
    # Start from current best and try to improve
    r_low = best_r
    r_high = 0.075  # upper bound (known best is ~0.07137)

    # First, do a quick relaxation from the best init to see if we can improve
    np.random.seed(42)
    relaxed, r_relaxed = relax(best_centers, best_r, max_iters=3000, dt_init=0.002, dt_decay=0.9995)
    if r_relaxed > best_r:
        best_r = r_relaxed
        best_centers = relaxed

    # Try annealing from best config
    if time.time() < deadline - 30:
        annealed, r_annealed = anneal_and_relax(best_centers, best_r, max_outer=30, max_inner=500)
        if r_annealed > best_r:
            best_r = r_annealed
            best_centers = annealed

    # Binary search: try to find larger r
    for search_iter in range(15):
        if time.time() > deadline - 15:
            break

        r_mid = (r_low + r_high) / 2.0
        if r_mid <= r_low + 1e-12:
            break

        # Try to find a valid packing at r_mid
        # Use multiple random restarts
        found = False
        for restart in range(5):
            if time.time() > deadline - 10:
                break

            if restart == 0:
                init = best_centers.copy()
            else:
                # Jitter the best config
                init = best_centers + 0.01 * np.random.randn(N, 2)
                init = np.clip(init, -0.4999, 0.4999)

            # Quick relaxation
            result, r_result = relax(init, r_mid, max_iters=1500, dt_init=0.001, dt_decay=0.999)
            if r_result >= r_mid - 1e-9:
                found = True
                if r_result > best_r:
                    best_r = r_result
                    best_centers = result
                r_low = r_mid
                break

            # If not found, try annealing
            if time.time() < deadline - 15:
                annealed, r_annealed = anneal_and_relax(init, r_mid, max_outer=10, max_inner=300)
                if r_annealed >= r_mid - 1e-9:
                    found = True
                    if r_annealed > best_r:
                        best_r = r_annealed
                        best_centers = annealed
                    r_low = r_mid
                    break

        if not found:
            r_high = r_mid

        # Convergence check
        if r_high - r_low < 1e-7:
            break

    # Final polish: relax at the best found radius
    if time.time() < deadline - 5:
        polished, r_polished = relax(best_centers, best_r, max_iters=5000, dt_init=0.0005, dt_decay=0.9998)
        if r_polished > best_r:
            best_r = r_polished
            best_centers = polished

    # Safety: ensure all centers are finite and within bounds
    if best_centers is None:
        return grid_5x10

    best_centers = np.clip(best_centers, -0.499999, 0.499999)

    return best_centers
# EVOLVE-BLOCK-END
