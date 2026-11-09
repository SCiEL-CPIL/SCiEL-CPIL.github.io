# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def construct():
    deadline = time.time() + 140  # 10s safety margin before 150s kill

    # Improved Initialization: Greedy Max-Min selection from a dense grid
    # This approximates a dense packing (like FCC) better than a simple cubic grid
    grid_res = 10  # 10x10x10 = 1000 candidates
    ticks = (np.arange(grid_res) + 0.5) / grid_res - 0.5
    candidates = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # Greedy selection of N points to maximize minimum distance
    centers = np.empty((N, 3))
    remaining = candidates.copy()
    
    # First point: center of the cube (or closest to it)
    # Actually, to be symmetric, let's pick the center if it's in the grid, else closest.
    # With grid_res=10, center is not exactly in grid (0.5, 0.5, 0.5 vs 0.45, 0.55 etc).
    # Let's just pick the point with max min-distance to boundary (which is the center-most ones).
    # Or simply start with the point closest to (0,0,0) which is (0.45, 0.45, 0.45) ? No, that's corner-ish.
    # (0.45, 0.45, 0.45) is distance 0.05 from wall.
    # (0.45, 0.45, -0.45) etc.
    # The points closest to origin are the 8 corners of the central cell?
    # Let's just pick the candidate with the largest distance to the nearest wall.
    wall_dists = np.min(0.5 - np.abs(candidates), axis=1)
    idx_max = np.argmax(wall_dists)
    centers[0] = candidates[idx_max]
    remaining = np.delete(remaining, idx_max, axis=0)
    
    for i in range(1, N):
        if len(remaining) == 0:
            # Fallback if we run out of candidates (should not happen with 1000 candidates)
            centers[i] = centers[i-1] + np.random.randn(3) * 0.01
            continue
            
        # Calculate min distance from each remaining candidate to all existing centers
        # and to the boundary
        # To speed up, we can just compute distances to existing centers and wall dists
        
        # Wall distances for remaining candidates
        rem_wall = np.min(0.5 - np.abs(remaining), axis=1)
        
        # Distances to existing centers
        # existing: (i, 3), remaining: (M, 3)
        diffs = remaining[:, np.newaxis, :] - centers[:i, np.newaxis, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=2))
        min_pair_dists = np.min(dists, axis=1)
        
        # The "radius" for this candidate is min(rem_wall, min_pair_dists / 2)
        # We want to maximize this
        candidate_radii = np.minimum(rem_wall, min_pair_dists / 2.0)
        
        idx_best = np.argmax(candidate_radii)
        centers[i] = remaining[idx_best]
        remaining = np.delete(remaining, idx_best, axis=0)
        
        # Optional: Time check during init? Init is fast (100 * 1000 * 100 ops ~ 10^6, negligible)
        
    x0 = centers

    def radius(c):
        """Compute the minimum radius for configuration c."""
        c = c.reshape(N, 3)
        # Wall distances
        wall_dists = np.min(0.5 - np.abs(c), axis=1)
        # Pairwise distances
        diffs = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        return min(np.min(wall_dists), min_pair_dist / 2.0)

    def neg_radius(c):
        return -radius(c)

    def smooth_neg_radius(c, alpha=200.0):
        """Smooth surrogate using log-sum-exp to approximate -min radius."""
        c = c.reshape(N, 3)
        # Wall distances
        wall_dists = np.min(0.5 - np.abs(c), axis=1)
        # Pairwise distances
        diffs = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists_sq = np.sum(diffs ** 2, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        dists = np.sqrt(dists_sq)
        pair_radii = dists / 2.0
        
        # All constraints as negative values (we want to maximize the minimum)
        # For smooth min: -logsumexp(-x * alpha) / alpha
        all_vals = np.concatenate([wall_dists, pair_radii])
        # Clip to avoid numerical issues
        all_vals = np.maximum(all_vals, 1e-10)
        
        # Soft min approximation
        log_sum = np.log(np.sum(np.exp(-alpha * all_vals)))
        soft_min = -log_sum / alpha
        
        return soft_min

    best_x = x0.copy()
    best_r = radius(x0)

    # Local optimization with multiple restarts
    n_restarts = 0
    max_restarts = 10

    for restart in range(max_restarts):
        if time.time() > deadline:
            break

        if restart == 0:
            x_start = x0.copy()
            # First pass: use smooth objective for better gradient flow
            try:
                result = minimize(
                    smooth_neg_radius,
                    x_start,
                    method='L-BFGS-B',
                    bounds=[(-0.5, 0.5)] * (N * 3),
                    options={'maxiter': 300, 'maxls': 50, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                best_x = result.x
                best_r = radius(best_x)
            except Exception:
                pass
        else:
            # Perturb the best solution found so far
            x_start = best_x + np.random.randn(N, 3) * 0.02

        # Clip to stay within bounds
        x_start = np.clip(x_start, -0.49, 0.49)

        # Use exact objective for refinement
        try:
            result = minimize(
                neg_radius,
                x_start,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 300, 'maxls': 50, 'ftol': 1e-12, 'gtol': 1e-10}
            )

            r_result = -result.fun
            if r_result > best_r + 1e-12:
                best_r = r_result
                best_x = result.x
                n_restarts = 0  # reset counter on improvement
            else:
                n_restarts += 1

        except Exception:
            n_restarts += 1

        # Check deadline
        if time.time() > deadline:
            break

    # Final high-precision polish: use exact objective with tighter tolerances
    if time.time() < deadline - 5:
        try:
            x_start = np.clip(best_x, -0.49, 0.49)
            result = minimize(
                neg_radius,
                x_start,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 500, 'maxls': 50, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            r_result = -result.fun
            if r_result > best_r + 1e-15:
                best_r = r_result
                best_x = result.x
        except Exception:
            pass

    # Final validation and clipping
    best_x = np.clip(best_x, -0.5, 0.5)

    return best_x
# EVOLVE-BLOCK-END
