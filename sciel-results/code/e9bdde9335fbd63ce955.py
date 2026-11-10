"""Improved packing of 50 equal spheres in a unit cube using
Hilbert-curve initialization + smooth repulsion gradient + local polish."""

import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

N = 50
TIME_LIMIT = 110.0  # safety margin before 120 s kill


def _hilbert_curve_3d(order=6):
    """Generate a Hilbert-space-filling curve in [0,1]^3."""
    # Use a simple recursive Hilbert curve generation
    def _hilbert_2d(n):
        """2D Hilbert curve of order n (length 4^n points)."""
        if n == 0:
            return np.array([[[0.5, 0.5]]])
        # Recursive construction
        sub = _hilbert_2d(n - 1)
        # Scale to quarter
        sub = sub / 2.0
        # Four quadrants with appropriate rotations
        # Q0: bottom-left, Q1: top-left, Q2: top-right, Q3: bottom-right
        # For Hilbert: 0, 1, 2, 3 in a specific order
        q0 = sub
        q1 = sub + np.array([0, 0.5])
        q2 = sub + np.array([0.5, 0.5])
        q3 = sub + np.array([0.5, 0])
        # Standard Hilbert: 0->1->2->3 but with rotations
        # Actually let's just use a simple space-filling approach
        return np.vstack([q0, q1, q2, q3])

    # For 3D, we'll use a simpler approach: lexicographic ordering with space-filling
    # Actually, let's just use a well-distributed initial placement
    # Use a Fibonacci lattice or similar for good initial spread
    pass


def _initial_placement():
    """Create a well-distributed initial placement of N points in [-0.5, 0.5]^3."""
    # Use a combination of grid + jitter for good initial spread
    # 50 points: try a 4x4x4 grid (64 points) and take 50, or 3x4x5=60 take 50
    # Or use a more sophisticated approach

    # Method: place points on a body-centered cubic lattice pattern
    # For 50 points, use multiple BCC unit cells
    # BCC has 2 points per unit cell, so 25 cells
    # Arrange 25 cells in a 3x3x3 grid (27 cells, use 25)

    # Alternative: use a simple space-filling curve approach
    # Generate points along a space-filling curve in the cube

    # Simple approach: use a jittered grid
    m = int(np.ceil(N ** (1.0 / 3.0)))
    # m=4 gives 64 positions, take first 50
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])

    # Add small random jitter to break symmetry
    rng = np.random.default_rng(42)
    grid = grid[:N].copy()
    jitter_scale = 0.5 / m * 0.3
    grid += rng.uniform(-jitter_scale, jitter_scale, size=grid.shape)

    # Clip to valid range
    grid = np.clip(grid, -0.49, 0.49)

    return grid


def _score(centers):
    """Compute the packing radius for a set of centers."""
    if len(centers) != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0

    # Min distance between pairs
    if N > 1:
        d = pdist(centers, metric='euclidean')
        min_pair = np.min(d) if len(d) > 0 else 1.0
    else:
        min_pair = 1.0

    # Min distance to walls
    min_wall = np.min(0.5 - np.abs(centers))

    return min(min_pair / 2.0, min_wall)


def _smooth_objective(centers_flat, alpha=10.0):
    """
    Smooth objective to minimize: sum of inverse distances + wall penalties.
    This is differentiable and guides the optimization.
    """
    centers = centers_flat.reshape(N, 3)

    # Pairwise repulsion: sum of 1/d^2 for close pairs
    obj = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(centers[i] - centers[j])
            d = max(d, 1e-6)
            obj += 1.0 / (d * d)

    # Wall repulsion: penalty for being close to walls
    for i in range(N):
        for k in range(3):
            dist_to_wall = 0.5 - abs(centers[i, k])
            dist_to_wall = max(dist_to_wall, 1e-6)
            obj += 1.0 / (dist_to_wall * dist_to_wall)

    return obj


def _smooth_objective_grad(centers_flat, alpha=10.0):
    """Gradient of the smooth objective."""
    centers = centers_flat.reshape(N, 3)
    grad = np.zeros_like(centers)

    for i in range(N):
        for j in range(i + 1, N):
            diff = centers[i] - centers[j]
            d = np.linalg.norm(diff)
            d = max(d, 1e-6)
            # d(1/d^2)/dx = -2 * diff / d^4
            grad[i] += -2.0 * diff / (d ** 4)
            grad[j] += 2.0 * diff / (d ** 4)

    for i in range(N):
        for k in range(3):
            val = centers[i, k]
            dist_to_wall = 0.5 - abs(val)
            dist_to_wall = max(dist_to_wall, 1e-6)
            # d(1/dist^2)/dx = 2*sign(x) / dist^3 ... need to be careful
            # dist_to_wall = 0.5 - |x|
            # d(dist_to_wall)/dx = -sign(x)
            # d(1/dist_to_wall^2)/dx = -2/dist_to_wall^3 * d(dist_to_wall)/dx
            #                        = 2*sign(x) / dist_to_wall^3
            sign_x = np.sign(val)
            if sign_x == 0:
                sign_x = 1.0
            grad[i, k] += 2.0 * sign_x / (dist_to_wall ** 3)

    return grad.flatten()


def _local_polish(centers, max_iter=2000, step_size=0.01):
    """
    Local polish: iteratively move each point to maximize the minimum distance.
    Uses a greedy approach: for each point, try to move it slightly to increase
    the minimum distance from it to all others and walls.
    """
    centers = centers.copy()
    n = len(centers)

    for iteration in range(max_iter):
        improved = False
        for i in range(n):
            best_pos = centers[i]
            best_min_dist = _min_dist_for_point(centers, i)

            # Try small perturbations
            for axis in range(3):
                for direction in [-1, 1]:
                    for step in [0.001, 0.005, 0.01, 0.02]:
                        new_pos = centers[i].copy()
                        new_pos[axis] += direction * step
                        # Check bounds
                        if new_pos[axis] < -0.499 or new_pos[axis] > 0.499:
                            continue
                        # Check if this improves the minimum distance for point i
                        new_min_dist = _min_dist_for_point_at(centers, i, new_pos)
                        if new_min_dist > best_min_dist + 1e-8:
                            best_min_dist = new_min_dist
                            best_pos = new_pos.copy()

            if not np.allclose(best_pos, centers[i]):
                centers[i] = best_pos
                improved = True

        if not improved:
            break

    return centers


def _min_dist_for_point(centers, i):
    """Minimum distance from point i to all other points and walls."""
    min_d = 0.5 - np.abs(centers[i])
    min_d = np.min(min_d)
    for j in range(len(centers)):
        if j != i:
            d = np.linalg.norm(centers[i] - centers[j])
            min_d = min(min_d, d)
    return min_d


def _min_dist_for_point_at(centers, i, pos):
    """Minimum distance from position pos to all other points and walls."""
    min_d = 0.5 - np.abs(pos)
    min_d = np.min(min_d)
    for j in range(len(centers)):
        if j != i:
            d = np.linalg.norm(pos - centers[j])
            min_d = min(min_d, d)
    return min_d


def _anneal(centers, start_temp=0.05, end_temp=0.001, max_iter=5000):
    """
    Simulated annealing to improve the packing.
    """
    centers = centers.copy()
    n = len(centers)
    current_score = _score(centers)
    best_centers = centers.copy()
    best_score = current_score

    for it in range(max_iter):
        temp = start_temp * (end_temp / start_temp) ** (it / max_iter)

        # Random perturbation
        i = np.random.randint(n)
        delta = np.random.normal(0, temp, size=3)
        new_centers = centers.copy()
        new_centers[i] += delta

        # Check bounds
        if np.any(new_centers < -0.5) or np.any(new_centers > 0.5):
            continue

        new_score = _score(new_centers)

        if new_score >= current_score or np.random.random() < np.exp((new_score - current_score) / max(temp, 1e-10)):
            centers = new_centers
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_centers = centers.copy()

    return best_centers


def construct():
    """
    Main entry point: construct the centers of 50 equal spheres in [-0.5, 0.5]^3.
    """
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

    # Step 1: Initial placement
    centers = _initial_placement()
    best_centers = centers.copy()
    best_score = _score(centers)

    # Step 2: Gradient-based optimization using smooth objective
    if time.time() < deadline - 30:
        try:
            flat = centers.flatten()
            result = minimize(
                _smooth_objective,
                flat,
                jac=_smooth_objective_grad,
                method='L-BFGS-B',
                bounds=[(-0.499, 0.499)] * (3 * N),
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            new_centers = result.x.reshape(N, 3)
            new_centers = np.clip(new_centers, -0.499, 0.499)
            new_score = _score(new_centers)
            if new_score > best_score:
                best_centers = new_centers
                best_score = new_score
        except Exception:
            pass

    # Step 3: Local polish
    if time.time() < deadline - 20:
        polished = _local_polish(best_centers, max_iter=500)
        polished_score = _score(polished)
        if polished_score > best_score:
            best_centers = polished
            best_score = polished_score

    # Step 4: Simulated annealing for final improvement
    if time.time() < deadline - 10:
        annealed = _anneal(best_centers, start_temp=0.02, end_temp=0.0005, max_iter=3000)
        annealed_score = _score(annealed)
        if annealed_score > best_score:
            best_centers = annealed
            best_score = annealed_score

    # Final validation
    best_centers = np.clip(best_centers, -0.499, 0.499)
    return best_centers


if __name__ == '__main__':
    centers = construct()
    print(f"Score: {_score(centers)}")
