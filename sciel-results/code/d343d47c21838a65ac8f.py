"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint

N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint


def construct():
    """
    Place 50 equal circles inside unit circle to maximize common radius.
    Uses a multi-stage approach:
    1. Generate diverse initial configurations (rings, hexagonal, random).
    2. For each, run a local optimization (BFGS) on the soft-min objective.
    3. Return the best configuration found within time budget.
    """
    start_time = time.time()
    time_limit = 120.0  # seconds
    safety_margin = 5.0  # keep 5s before limit

    def elapsed():
        return time.time() - start_time

    def time_left():
        return time_limit - safety_margin - elapsed()

    # Objective: maximize r where r = min(min_{i<j} |c_i - c_j|/2, min_i (1 - |c_i|))
    # We minimize -r, or equivalently minimize a smooth surrogate.
    # Use log-sum-exp soft-min for smoothness.

    def soft_min(values, beta=50.0):
        """Approximate min via log-sum-exp."""
        m = np.min(values)
        return m - np.log(np.sum(np.exp(beta * (values - m)))) / beta

    def objective(x):
        """Minimize negative soft-min of constraints."""
        n = N
        x = x.reshape(n, 2)
        # Pairwise distances
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = dists.min() / 2.0

        # Distance to boundary
        radii = np.sqrt(np.sum(x**2, axis=1))
        min_boundary = 1.0 - radii.max()

        # Soft min of all constraints
        # Collect all constraint values
        vals = np.concatenate([dists / 2.0, 1.0 - radii])
        # Remove inf from diagonal
        vals = vals[np.isfinite(vals)]
        r = soft_min(vals, beta=80.0)
        return -r

    def run_optimization(x0, max_iter=3000, tol=1e-8):
        """Run BFGS optimization from x0."""
        try:
            res = minimize(objective, x0, method='BFGS',
                           options={'maxiter': max_iter, 'gtol': tol, 'disp': False})
            return res.x
        except Exception:
            return x0

    def compute_radius(x):
        """Compute the actual radius for a configuration."""
        x = x.reshape(N, 2)
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = dists.min() / 2.0
        radii = np.sqrt(np.sum(x**2, axis=1))
        min_boundary = 1.0 - radii.max()
        return min(min_pair, min_boundary)

    def is_valid(x):
        """Check if configuration is valid."""
        x = x.reshape(N, 2)
        if not np.all(np.isfinite(x)):
            return False
        radii = np.sqrt(np.sum(x**2, axis=1))
        if np.any(radii > 1.0):
            return False
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        if dists.min() < 1e-10:
            return False
        return True

    def generate_ring_config(num_rings, seed=42):
        """Generate a ring-based configuration."""
        rng = np.random.default_rng(seed)
        points = []
        # Center point
        points.append([0.0, 0.0])
        remaining = N - 1
        # Distribute remaining points in rings
        ring_sizes = []
        r = 1
        while remaining > 0:
            # Points per ring grows roughly as 2*pi*r
            size = min(remaining, max(1, int(2 * np.pi * r)))
            if size > remaining:
                size = remaining
            ring_sizes.append(size)
            remaining -= size
            r += 1

        for ring_idx, size in enumerate(ring_sizes):
            radius = (ring_idx + 1) * 0.15  # approximate spacing
            # Adjust radius to fit
            radius = min(radius, 0.95)
            angles = np.linspace(0, 2 * np.pi, size, endpoint=False)
            for a in angles:
                points.append([radius * np.cos(a), radius * np.sin(a)])

        # Trim or pad to N
        points = points[:N]
        while len(points) < N:
            points.append([0.0, 0.0])
        return np.array(points, dtype=float).flatten()

    def generate_hex_config(spacing, seed=42):
        """Generate hexagonal lattice configuration."""
        rng = np.random.default_rng(seed)
        points = []
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points_arr = np.array(rows)
        radius = np.hypot(points_arr[:, 0], points_arr[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points_arr = points_arr[mask]
        # Sort by radius and take first N
        radii = np.hypot(points_arr[:, 0], points_arr[:, 1])
        idx = np.argsort(radii)
        points_arr = points_arr[idx[:N]]
        return points_arr.flatten()

    def generate_random_config(seed=42):
        """Generate random configuration inside unit circle."""
        rng = np.random.default_rng(seed)
        points = []
        for _ in range(N):
            while True:
                r = np.sqrt(rng.uniform(0, 1))
                theta = rng.uniform(0, 2 * np.pi)
                x, y = r * np.cos(theta), r * np.sin(theta)
                points.append([x, y])
        return np.array(points, dtype=float).flatten()

    def generate_concentric_rings(seed=42):
        """Generate concentric ring configuration with optimized radii."""
        rng = np.random.default_rng(seed)
        # Try different ring distributions
        best_x = None
        best_r = -1
        for n_rings in range(1, 8):
            # Distribute N points into n_rings
            base = N // n_rings
            extra = N % n_rings
            ring_sizes = [base + (1 if i < extra else 0) for i in range(n_rings)]
            if ring_sizes[0] != 1:
                # Ensure first ring has 1 point (center)
                ring_sizes[0] = 1
                ring_sizes[1] += ring_sizes[0] - 1
                if ring_sizes[1] > N:
                    continue

            points = []
            for ring_idx, size in enumerate(ring_sizes):
                if ring_idx == 0 and size == 1:
                    points.append([0.0, 0.0])
                    continue
                # Radius for this ring
                radius = 0.3 + ring_idx * 0.12
                radius = min(radius, 0.9)
                angles = np.linspace(0, 2 * np.pi, size, endpoint=False) + rng.uniform(0, 2*np.pi)
                for a in angles:
                    points.append([radius * np.cos(a), radius * np.sin(a)])

            points = points[:N]
            while len(points) < N:
                points.append([0.0, 0.0])
            x = np.array(points, dtype=float).flatten()
            r = compute_radius(x)
            if r > best_r:
                best_r = r
                best_x = x
        return best_x

    # Generate diverse initial configurations
    configs = []

    # 1. Hexagonal with various spacings
    for spacing in [0.25, 0.28, 0.30, 0.32, 0.35]:
        x0 = generate_hex_config(spacing)
        if is_valid(x0):
            configs.append(x0)

    # 2. Ring-based
    for seed in range(10):
        x0 = generate_ring_config(3, seed=seed)
        if is_valid(x0):
            configs.append(x0)

    # 3. Concentric rings
    for seed in range(5):
        x0 = generate_concentric_rings(seed=seed)
        if x0 is not None and is_valid(x0):
            configs.append(x0)

    # 4. Random
    for seed in range(10):
        x0 = generate_random_config(seed=seed)
        if is_valid(x0):
            configs.append(x0)

    # Run optimization on each config
    best_x = None
    best_r = -1

    for i, x0 in enumerate(configs):
        if time_left() < 10.0:
            break

        # Scale down slightly to ensure feasibility
        x0_scaled = x0.reshape(N, 2) * 0.95
        x0_scaled = x0_scaled.flatten()

        x_opt = run_optimization(x0_scaled, max_iter=2000)

        if not is_valid(x_opt):
            continue

        r = compute_radius(x_opt)
        if r > best_r:
            best_r = r
            best_x = x_opt

    # If no valid result, use a simple fallback
    if best_x is None:
        x0 = generate_hex_config(0.30)
        best_x = x0
        best_r = compute_radius(best_x)

    # Final polish: run a few more iterations on the best solution
    if time_left() > 20.0:
        x_polished = run_optimization(best_x, max_iter=5000, tol=1e-10)
        if is_valid(x_polished):
            r_polished = compute_radius(x_polished)
            if r_polished > best_r:
                best_x = x_polished
                best_r = r_polished

    return best_x.reshape(N, 2)
# EVOLVE-BLOCK-END
