# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def squares_in_square10() -> tuple:
    """
    Erdos "squares in a square" problem. Place n squares with pairwise disjoint
    interiors inside the unit square [0,1]^2 so as to MAXIMIZE the sum of their
    side lengths. Here n = 10.

    Uses a smooth repulsion-potential approach: each square's side length is
    derived from local clearance to neighbors and boundaries, and the optimizer
    minimizes a smooth energy that penalizes overlaps while rewarding large sides.

    Returns:
        centers: np.ndarray of shape (10, 2) with the (x, y) center of each square.
        sides:   np.ndarray of shape (10,) with the side length of each square.
    The i-th axis-aligned square occupies
        [cx - s/2, cx + s/2] x [cy - s/2, cy + s/2].
    """
    n = 10
    tol = 1e-7

    def smooth_max(a, b, k=100.0):
        """Smooth approximation of max(a, b) using log-sum-exp."""
        m = np.maximum(a, b)
        return m + np.logaddexp(k*(a - m), k*(b - m)) / k

    def compute_smooth_sides(centers):
        """
        For each square i, compute a smooth estimate of the maximum side length
        based on clearance to boundaries and other squares.
        Uses soft constraints so the resulting function is differentiable.
        """
        sides = np.zeros(n)
        for i in range(n):
            cx, cy = centers[i]
            # Clearance to boundaries
            d_left = cx
            d_right = 1.0 - cx
            d_bottom = cy
            d_top = 1.0 - cy
            # Max side from boundary constraints: s <= 2 * min(d_left, d_right, d_bottom, d_top)
            # Use smooth min for differentiability
            min_boundary = d_left
            min_boundary = -smooth_max(-min_boundary, -d_right)
            min_boundary = -smooth_max(-min_boundary, -d_bottom)
            min_boundary = -smooth_max(-min_boundary, -d_top)
            s_boundary = 2.0 * min_boundary

            s_max = s_boundary
            for j in range(n):
                if i == j:
                    continue
                dx = abs(cx - centers[j][0])
                dy = abs(cy - centers[j][1])
                # For non-overlap, we need: dx >= (s_i + s_j)/2 OR dy >= (s_i + s_j)/2
                # The clearance in x gives: s_i <= 2*dx - s_j
                # The clearance in y gives: s_i <= 2*dy - s_j
                # So s_i <= min(2*dx - s_j, 2*dy - s_j, s_boundary)
                # We use a smooth version and assume s_j is roughly similar to s_i initially
                # For the potential approach, we treat this as a repulsion:
                # The "effective" max side contribution from neighbor j is limited by the gap
                gap_x = 2.0 * dx
                gap_y = 2.0 * dy
                # The constraint is s_i + s_j <= gap_x OR s_i + s_j <= gap_y
                # Smoothly: s_i <= max(gap_x - s_j, gap_y - s_j)
                # Since we don't know s_j, we use an iterative estimate or just use the gap
                # For the potential field approach, we compute a "local density" based on proximity
                # and derive side from that. Here we use a simpler coupled estimate.
                # Assume s_j ~ s_i for the constraint (fixed-point style)
                # s_i <= gap_x - s_j and s_i <= gap_y - s_j
                # => 2*s_i <= min(gap_x, gap_y) => s_i <= min(gap_x, gap_y) / 2
                # But this is conservative. Let's use: s_i + s_j <= max(gap_x, gap_y)
                # which allows separation in either axis.
                # Smooth max of the two gaps:
                gap_eff = smooth_max(gap_x, gap_y)
                # s_i + s_j <= gap_eff => s_i <= gap_eff - s_j
                # For a self-consistent estimate, assume s_j ≈ current best guess
                # We'll use s_j = 0 as a lower bound for the constraint (optimistic)
                # and let the optimizer figure out the actual sizes.
                # Actually, for the potential approach, let's just compute clearance-based sides
                # and let the penalty handle the coupling.
                # For now, use the gap as an upper bound:
                s_from_j = gap_eff
                s_max = min(s_max, s_from_j)
            sides[i] = max(s_max, 0.01)
        return sides

    def potential_energy(x):
        """
        Compute the total energy: -sum(sides) + overlap_penalty + boundary_penalty.
        The overlap penalty uses a smooth repulsive potential.
        """
        centers = x[:2*n].reshape(n, 2)
        sides = x[2*n:]
        sides_clipped = np.clip(sides, 1e-10, 1.0)

        # Reward: maximize sum of sides
        reward = np.sum(sides_clipped)

        # Boundary penalty: squares must stay within [0,1]^2
        boundary_penalty = 0.0
        for i in range(n):
            cx, cy = centers[i]
            s = sides_clipped[i]
            # Soft penalty for going out of bounds
            # Use exponential penalty for smoothness
            left_viol = max(0, -(cx - s/2))
            right_viol = max(0, (cx + s/2) - 1.0)
            bottom_viol = max(0, -(cy - s/2))
            top_viol = max(0, (cy + s/2) - 1.0)
            boundary_penalty += (left_viol**2 + right_viol**2 + bottom_viol**2 + top_viol**2)

        # Overlap penalty: smooth repulsive potential between pairs
        overlap_penalty = 0.0
        for i in range(n):
            for j in range(i+1, n):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                abs_dx = abs(dx)
                abs_dy = abs(dy)
                s_i = sides_clipped[i]
                s_j = sides_clipped[j]
                t = (s_i + s_j) / 2.0

                # Overlap occurs when abs_dx < t AND abs_dy < t
                # Use a smooth potential: penalize when both are violated
                # Potential: V = max(0, t - abs_dx) * max(0, t - abs_dy)
                # This is zero when separated, positive when overlapping
                # For smoothness, use softplus approximation
                px = np.logaddexp(0, 100*(t - abs_dx)) / 100  # smooth max(0, t - abs_dx)
                py = np.logaddexp(0, 100*(t - abs_dy)) / 100  # smooth max(0, t - abs_dy)
                overlap_penalty += px * py

        # Total energy to minimize
        energy = -reward + 500.0 * boundary_penalty + 500.0 * overlap_penalty
        return energy

    def potential_energy_grad(x):
        """
        Numerical gradient of the potential energy.
        """
        eps = 1e-7
        grad = np.zeros_like(x)
        for k in range(len(x)):
            x_plus = x.copy()
            x_plus[k] += eps
            x_minus = x.copy()
            x_minus[k] -= eps
            grad[k] = (potential_energy(x_plus) - potential_energy(x_minus)) / (2 * eps)
        return grad

    # Initial guess: grid packing with some randomness to escape local minima
    best_centers = None
    best_sides = None
    best_value = -np.inf

    np.random.seed(42)

    # Try multiple initial configurations
    initial_configs = []

    # Config 1: 3x3 grid + 1 extra (4x3 grid missing one corner)
    k = 3
    side_init = 1.0 / k
    idx = np.arange(n)
    cols = idx % 4
    rows = idx // 4
    # Place 10 squares in a 4x3 arrangement (last row has 2)
    cx_init = np.zeros(n)
    cy_init = np.zeros(n)
    for i in range(n):
        col = i % 4
        row = i // 4
        if row < 3:
            cx_init[i] = (col + 0.5) * (1.0/4)
            cy_init[i] = (row + 0.5) * (1.0/3)
        else:
            # Last two squares
            cx_init[i] = (col + 0.5) * (1.0/4)
            cy_init[i] = 0.5 * (1.0/3)
    sides_init = np.full(n, 0.2)
    initial_configs.append(np.concatenate([np.stack([cx_init, cy_init], axis=1).flatten(), sides_init]))

    # Config 2: Random perturbation of grid
    base_side = 1.0 / 3.5
    cx_rand = np.random.uniform(0.1, 0.9, n)
    cy_rand = np.random.uniform(0.1, 0.9, n)
    sides_rand = np.full(n, base_side)
    initial_configs.append(np.concatenate([np.stack([cx_rand, cy_rand], axis=1).flatten(), sides_rand]))

    # Config 3: Two rows of 5
    cx_row = np.zeros(n)
    cy_row = np.zeros(n)
    for i in range(n):
        if i < 5:
            cx_row[i] = (i + 0.5) * 0.2
            cy_row[i] = 0.25
        else:
            cx_row[i] = (i - 5 + 0.5) * 0.2
            cy_row[i] = 0.75
    sides_row = np.full(n, 0.18)
    initial_configs.append(np.concatenate([np.stack([cx_row, cy_row], axis=1).flatten(), sides_row]))

    # Config 4: Larger squares with some small ones
    cx_mix = np.array([0.25, 0.75, 0.25, 0.75, 0.5, 0.25, 0.75, 0.25, 0.75, 0.5])
    cy_mix = np.array([0.25, 0.25, 0.75, 0.75, 0.5, 0.5, 0.5, 0.1, 0.9, 0.9])
    sides_mix = np.array([0.3, 0.3, 0.3, 0.3, 0.1, 0.15, 0.15, 0.1, 0.1, 0.1])
    initial_configs.append(np.concatenate([np.stack([cx_mix, cy_mix], axis=1).flatten(), sides_mix]))

    bounds = []
    for i in range(n):
        bounds.append((0.01, 0.99))
        bounds.append((0.01, 0.99))
        bounds.append((0.01, 0.99))

    for x0 in initial_configs:
        # Clip initial values to bounds
        for b_idx, (lo, hi) in enumerate(bounds):
            x0[b_idx] = np.clip(x0[b_idx], lo, hi)

        result = minimize(potential_energy, x0, jac=potential_energy_grad,
                          method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-6, 'maxls': 50})

        x_opt = result.x
        centers = x_opt[:2*n].reshape(n, 2)
        sides = x_opt[2*n:]
        sides = np.clip(sides, 1e-10, 1.0)

        # Evaluate the actual objective (sum of sides) with validity check
        value = np.sum(sides)
        
        # Check validity
        valid = True
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            if cx - s/2 < -tol or cx + s/2 > 1 + tol or cy - s/2 < -tol or cy + s/2 > 1 + tol:
                valid = False
                break
            for j in range(i+1, n):
                dx = abs(cx - centers[j][0])
                dy = abs(cy - centers[j][1])
                t = (s + sides[j]) / 2 - tol
                if dx < t and dy < t:
                    valid = False
                    break
            if not valid:
                break

        if valid and value > best_value:
            best_value = value
            best_centers = centers.copy()
            best_sides = sides.copy()

    # If no valid solution found, use a safe grid fallback
    if best_centers is None:
        k = 4
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        best_centers = np.stack([cx, cy], axis=1).astype(np.float64)
        best_sides = np.full(n, side, dtype=np.float64)

    return best_centers.astype(np.float64), best_sides.astype(np.float64)


# EVOLVE-BLOCK-END
