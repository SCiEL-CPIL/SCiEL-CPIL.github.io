# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog
from itertools import product


def squares_in_square10() -> tuple:
    """
    Erdos "squares in a square" problem. Place n squares with pairwise disjoint
    interiors inside the unit square [0,1]^2 so as to MAXIMIZE the sum of their
    side lengths. Here n = 10.

    Strategy:
    1. Enumerate all possible topological orderings (separation patterns) for
       pairs of squares. For each pair (i,j), either i is left of j, i is right
       of j, i is below j, or i is above j. This gives 4^(n*(n-1)/2) possible
       orderings, which is too many to enumerate fully.
    2. Instead, we use a MILP formulation with binary variables to encode the
       separation constraints, and solve using scipy's linprog with integer
       constraints (via branch and bound if available, or we use a heuristic
       approach).
    3. Since scipy.optimize.linprog doesn't support integer variables directly,
       we use a different approach: we enumerate a subset of likely optimal
       topological structures and solve LPs for each.

    Actually, let's use a simpler but effective approach:
    - Use scipy.optimize.milp if available (SciPy >= 1.7)
    - Otherwise, fall back to a structured search over topological patterns

    Returns:
        centers: np.ndarray of shape (10, 2) with the (x, y) center of each square.
        sides:   np.ndarray of shape (10,) with the side length of each square.
    """
    n = 10
    tol = 1e-7
    deadline = time.time() + 570  # safety margin before 630s kill

    # Grid baseline
    k = 4
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx0 = (cols + 0.5) * side
    cy0 = (rows + 0.5) * side
    s0 = np.full(n, side)
    best_cx = cx0.copy()
    best_cy = cy0.copy()
    best_s = s0.copy()
    best_obj = np.sum(s0)

    def check_feasible(cx, cy, s):
        """Check if configuration is feasible."""
        for i in range(n):
            if cx[i] - s[i]/2 < -tol or cx[i] + s[i]/2 > 1 + tol:
                return False
            if cy[i] - s[i]/2 < -tol or cy[i] + s[i]/2 > 1 + tol:
                return False
        for i in range(n):
            for j in range(i + 1, n):
                t = (s[i] + s[j]) / 2.0 - tol
                if t <= 0:
                    continue
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                if dx < t and dy < t:
                    return False
        return True

    # Try to use MILP if available
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        has_milp = True
    except ImportError:
        has_milp = False

    if has_milp:
        # MILP formulation
        # Variables: cx[0..n-1], cy[0..n-1], s[0..n-1], and binary vars for separation
        # For each pair (i,j), we need:
        #   cx_i + s_i/2 <= cx_j - s_j/2  (i left of j)
        #   cx_j + s_j/2 <= cx_i - s_i/2  (j left of i)
        #   cy_i + s_i/2 <= cy_j - s_j/2  (i below j)
        #   cy_j + s_j/2 <= cy_i - s_i/2  (j below i)
        # At least one of these must hold.
        # We use binary variables to select which separation applies.

        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        m_pairs = len(pairs)  # 45 pairs

        # Total variables: 3n continuous + 4*m_pairs binary = 30 + 180 = 210
        # This is too large for MILP in reasonable time. Let's use a heuristic instead.
        pass

    # Heuristic approach: try many random configurations and local improvements
    rng = np.random.default_rng(42)

    def local_improve(cx, cy, s):
        """Try to improve the configuration locally."""
        improved = True
        max_iter = 1000
        it = 0
        while improved and it < max_iter and time.time() < deadline - 60:
            improved = False
            it += 1
            # Try moving each square slightly
            for i in range(n):
                for dx in [-0.01, 0.01]:
                    for dy in [-0.01, 0.01]:
                        cx_new = cx.copy()
                        cy_new = cy.copy()
                        cx_new[i] += dx
                        cy_new[i] += dy
                        if check_feasible(cx_new, cy_new, s):
                            # Check if we can increase any side
                            for j in range(n):
                                s_try = s.copy()
                                # Try increasing side j
                                delta = 0.005
                                s_try[j] += delta
                                if check_feasible(cx_new, cy_new, s_try):
                                    cx, cy, s = cx_new, cy_new, s_try
                                    improved = True
                                    break
            # Try increasing sides
            for i in range(n):
                delta = 0.005
                s_new = s.copy()
                s_new[i] += delta
                if check_feasible(cx, cy, s_new):
                    s = s_new
                    improved = True

        return cx, cy, s

    # Generate diverse initial configurations
    n_attempts = 50
    for attempt in range(n_attempts):
        if time.time() > deadline - 60:
            break

        if attempt == 0:
            # Start with grid
            cx, cy, s = cx0.copy(), cy0.copy(), s0.copy()
        elif attempt == 1:
            # Try 2x5 grid
            kx, ky = 2, 5
            side = 1.0 / max(kx, ky)
            idx = np.arange(n)
            cols = idx % kx
            rows = idx // kx
            cx = (cols + 0.5) * (1.0 / kx)
            cy = (rows + 0.5) * (1.0 / ky)
            s = np.full(n, min(1.0/kx, 1.0/ky))
        elif attempt == 2:
            # Try 5x2 grid
            kx, ky = 5, 2
            idx = np.arange(n)
            cols = idx % kx
            rows = idx // kx
            cx = (cols + 0.5) * (1.0 / kx)
            cy = (rows + 0.5) * (1.0 / ky)
            s = np.full(n, min(1.0/kx, 1.0/ky))
        else:
            # Random configuration with small squares
            cx = rng.uniform(0.1, 0.9, n)
            cy = rng.uniform(0.1, 0.9, n)
            s = rng.uniform(0.1, 0.3, n)
            # Clip to ensure boundary feasibility
            for i in range(n):
                s[i] = min(s[i], 0.4)
                cx[i] = np.clip(cx[i], s[i]/2 + tol, 1 - s[i]/2 - tol)
                cy[i] = np.clip(cy[i], s[i]/2 + tol, 1 - s[i]/2 - tol)

        if not check_feasible(cx, cy, s):
            # Try to fix by reducing sides
            for i in range(n):
                while not check_feasible(cx, cy, s) and s[i] > 0.05:
                    s[i] *= 0.9
            if not check_feasible(cx, cy, s):
                continue

        # Local improvement
        cx, cy, s = local_improve(cx, cy, s)

        obj = np.sum(s)
        if obj > best_obj:
            best_obj = obj
            best_cx = cx.copy()
            best_cy = cy.copy()
            best_s = s.copy()

    # Final cleanup
    best_s = np.maximum(best_s, 1e-8)
    for i in range(n):
        best_cx[i] = np.clip(best_cx[i], best_s[i]/2 - tol, 1.0 - best_s[i]/2 + tol)
        best_cy[i] = np.clip(best_cy[i], best_s[i]/2 - tol, 1.0 - best_s[i]/2 + tol)

    centers = np.stack([best_cx, best_cy], axis=1).astype(np.float64)
    sides = best_s.astype(np.float64)

    return centers, sides


# EVOLVE-BLOCK-END
