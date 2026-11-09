# EVOLVE-BLOCK-START
import numpy as np
import time


def squares_in_square10() -> tuple:
    """
    Erdos "squares in a square" problem. Place n squares with pairwise disjoint
    interiors inside the unit square [0,1]^2 so as to MAXIMIZE the sum of their
    side lengths. Here n = 10.

    Returns:
        centers: np.ndarray of shape (10, 2) with the (x, y) center of each square.
        sides:   np.ndarray of shape (10,) with the side length of each square.
    The i-th axis-aligned square occupies
        [cx - s/2, cx + s/2] x [cy - s/2, cy + s/2].
    """
    n = 10
    deadline = time.time() + 580  # safety margin before 600s limit

    def _is_feasible(centers, sides, tol=1e-7):
        """Check if the configuration is feasible."""
        if centers.shape != (n, 2) or sides.shape != (n,):
            return False
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
            return False
        if np.any(sides <= 0):
            return False
        # Boundary check
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            if cx - s/2 < -tol or cx + s/2 > 1 + tol:
                return False
            if cy - s/2 < -tol or cy + s/2 > 1 + tol:
                return False
        # Overlap check
        for i in range(n):
            for j in range(i+1, n):
                t = (sides[i] + sides[j]) / 2 - tol
                if abs(centers[i,0] - centers[j,0]) < t and abs(centers[i,1] - centers[j,1]) < t:
                    return False
        return True

    def _sum_sides(sides):
        return np.sum(sides)

    def _objective(centers, sides):
        """Maximize sum of sides, penalize infeasibility."""
        if _is_feasible(centers, sides):
            return _sum_sides(sides)
        return -1e6

    # Start with the baseline grid packing
    k = int(np.ceil(np.sqrt(n)))
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    centers = np.stack([cx, cy], axis=1).astype(np.float64)
    sides = np.full(n, side, dtype=np.float64)
    
    best_centers = centers.copy()
    best_sides = sides.copy()
    best_obj = _objective(best_centers, best_sides)

    # Local search: perturb centers and sides, accept if better
    rng = np.random.default_rng(42)
    step_size = 0.05
    max_iters = 50000
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Random perturbation
        centers_p = centers + rng.normal(0, step_size, (n, 2))
        sides_p = sides + rng.normal(0, step_size * 0.5, n)
        sides_p = np.maximum(sides_p, 0.01)  # keep sides positive
        
        # Simple repair: clip to boundaries
        for i in range(n):
            s = sides_p[i]
            centers_p[i, 0] = np.clip(centers_p[i, 0], s/2, 1 - s/2)
            centers_p[i, 1] = np.clip(centers_p[i, 1], s/2, 1 - s/2)
        
        obj_p = _objective(centers_p, sides_p)
        if obj_p > best_obj:
            best_centers = centers_p.copy()
            best_sides = sides_p.copy()
            best_obj = obj_p
            # Reduce step size on improvement
            step_size *= 0.95
        else:
            # Occasionally accept worse solutions to escape local optima
            if rng.random() < 0.1:
                centers = centers_p.copy()
                sides = sides_p.copy()
                step_size *= 1.05
            else:
                # Keep current
                pass
        
        # Update current position occasionally
        if it % 100 == 0:
            centers = best_centers.copy()
            sides = best_sides.copy()
    
    # Final high-precision polish
    # Try to tighten the solution
    for _ in range(1000):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            # Try small adjustments to each square
            for dx in [-0.001, 0, 0.001]:
                for dy in [-0.001, 0, 0.001]:
                    for ds in [-0.001, 0, 0.001]:
                        c_new = best_centers.copy()
                        s_new = best_sides.copy()
                        c_new[i, 0] += dx
                        c_new[i, 1] += dy
                        s_new[i] = max(s_new[i] + ds, 0.01)
                        # Clip to boundaries
                        s = s_new[i]
                        c_new[i, 0] = np.clip(c_new[i, 0], s/2, 1 - s/2)
                        c_new[i, 1] = np.clip(c_new[i, 1], s/2, 1 - s/2)
                        
                        obj_new = _objective(c_new, s_new)
                        if obj_new > best_obj + 1e-10:
                            best_centers = c_new
                            best_sides = s_new
                            best_obj = obj_new
                            improved = True
        if not improved:
            break
    
    return best_centers, best_sides


# EVOLVE-BLOCK-END
