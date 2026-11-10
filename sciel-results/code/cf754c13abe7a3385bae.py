# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

TOL = 1e-7

def squares_in_square10() -> tuple:
    """
    Place 10 axis-aligned squares with pairwise disjoint interiors inside the
    unit square [0,1]^2 so as to MAXIMIZE the sum of their side lengths.

    Returns:
        centers: np.ndarray of shape (10, 2) with the (x, y) center of each square.
        sides:   np.ndarray of shape (10,) with the side length of each square.
    """
    n = 10
    t0 = time.time()
    DEADLINE = t0 + 580.0  # Safety margin

    def _is_feasible(centers, sides, tol=TOL):
        cx = centers[:, 0]
        cy = centers[:, 1]
        s = sides
        if np.any(~np.isfinite(centers)) or np.any(~np.isfinite(s)):
            return False
        if np.any(s <= 0):
            return False
        for i in range(n):
            if cx[i] - s[i] / 2.0 < -tol or cx[i] + s[i] / 2.0 > 1.0 + tol:
                return False
            if cy[i] - s[i] / 2.0 < -tol or cy[i] + s[i] / 2.0 > 1.0 + tol:
                return False
        for i in range(n):
            for j in range(i + 1, n):
                t = (s[i] + s[j]) / 2.0 - tol
                if abs(cx[i] - cx[j]) < t and abs(cy[i] - cy[j]) < t:
                    return False
        return True

    def _sum_sides(centers, sides):
        return np.sum(sides)

    def _local_refine(centers, sides, deadline):
        """
        Perform local refinement using coordinate descent.
        We alternate between:
        1. For each square, try to increase its side length while keeping center fixed.
        2. For each square, try small shifts in x and y to maintain feasibility.
        
        We use a decreasing step size for side length increases.
        """
        best_centers = centers.copy()
        best_sides = sides.copy()
        best_score = np.sum(best_sides)
        
        # Try multiple step size schedules
        step = 0.005
        min_step = 1e-5
        no_improve = 0
        max_no_improve = 200
        
        while time.time() < deadline and step > min_step:
            improved = False
            for i in range(n):
                # Try to increase side length
                new_side = best_sides[i] + step
                if new_side > 1.0:
                    continue
                
                trial_s = best_sides.copy()
                trial_s[i] = new_side
                
                # First, try current center
                if _is_feasible(best_centers, trial_s):
                    best_sides = trial_s
                    best_score = np.sum(best_sides)
                    improved = True
                else:
                    # Try small shifts
                    for dx in [-step, -step/2, 0, step/2, step]:
                        for dy in [-step, -step/2, 0, step/2, step]:
                            trial_c = best_centers.copy()
                            trial_c[i, 0] += dx
                            trial_c[i, 1] += dy
                            if _is_feasible(trial_c, trial_s):
                                best_centers = trial_c
                                best_sides = trial_s
                                best_score = np.sum(best_sides)
                                improved = True
                                break
                        if improved:
                            break
                
                if improved:
                    no_improve = 0
                else:
                    no_improve += 1
            
            if not improved:
                step *= 0.9
                if no_improve > max_no_improve:
                    break
            else:
                no_improve = 0
        
        return best_centers, best_sides, best_score

    # --- Analytic Initialization: 3.0 Sum Layout ---
    # 8 squares of side 0.25 (2 rows of 4)
    # 2 squares of side 0.5 (1 row of 2)
    centers = np.zeros((n, 2))
    sides = np.zeros(n)
    
    # Bottom 2 rows: 4 squares each, side 0.25
    for i in range(8):
        col = i % 4
        row = i // 4
        side = 0.25
        centers[i, 0] = (col + 0.5) * side
        centers[i, 1] = (row + 0.5) * side
        sides[i] = side
        
    # Top row: 2 squares, side 0.5
    for i in range(2):
        col = i
        row = 2
        side = 0.5
        centers[i + 8, 0] = (col + 0.5) * side
        centers[i + 8, 1] = (row + 0.5) * side
        sides[i + 8] = side

    # Verify feasibility of start
    if not _is_feasible(centers, sides):
        # Fallback to 2.5 layout if analytic fails (should not happen)
        centers = np.zeros((n, 2))
        sides = np.zeros(n)
        for i in range(n):
            col = i % 4
            row = i // 4
            side = 0.25
            centers[i, 0] = (col + 0.5) * side
            centers[i, 1] = (row + 0.5) * side
            sides[i] = side

    # Refine from this initial layout
    centers, sides, score = _local_refine(centers, sides, DEADLINE)

    # --- Additional Multi-Start with Perturbations ---
    best_centers = centers
    best_sides = sides
    best_score = score

    if time.time() < DEADLINE - 60:  # Need at least 60s remaining
        # Generate several perturbed versions of the initial layout
        n_perturbations = 8
        for p in range(n_perturbations):
            if time.time() > DEADLINE - 30:
                break
            
            perturbed_centers = best_centers.copy()
            perturbed_sides = best_sides.copy()
            
            # Randomly perturb a few squares
            num_perturb = 3
            indices = np.random.choice(n, num_perturb, replace=False)
            for idx in indices:
                # Small random shift
                dx = np.random.uniform(-0.05, 0.05)
                dy = np.random.uniform(-0.05, 0.05)
                perturbed_centers[idx, 0] += dx
                perturbed_centers[idx, 1] += dy
                # Slightly reduce side to maintain feasibility
                perturbed_sides[idx] *= 0.95
            
            # Refine
            c_ref, s_ref, sc_ref = _local_refine(perturbed_centers, perturbed_sides, DEADLINE)
            
            if sc_ref > best_score:
                best_centers = c_ref
                best_sides = s_ref
                best_score = sc_ref

    return best_centers.astype(np.float64), best_sides.astype(np.float64)

# EVOLVE-BLOCK-END
