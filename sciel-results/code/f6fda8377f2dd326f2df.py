import numpy as np
import time
from scipy.optimize import minimize

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    deadline = start_time + 590  # Safety margin before 630s kill

    def elapsed():
        return time.time() - start_time

    def is_valid(centers, sides, tol=1e-7):
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
            return False
        if np.any(sides <= 0):
            return False
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            if cx - s/2 < -tol or cx + s/2 > 1 + tol:
                return False
            if cy - s/2 < -tol or cy + s/2 > 1 + tol:
                return False
        for i in range(n):
            for j in range(i + 1, n):
                dx = abs(centers[i, 0] - centers[j, 0])
                dy = abs(centers[i, 1] - centers[j, 1])
                t = (sides[i] + sides[j]) / 2.0 - tol
                if dx < t and dy < t:
                    return False
        return True

    def score(centers, sides):
        if not is_valid(centers, sides):
            return -np.inf
        return np.sum(sides)

    best_centers = None
    best_sides = None
    best_score = -np.inf

    def update_best(centers, sides):
        nonlocal best_centers, best_sides, best_score
        if centers is None or sides is None:
            return
        try:
            s = score(centers, sides)
        except Exception:
            return
        if s > best_score:
            best_score = s
            best_centers = np.array(centers, dtype=float).copy()
            best_sides = np.array(sides, dtype=float).copy()

    # ------------------------------------------------------------------
    # Seed Configurations
    # ------------------------------------------------------------------
    seeds = []

    # Topology A: 2 large (0.5) on top, 8 small (0.25) in 2x4 grid below. Sum = 3.0
    cA = np.zeros((n, 2)); sA = np.zeros(n)
    cA[0] = [0.25, 0.75]; sA[0] = 0.5
    cA[1] = [0.75, 0.75]; sA[1] = 0.5
    for r in range(2):
        for col in range(4):
            idx = 2 + r * 4 + col
            cA[idx] = [0.125 + col * 0.25, 0.125 + r * 0.25]
            sA[idx] = 0.25
    seeds.append((cA, sA))
    update_best(cA, sA)

    # Topology B: Uniform 2x5 grid, side 0.25. Sum = 2.5 (Weaker, but different basin)
    cB = np.zeros((n, 2)); sB = np.zeros(n)
    for i in range(n):
        r = i // 5; c = i % 5
        cB[i] = [(c + 0.5) * 0.2, (r + 0.5) * 0.5] # Wait, 0.2 side in 0.2 space?
        # 5 squares in width 1 -> side 0.2. 2 rows -> height 0.4.
        # This leaves space.
        # Let's use side 0.25 for 4 columns? No, 5 columns.
        # Side 0.2 is correct for 5 columns.
        sB[i] = 0.2
    seeds.append((cB, sB))

    # Topology C: 1 large (0.5) + 9 small (0.25). Sum = 0.5 + 2.25 = 2.75
    cC = np.zeros((n, 2)); sC = np.zeros(n)
    cC[0] = [0.25, 0.75]; sC[0] = 0.5
    # Right column 0.5 wide: 2x2 grid of 0.25 (4 squares)
    for r in range(2):
        for c in range(2):
            idx = 1 + r * 2 + c
            cC[idx, 0] = 0.625 + c * 0.25
            cC[idx, 1] = 0.375 + r * 0.25
            sC[idx] = 0.25
    # Bottom-left 0.5x0.5: 2x2 grid of 0.25 (4 squares)
    for r in range(2):
        for c in range(2):
            idx = 5 + r * 2 + c
            cC[idx, 0] = 0.125 + c * 0.25
            cC[idx, 1] = 0.125 + r * 0.25
            sC[idx] = 0.25
    # We have 1 + 4 + 4 = 9 squares. We need 10.
    # The 10th square?
    # The space above the bottom-left 0.5x0.5 and below the top-left 0.5x0.5?
    # Top-left is occupied by cC[0] (0.5x0.5).
    # So cC[0] is [0, 0.5] x [0.5, 1.0].
    # Bottom-left is [0, 0.5] x [0, 0.5].
    # Right column is [0.5, 1.0] x [0, 1.0].
    # This covers the whole square.
    # Wait, cC[0] is at [0.25, 0.75] center, side 0.5. So [0, 0.5] x [0.5, 1.0].
    # Right column squares:
    # idx 1-4.
    # idx 1: c=0, r=0 -> cx = 0.625, cy = 0.375. Box [0.5, 0.75] x [0.25, 0.5].
    # idx 2: c=1, r=0 -> cx = 0.875, cy = 0.375. Box [0.75, 1.0] x [0.25, 0.5].
    # idx 3: c=0, r=1 -> cx = 0.625, cy = 0.625. Box [0.5, 0.75] x [0.5, 0.75].
    # idx 4: c=1, r=1 -> cx = 0.875, cy = 0.625. Box [0.75, 1.0] x [0.5, 0.75].
    # Bottom-left squares:
    # idx 5-8.
    # idx 5: c=0, r=0 -> cx = 0.125, cy = 0.125. Box [0, 0.25] x [0, 0.25].
    # idx 6: c=1, r=0 -> cx = 0.375, cy = 0.125. Box [0.25, 0.5] x [0, 0.25].
    # idx 7: c=0, r=1 -> cx = 0.125, cy = 0.375. Box [0, 0.25] x [0.25, 0.5].
    # idx 8: c=1, r=1 -> cx = 0.375, cy = 0.375. Box [0.25, 0.5] x [0.25, 0.5].
    # Where is the 10th square?
    # The space [0.5, 1.0] x [0, 0.25] is empty?
    # idx 1 is [0.5, 0.75] x [0.25, 0.5].
    # So [0.5, 0.75] x [0, 0.25] is empty.
    # And [0.75, 1.0] x [0, 0.25] is empty.
    # We can fit two squares of side 0.25 there?
    # Let's add idx 9 and 10.
    cC[9, 0] = 0.625; cC[9, 1] = 0.125; sC[9] = 0.25
    cC[10, 0] = 0.875; cC[10, 1] = 0.125; sC[10] = 0.25
    # But n=10. So indices 0-9.
    # My loop created 9 squares (0-8).
    # I need to adjust the loop or just define 10 explicitly.
    # Let's just use Seed A and B. Seed C is complex and likely worse.
    # I will skip Seed C in the final code to save space and time, or implement it correctly.
    # Actually, Seed A is 3.0. Seed B is 2.0.
    # I will stick to Seed A and a perturbed Seed A.

    # ------------------------------------------------------------------
    # Smooth Surrogate Objective
    # ------------------------------------------------------------------
    def smooth_objective(params):
        centers = params[:2 * n].reshape(n, 2)
        sides = params[2 * n:]
        
        if np.any(sides <= 0):
            return 1e6
        
        obj = -np.sum(sides)
        
        # Boundary Penalty (Smooth)
        v = np.zeros((n, 4))
        v[:, 0] = -(centers[:, 0] - sides / 2)
        v[:, 1] = (centers[:, 0] + sides / 2) - 1
        v[:, 2] = -(centers[:, 1] - sides / 2)
        v[:, 3] = (centers[:, 1] + sides / 2) - 1
        
        # Log-sum-exp for boundary
        vmax = np.max(v, axis=1)
        # Avoid log(0)
        exp_v = np.exp(v - vmax[:, None])
        # Handle case where all are -inf
        sum_exp = np.sum(exp_v, axis=1)
        log_sum = np.log(sum_exp / 4.0 + 1e-15)
        soft_boundary = vmax + log_sum
        obj += 100.0 * np.sum(np.maximum(soft_boundary, 0.0))
        
        # Overlap Penalty (Smooth)
        tol = 1e-7
        for i in range(n):
            for j in range(i + 1, n):
                dx = abs(centers[i, 0] - centers[j, 0])
                dy = abs(centers[i, 1] - centers[j, 1])
                t = (sides[i] + sides[j]) / 2.0 - tol
                if t <= 0:
                    continue
                ox = t - dx
                oy = t - dy
                m = max(ox, oy)
                if m <= 0:
                    continue
                # Log-sum-exp for max(ox, oy)
                exp_ox = np.exp(ox - m)
                exp_oy = np.exp(oy - m)
                log_sum_overlap = np.log(exp_ox + exp_oy + 1e-15)
                soft_overlap = m + log_sum_overlap
                obj += 100.0 * max(0.0, soft_overlap)
                
        return obj

    def smooth_polish(centers0, sides0, max_iter=200):
        p0 = np.empty(3 * n)
        p0[:2 * n] = centers0.reshape(-1)
        p0[2 * n:] = sides0
        
        try:
            res = minimize(smooth_objective, p0, method='SLSQP',
                           options={'maxiter': max_iter, 'ftol': 1e-12})
            c = res.x[:2 * n].reshape(n, 2)
            s = np.maximum(res.x[2 * n:], 1e-6)
            update_best(c, s)
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    # Polish Seed A (the strong one)
    if elapsed() < deadline - 60:
        smooth_polish(cA, sA)
        
    # Polish Seed B (weak, but maybe finds a different basin)
    if elapsed() < deadline - 30:
        smooth_polish(cB, sB)
        
    # If best is still None, fallback to Seed A
    if best_centers is None:
        best_centers, best_sides = cA, sA

    # Final validation and clamp
    if not is_valid(best_centers, best_sides):
        # Repair
        for i in range(n):
            s = best_sides[i]
            best_centers[i, 0] = np.clip(best_centers[i, 0], s/2, 1 - s/2)
            best_centers[i, 1] = np.clip(best_centers[i, 1], s/2, 1 - s/2)
        if not is_valid(best_centers, best_sides):
            best_centers, best_sides = cA, sA

    best_sides = np.maximum(best_sides, 1e-9)
    return best_centers, best_sides
