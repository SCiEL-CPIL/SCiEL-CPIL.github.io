import numpy as np
import time
from scipy.optimize import linprog
from scipy.spatial import Delaunay

def squares_in_square10():
    n = 10
    start = time.time()
    deadline = start + 570.0
    tol = 1e-7

    def check_valid(cx, cy, s):
        if not (np.all(np.isfinite(cx)) and np.all(np.isfinite(cy)) and np.all(np.isfinite(s))):
            return False
        if np.any(s <= 0):
            return False
        if np.any(cx - s/2 < -tol) or np.any(cx + s/2 > 1 + tol):
            return False
        if np.any(cy - s/2 < -tol) or np.any(cy + s/2 > 1 + tol):
            return False
        for i in range(n):
            for j in range(i+1, n):
                t = (s[i] + s[j])/2 - tol
                if abs(cx[i]-cx[j]) < t and abs(cy[i]-cy[j]) < t:
                    return False
        return True

    def local_search(cx, cy, s, time_budget):
        """Local search with expansion and movement."""
        best_score = np.sum(s)
        best_cx, best_cy, best_s = cx.copy(), cy.copy(), s.copy()
        steps = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
        dirs = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1),(0,0)]
        
        improved = True
        iter_count = 0
        while improved and time.time() < start + time_budget:
            improved = False
            iter_count += 1
            for i in range(n):
                # Try expanding
                for ds in [0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]:
                    new_s = best_s.copy()
                    new_s[i] += ds
                    if new_s[i] > 1.0:
                        continue
                    if check_valid(best_cx, best_cy, new_s):
                        best_s = new_s
                        best_score = np.sum(best_s)
                        improved = True
                        break
                # Try moving
                for dx, dy in dirs:
                    for step in steps:
                        new_cx = best_cx.copy()
                        new_cy = best_cy.copy()
                        new_cx[i] += dx * step
                        new_cy[i] += dy * step
                        if check_valid(new_cx, new_cy, best_s):
                            # Only accept if it enables future expansion or improves packing
                            # For now, accept any valid move that doesn't reduce score
                            # Actually, just accept to explore
                            if dx != 0 or dy != 0:
                                best_cx = new_cx
                                best_cy = new_cy
                                improved = True
                                break
        return best_cx, best_cy, best_s

    def random_packing(min_s=0.05, max_s=0.45, n_trials=500):
        """Random greedy packing."""
        cx = np.zeros(n)
        cy = np.zeros(n)
        s = np.zeros(n)
        for i in range(n):
            placed = False
            for _ in range(n_trials):
                s_i = min_s + (max_s - min_s) * np.random.rand()
                cx_i = s_i/2 + (1 - s_i) * np.random.rand()
                cy_i = s_i/2 + (1 - s_i) * np.random.rand()
                valid = True
                for j in range(i):
                    t = (s[j] + s_i)/2 - tol
                    if abs(cx_i - cx[j]) < t and abs(cy_i - cy[j]) < t:
                        valid = False
                        break
                if valid and cx_i - s_i/2 >= -tol and cx_i + s_i/2 <= 1+tol and cy_i - s_i/2 >= -tol and cy_i + s_i/2 <= 1+tol:
                    cx[i] = cx_i
                    cy[i] = cy_i
                    s[i] = s_i
                    placed = True
                    break
            if not placed:
                s[i] = min_s
                cx[i] = s[i]/2
                cy[i] = s[i]/2
        return cx, cy, s

    def grid_packing(cols, rows, s_val):
        """Create a grid layout."""
        cx = np.zeros(n)
        cy = np.zeros(n)
        s = np.zeros(n)
        count = 0
        for r in range(rows):
            for c in range(cols):
                if count >= n:
                    break
                cx[count] = (c + 0.5) * s_val
                cy[count] = (r + 0.5) * s_val
                s[count] = s_val
                count += 1
        # Fill remaining
        for i in range(count, n):
            s[i] = min_s
            cx[i] = s[i]/2
            cy[i] = s[i]/2
        return cx, cy, s

    def guillotine_layout():
        """Various guillotine-style layouts."""
        layouts = []
        
        # Layout 1: 2x5 grid
        cx = np.array([0.1, 0.3, 0.5, 0.7, 0.9]*2)
        cy = np.array([0.1]*5 + [0.3]*5)
        s = np.full(10, 0.2)
        layouts.append((cx, cy, s))
        
        # Layout 2: 1 large + 9 small
        cx = np.array([0.25, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75])
        cy = np.array([0.5, 0.125, 0.375, 0.625, 0.875, 0.125, 0.375, 0.625, 0.875, 0.5])
        s = np.array([0.5] + [0.25]*9)
        # Fix: right side should be 4 rows x 2 cols of 0.25? No, width is 0.5, so 2 cols of 0.25
        cx2 = np.array([0.25, 0.625, 0.875, 0.625, 0.875, 0.625, 0.875, 0.625, 0.875, 0.625])
        cy2 = np.array([0.5, 0.125, 0.125, 0.375, 0.375, 0.625, 0.625, 0.875, 0.875, 0.5])
        s2 = np.array([0.5] + [0.25]*9)
        layouts.append((cx2, cy2, s2))
        
        # Layout 3: 3 in top, 3 in middle, 4 in bottom (uneven)
        cx3 = np.array([0.166, 0.5, 0.833, 0.166, 0.5, 0.833, 0.125, 0.375, 0.625, 0.875])
        cy3 = np.array([0.833]*3 + [0.5]*3 + [0.166]*4)
        s3 = np.array([0.333]*6 + [0.25]*4)
        layouts.append((cx3, cy3, s3))
        
        # Layout 4: 2 large, 8 small
        cx4 = np.array([0.25, 0.75, 0.125, 0.375, 0.625, 0.875, 0.125, 0.375, 0.625, 0.875])
        cy4 = np.array([0.25, 0.25, 0.5, 0.5, 0.5, 0.5, 0.75, 0.75, 0.75, 0.75])
        s4 = np.array([0.5, 0.5] + [0.25]*8)
        layouts.append((cx4, cy4, s4))
        
        return [l for l in layouts if check_valid(l[0], l[1], l[2])]

    best_score = -np.inf
    best_result = None

    # Strategy 1: Guillotine layouts with local search
    for cx, cy, s in guillotine_layout():
        if time.time() > deadline - 30:
            break
        rcx, rcy, rs = local_search(cx, cy, s, time.time() + 50)
        score = np.sum(rs)
        if score > best_score and check_valid(rcx, rcy, rs):
            best_score = score
            best_result = (rcx, rcy, rs)

    # Strategy 2: Random packings with local search
    np.random.seed(42)
    for trial in range(8):
        if time.time() > deadline - 20:
            break
        cx, cy, s = random_packing()
        rcx, rcy, rs = local_search(cx, cy, s, time.time() + 40)
        score = np.sum(rs)
        if score > best_score and check_valid(rcx, rcy, rs):
            best_score = score
            best_result = (rcx, rcy, rs)

    # Strategy 3: Equal-size grid as baseline
    for k in [3, 4, 5]:
        s_val = 1.0/k
        cx, cy, s = grid_packing(k, k, s_val)
        if check_valid(cx, cy, s):
            score = np.sum(s)
            if score > best_score:
                best_score = score
                best_result = (cx, cy, s)

    # Strategy 4: 2x5 grid with expansion
    cx = np.array([0.1, 0.3, 0.5, 0.7, 0.9]*2)
    cy = np.array([0.1]*5 + [0.3]*5)
    s = np.full(10, 0.2)
    if check_valid(cx, cy, s):
        rcx, rcy, rs = local_search(cx, cy, s, time.time() + 60)
        score = np.sum(rs)
        if score > best_score and check_valid(rcx, rcy, rs):
            best_score = score
            best_result = (rcx, rcy, rs)

    # Fallback if nothing valid found
    if best_result is None:
        k = 4
        side = 1.0/k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        s = np.full(n, side)
        centers = np.stack([cx, cy], axis=1)
        return centers, s

    cx, cy, s = best_result
    centers = np.stack([cx, cy], axis=1)
    return centers, s
