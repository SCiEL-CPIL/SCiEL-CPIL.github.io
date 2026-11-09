# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, linprog


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
    deadline = time.time() + 570  # safety margin before 600s limit

    def _is_feasible(centers, sides, tol=1e-7):
        """Check if the configuration is feasible."""
        if centers.shape != (n, 2) or sides.shape != (n,):
            return False
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
            return False
        if np.any(sides <= 0):
            return False
        # Boundary check
        if np.any(centers[:, 0] - sides/2 < -tol) or np.any(centers[:, 0] + sides/2 > 1 + tol):
            return False
        if np.any(centers[:, 1] - sides/2 < -tol) or np.any(centers[:, 1] + sides/2 > 1 + tol):
            return False
        # Overlap check
        for i in range(n):
            for j in range(i + 1, n):
                t = (sides[i] + sides[j]) / 2 - tol
                if abs(centers[i, 0] - centers[j, 0]) < t and abs(centers[i, 1] - centers[j, 1]) < t:
                    return False
        return True

    def _sum_sides(sides):
        return np.sum(sides)

    def _objective(centers, sides):
        """Maximize sum of sides, penalize infeasibility."""
        if _is_feasible(centers, sides):
            return _sum_sides(sides)
        return -1e6

    def _softplus(x):
        """Numerically stable softplus for smooth penalties."""
        return np.where(x > 30, x, np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0))

    def _smooth_penalty(x):
        """
        Smooth penalty function for scipy minimize.
        x = [cx_0, cy_0, s_0, cx_1, cy_1, s_1, ..., cx_9, cy_9, s_9]
        Uses softplus to provide continuous gradients.
        """
        centers = x[:2*n].reshape(n, 2)
        sides = x[2*n:]
        
        # Boundary violations: push centers inward
        penalty = np.zeros(1)
        # Left boundary: cx - s/2 >= 0
        penalty[0] += np.sum(_softplus(-(centers[:, 0] - sides/2)))
        # Right boundary: cx + s/2 <= 1
        penalty[0] += np.sum(_softplus(centers[:, 0] + sides/2 - 1))
        # Bottom boundary: cy - s/2 >= 0
        penalty[0] += np.sum(_softplus(-(centers[:, 1] - sides/2)))
        # Top boundary: cy + s/2 <= 1
        penalty[0] += np.sum(_softplus(centers[:, 1] + sides/2 - 1))
        
        # Overlap violations: for each pair, |cx_i - cx_j| >= (s_i+s_j)/2 OR |cy_i - cy_j| >= (s_i+s_j)/2
        for i in range(n):
            for j in range(i + 1, n):
                t = (sides[i] + sides[j]) / 2
                dx = abs(centers[i, 0] - centers[j, 0])
                dy = abs(centers[i, 1] - centers[j, 1])
                vx = t - dx
                vy = t - dy
                penalty[0] += _softplus(vx) + _softplus(vy) - _softplus(vx + vy)
        
        penalty[0] += np.sum(_softplus(-sides))
        
        return penalty[0]

    def _scipy_objective(x):
        """Combined objective: maximize sum of sides, penalize infeasibility with smooth penalties."""
        centers = x[:2*n].reshape(n, 2)
        sides = x[2*n:]
        
        if np.any(sides <= 0.001):
            return 1e6
        
        sum_sides = np.sum(sides)
        pen = _smooth_penalty(x)
        
        return -sum_sides + 100.0 * pen

    # Start with the analytical construction: 2 large (side 0.5) + 8 small (side 0.25)
    centers = np.zeros((n, 2), dtype=np.float64)
    sides = np.zeros(n, dtype=np.float64)

    centers[0] = [0.25, 0.25]
    sides[0] = 0.5
    centers[1] = [0.75, 0.75]
    sides[1] = 0.5

    tl_positions = [(0.125, 0.625), (0.375, 0.625), (0.125, 0.875), (0.375, 0.875)]
    br_positions = [(0.625, 0.125), (0.875, 0.125), (0.625, 0.375), (0.875, 0.375)]

    all_small = tl_positions + br_positions
    for i, (cx, cy) in enumerate(all_small):
        centers[2 + i] = [cx, cy]
        sides[2 + i] = 0.25

    best_centers = centers.copy()
    best_sides = sides.copy()
    best_obj = _objective(best_centers, best_sides)

    def _to_x(centers, sides):
        x = np.empty(3 * n)
        x[:2*n] = centers.flatten()
        x[2*n:] = sides
        return x

    def _from_x(x):
        centers = x[:2*n].reshape(n, 2)
        sides = x[2*n:]
        return centers, sides

    def _solve_lp_for_separation(sep_matrix):
        """
        Given a separation pattern, solve the LP to maximize sum of sides.
        sep_matrix[i,j] for i<j: 0 = i left of j, 1 = i right of j, 2 = i below j, 3 = i above j
        Returns (centers, sides, objective_value) or None if infeasible.
        """
        # Variables: x_i, y_i, s_i for i in 0..n-1, plus objective variable z
        # Total variables: 3n + 1
        # We want to maximize sum(s_i) subject to constraints
        
        # Use linprog: minimize c^T x
        # Variables: [x_0, y_0, s_0, x_1, y_1, s_1, ..., x_9, y_9, s_9]
        # Objective: minimize -sum(s_i)
        
        c = np.zeros(3 * n)
        for i in range(n):
            c[3*i + 2] = -1.0  # maximize s_i
        
        # Constraints in form A_ub @ x <= b_ub
        A_rows = []
        b_rows = []
        
        # Boundary constraints:
        # x_i - s_i/2 >= 0  =>  -x_i + s_i/2 <= 0
        # x_i + s_i/2 <= 1  =>  x_i + s_i/2 <= 1
        # y_i - s_i/2 >= 0  =>  -y_i + s_i/2 <= 0
        # y_i + s_i/2 <= 1  =>  y_i + s_i/2 <= 1
        for i in range(n):
            # -x_i + s_i/2 <= 0
            row = np.zeros(3 * n)
            row[3*i] = -1.0
            row[3*i + 2] = 0.5
            A_rows.append(row)
            b_rows.append(0.0)
            
            # x_i + s_i/2 <= 1
            row = np.zeros(3 * n)
            row[3*i] = 1.0
            row[3*i + 2] = 0.5
            A_rows.append(row)
            b_rows.append(1.0)
            
            # -y_i + s_i/2 <= 0
            row = np.zeros(3 * n)
            row[3*i + 1] = -1.0
            row[3*i + 2] = 0.5
            A_rows.append(row)
            b_rows.append(0.0)
            
            # y_i + s_i/2 <= 1
            row = np.zeros(3 * n)
            row[3*i + 1] = 1.0
            row[3*i + 2] = 0.5
            A_rows.append(row)
            b_rows.append(1.0)
        
        # Separation constraints for each pair
        for i in range(n):
            for j in range(i + 1, n):
                sep = sep_matrix[i, j]
                # s_i/2 + s_j/2 <= distance in separation direction
                # Equivalent to: -s_i/2 - s_j/2 + separation_coord <= 0
                
                row = np.zeros(3 * n)
                row[3*i + 2] = -0.5
                row[3*j + 2] = -0.5
                
                if sep == 0:  # i left of j: x_j - x_i >= (s_i + s_j)/2
                    row[3*i] = -1.0
                    row[3*j] = 1.0
                elif sep == 1:  # i right of j: x_i - x_j >= (s_i + s_j)/2
                    row[3*i] = 1.0
                    row[3*j] = -1.0
                elif sep == 2:  # i below j: y_j - y_i >= (s_i + s_j)/2
                    row[3*i + 1] = -1.0
                    row[3*j + 1] = 1.0
                elif sep == 3:  # i above j: y_i - y_j >= (s_i + s_j)/2
                    row[3*i + 1] = 1.0
                    row[3*j + 1] = -1.0
                
                A_rows.append(row)
                b_rows.append(0.0)
        
        A_ub = np.array(A_rows)
        b_ub = np.array(b_rows)
        
        # Bounds: x_i, y_i in [0, 1], s_i in [0.001, 1.0]
        bounds = []
        for i in range(n):
            bounds.append((0.0, 1.0))  # x_i
            bounds.append((0.0, 1.0))  # y_i
            bounds.append((0.001, 1.0))  # s_i
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0 and res.success:
                x = res.x
                c_res = np.array([[x[3*i], x[3*i+1]] for i in range(n)])
                s_res = np.array([x[3*i+2] for i in range(n)])
                obj_val = -res.fun  # negated because we minimized
                return c_res, s_res, obj_val
        except Exception:
            pass
        return None

    def _generate_separation_patterns(rng, num_patterns=200):
        """Generate diverse separation patterns using greedy + random approaches."""
        patterns = []
        
        # Pattern 1: The known optimal-looking tiling pattern
        # 2 large diagonal + 8 small in L-shape
        sep1 = np.zeros((n, n), dtype=int)
        # Large squares: 0 bottom-left, 1 top-right
        # Small squares 2-5 in top-left, 6-9 in bottom-right
        # 0 below 1: sep[0,1] = 2
        sep1[0, 1] = 2
        # 0 left of all top-left smalls (2-5)
        for j in range(2, 6):
            sep1[0, j] = 0  # 0 left of j
        # 0 below all bottom-right smalls (6-9)
        for j in range(6, 10):
            sep1[0, j] = 2  # 0 below j
        # 1 right of all top-left smalls (2-5)
        for j in range(2, 6):
            sep1[1, j] = 1  # 1 right of j
        # 1 above all bottom-right smalls (6-9)
        for j in range(6, 10):
            sep1[1, j] = 3  # 1 above j
        # Top-left smalls among themselves (2x2 grid)
        # 2 bottom-left, 3 bottom-right, 4 top-left, 5 top-right of TL region
        sep1[2, 3] = 0  # 2 left of 3
        sep1[2, 4] = 2  # 2 below 4
        sep1[2, 5] = 0  # 2 left of 5 (also below, but use one)
        sep1[3, 4] = 2  # 3 below 4 (also right, but use one)
        sep1[3, 5] = 2  # 3 below 5
        sep1[4, 5] = 0  # 4 left of 5
        # Bottom-right smalls among themselves (2x2 grid)
        # 6 bottom-left, 7 bottom-right, 8 top-left, 9 top-right of BR region
        sep1[6, 7] = 0
        sep1[6, 8] = 2
        sep1[6, 9] = 0
        sep1[7, 8] = 2
        sep1[7, 9] = 2
        sep1[8, 9] = 0
        patterns.append(sep1)
        
        # Generate random patterns
        for _ in range(num_patterns):
            sep = np.zeros((n, n), dtype=int)
            for i in range(n):
                for j in range(i + 1, n):
                    sep[i, j] = rng.integers(0, 4)
            patterns.append(sep)
        
        # Generate grid-based patterns (different grid configurations)
        # 5x2 grid
        for rows in [2, 3, 4, 5]:
            for cols in [2, 3, 4, 5]:
                if rows * cols >= n:
                    sep = np.zeros((n, n), dtype=int)
                    # Assign squares to grid positions
                    positions = []
                    for r in range(rows):
                        for c in range(cols):
                            if len(positions) < n:
                                positions.append((r, c))
                    if len(positions) == n:
                        for i in range(n):
                            for j in range(i + 1, n):
                                ri, ci = positions[i]
                                rj, cj = positions[j]
                                if ci < cj:
                                    sep[i, j] = 0
                                elif ci > cj:
                                    sep[i, j] = 1
                                elif ri < rj:
                                    sep[i, j] = 2
                                elif ri > rj:
                                    sep[i, j] = 3
                                else:
                                    sep[i, j] = 0
                        patterns.append(sep)
        
        return patterns

    # Phase 1: LP-based search over separation topologies
    rng = np.random.default_rng(42)
    patterns = _generate_separation_patterns(rng, num_patterns=100)
    
    for pat_idx, sep_pattern in enumerate(patterns):
        if time.time() > deadline - 120:
            break
        
        result = _solve_lp_for_separation(sep_pattern)
        if result is not None:
            c_res, s_res, obj_val = result
            if _is_feasible(c_res, s_res):
                if obj_val > best_obj:
                    best_centers = c_res.copy()
                    best_sides = s_res.copy()
                    best_obj = obj_val
    
    # Phase 2: Local search with random perturbations
    step_size = 0.05
    max_iters = 15000
    
    current_x = _to_x(best_centers, best_sides)
    
    for it in range(max_iters):
        if time.time() > deadline - 60:
            break
        
        x_p = current_x + rng.normal(0, step_size, 3 * n)
        centers_p, sides_p = _from_x(x_p)
        sides_p = np.maximum(sides_p, 0.01)
        
        for i in range(n):
            s = sides_p[i]
            centers_p[i, 0] = np.clip(centers_p[i, 0], s/2, 1 - s/2)
            centers_p[i, 1] = np.clip(centers_p[i, 1], s/2, 1 - s/2)
        
        x_p = _to_x(centers_p, sides_p)
        obj_p = _objective(centers_p, sides_p)
        if obj_p > best_obj:
            best_centers = centers_p.copy()
            best_sides = sides_p.copy()
            best_obj = obj_p
            current_x = x_p
            step_size *= 0.95
        else:
            if rng.random() < 0.1:
                current_x = x_p
                step_size *= 1.05
        
        if it % 100 == 0:
            current_x = _to_x(best_centers, best_sides)

    # Phase 3: Use scipy with smooth penalties to polish
    if time.time() < deadline - 30:
        bounds = []
        for i in range(n):
            bounds.append((0.0, 1.0))
            bounds.append((0.0, 1.0))
            bounds.append((0.01, 1.0))
        
        starting_points = [_to_x(best_centers, best_sides)]
        
        for _ in range(3):
            if time.time() > deadline - 30:
                break
            c_rand = np.random.uniform(0.1, 0.9, (n, 2))
            s_rand = np.random.uniform(0.08, 0.25, n)
            for i in range(n):
                s = s_rand[i]
                c_rand[i, 0] = np.clip(c_rand[i, 0], s/2, 1 - s/2)
                c_rand[i, 1] = np.clip(c_rand[i, 1], s/2, 1 - s/2)
            starting_points.append(_to_x(c_rand, s_rand))
        
        for x0 in starting_points:
            if time.time() > deadline - 15:
                break
            try:
                res = minimize(_scipy_objective, x0, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12,
                                       'maxls': 50, 'maxfun': 3000})
                c_res, s_res = _from_x(res.x)
                if _is_feasible(c_res, s_res):
                    obj_res = _sum_sides(s_res)
                    if obj_res > best_obj:
                        best_centers = c_res.copy()
                        best_sides = s_res.copy()
                        best_obj = obj_res
            except Exception:
                pass

    # Phase 4: Final high-precision polish with coordinate descent
    scales = [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005]
    for scale in scales:
        if time.time() > deadline:
            break
        for _ in range(100):
            if time.time() > deadline:
                break
            improved = False
            for i in range(n):
                for dx in [-scale, 0, scale]:
                    for dy in [-scale, 0, scale]:
                        for ds in [-scale, 0, scale]:
                            if dx == 0 and dy == 0 and ds == 0:
                                continue
                            c_new = best_centers.copy()
                            s_new = best_sides.copy()
                            c_new[i, 0] += dx
                            c_new[i, 1] += dy
                            s_new[i] = max(s_new[i] + ds, 0.01)
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

    # Final feasibility check and return
    if not _is_feasible(best_centers, best_sides):
        centers_fb = np.zeros((n, 2), dtype=np.float64)
        sides_fb = np.zeros(n, dtype=np.float64)
        centers_fb[0] = [0.25, 0.25]; sides_fb[0] = 0.5
        centers_fb[1] = [0.75, 0.75]; sides_fb[1] = 0.5
        tl = [(0.125,0.625),(0.375,0.625),(0.125,0.875),(0.375,0.875)]
        br = [(0.625,0.125),(0.875,0.125),(0.625,0.375),(0.875,0.375)]
        for i, (cx, cy) in enumerate(tl + br):
            centers_fb[2+i] = [cx, cy]; sides_fb[2+i] = 0.25
        best_centers = centers_fb
        best_sides = sides_fb

    return best_centers, best_sides


# EVOLVE-BLOCK-END
