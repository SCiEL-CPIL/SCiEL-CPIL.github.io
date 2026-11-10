# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, linprog


def squares_in_square10() -> tuple:
    """
    Maximize sum of side lengths of 10 axis-aligned squares in unit square.
    Returns (centers, sides) where centers is (10, 2) and sides is (10,).
    """
    start_time = time.time()
    deadline = start_time + 570.0  # Safety margin

    def _is_valid(centers, sides, tol=1e-7):
        n = len(sides)
        if n != 10:
            return False
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
            return False
        if np.any(sides <= 0):
            return False
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            if cx - s/2 < -tol or cx + s/2 > 1.0 + tol:
                return False
            if cy - s/2 < -tol or cy + s/2 > 1.0 + tol:
                return False
        for i in range(n):
            for j in range(i+1, n):
                t = (sides[i] + sides[j]) / 2.0 - tol
                dx = abs(centers[i,0] - centers[j,0])
                dy = abs(centers[i,1] - centers[j,1])
                if dx < t and dy < t:
                    return False
        return True

    def _max_side_for_square(cx_i, cy_i, s_i, other_cx, other_cy, other_s, i):
        """
        Compute maximum side for square i given its center and other squares.
        """
        max_s = min(2.0 * cx_i, 2.0 * (1.0 - cx_i), 2.0 * cy_i, 2.0 * (1.0 - cy_i))
        max_s = max(max_s, 1e-10)
        
        for j in range(10):
            if j == i:
                continue
            dx = abs(cx_i - other_cx[j])
            dy = abs(cy_i - other_cy[j])
            s_j = other_s[j]
            
            limit_x = 2.0 * dx - s_j
            limit_y = 2.0 * dy - s_j
            limit = max(limit_x, limit_y)
            if limit < max_s:
                max_s = limit
        
        return max(1e-10, max_s)

    def _compute_optimal_sides(centers):
        """
        Given centers, compute the maximum valid side for each square.
        """
        cx = centers[:, 0].copy()
        cy = centers[:, 1].copy()
        s = np.full(10, 0.1)
        
        for _ in range(50):
            s_new = np.zeros(10)
            for i in range(10):
                s_new[i] = _max_side_for_square(cx[i], cy[i], s[i], cx, cy, s, i)
            s = 0.7 * s + 0.3 * s_new
            if np.max(np.abs(s - s_new)) < 1e-12:
                break
        
        for i in range(10):
            s[i] = _max_side_for_square(cx[i], cy[i], s[i], cx, cy, s, i)
        
        return np.maximum(s, 1e-10)

    def _solve_layout_lp(constraints):
        """
        Solve the linear programming problem to maximize sum of sides
        given a set of binary orientation constraints.
        
        constraints: list of tuples (i, j, type) where type is:
                     'L' (i left of j), 'R' (i right of j), 'B' (i below j), 'T' (i above j)
        
        Variables: s_0..s_9, cx_0..cx_9, cy_0..cy_9 (30 vars)
        Objective: maximize sum(s_i) -> minimize -sum(s_i)
        
        Constraints:
        1. Boundary: 0 <= cx_i - s_i/2, cx_i + s_i/2 <= 1, etc.
           cx_i - s_i/2 >= 0  => -cx_i + 0.5*s_i <= 0
           cx_i + s_i/2 <= 1 =>  cx_i - 0.5*s_i <= 1
           Same for y.
        
        2. Orientation:
           If i is Left of j: cx_j - cx_i >= 0.5*s_j + 0.5*s_i
           => -cx_i - cx_j + 0.5*s_i + 0.5*s_j <= 0  (Wait, cx_j - cx_i - 0.5s_j - 0.5s_i >= 0)
           => cx_i - cx_j + 0.5*s_i + 0.5*s_j <= 0  (This is wrong sign for linprog which uses <=)
           Let's stick to: A_ub x <= b_ub
            
           cx_j - cx_i >= 0.5(s_i + s_j)
           => -cx_j + cx_i - 0.5*s_i - 0.5*s_j <= 0  (Incorrect rearrangement)
           => cx_i - cx_j + 0.5*s_i + 0.5*s_j <= 0 is false.
           cx_j - cx_i - 0.5*s_i - 0.5*s_j >= 0
           => -cx_j + cx_i + 0.5*s_i + 0.5*s_j <= 0 (This implies cx_i + ... <= cx_j)
           Let's just use:
           cx_i - cx_j + 0.5*s_i + 0.5*s_j <= 0  is equivalent to cx_i + 0.5s_i <= cx_j - 0.5s_j.
           This means right edge of i <= left edge of j. Correct for i LEFT of j.
           
           If i is Right of j: cx_i - cx_j >= 0.5(s_i + s_j)
           => -cx_i + cx_j + 0.5*s_i + 0.5*s_j <= 0
           
           If i is Below j: cy_j - cy_i >= 0.5(s_i + s_j)
           => cy_i - cy_j + 0.5*s_i + 0.5*s_j <= 0
           
           If i is Above j: cy_i - cy_j >= 0.5(s_i + s_j)
           => -cy_i + cy_j + 0.5*s_i + 0.5*s_j <= 0
        """
        n = 10
        # Variable order: s[0..9], cx[0..9], cy[0..9]
        # Index mapping:
        # s_i: i
        # cx_i: 10 + i
        # cy_i: 20 + i
        
        c = np.zeros(30)
        c[0:10] = -1.0  # Minimize -sum(s)
        
        A_ub = []
        b_ub = []
        
        # Boundary constraints
        for i in range(n):
            # cx_i - 0.5*s_i >= 0  => -cx_i + 0.5*s_i <= 0
            row = np.zeros(30)
            row[i] = 0.5
            row[10+i] = -1.0
            A_ub.append(row)
            b_ub.append(0.0)
            
            # cx_i + 0.5*s_i <= 1  => cx_i - 0.5*s_i <= 1
            row = np.zeros(30)
            row[i] = -0.5
            row[10+i] = 1.0
            A_ub.append(row)
            b_ub.append(1.0)
            
            # cy_i - 0.5*s_i >= 0  => -cy_i + 0.5*s_i <= 0
            row = np.zeros(30)
            row[i] = 0.5
            row[20+i] = -1.0
            A_ub.append(row)
            b_ub.append(0.0)
            
            # cy_i + 0.5*s_i <= 1  => cy_i - 0.5*s_i <= 1
            row = np.zeros(30)
            row[i] = -0.5
            row[20+i] = 1.0
            A_ub.append(row)
            b_ub.append(1.0)
            
        # Orientation constraints
        for i, j, typ in constraints:
            if i > j:
                i, j = j, i
                # Flip logic if we swapped? No, constraints are directed.
                # The input list should have i < j for consistency or handle explicitly.
                # Let's assume input is (i, j, type) where type describes relation of i to j.
                # If we swap i,j, we must swap type.
                # L (i left of j) -> R (j left of i)
                # R (i right of j) -> L (j left of i)
                # B (i below j) -> T (j below i)
                # T (i above j) -> B (j below i)
                if typ == 'L': typ = 'R'
                elif typ == 'R': typ = 'L'
                elif typ == 'B': typ = 'T'
                elif typ == 'T': typ = 'B'
                
            row = np.zeros(30)
            # Coefficients for s_i, s_j
            row[i] = 0.5
            row[j] = 0.5
            
            if typ == 'L': # i left of j: cx_i + 0.5s_i <= cx_j - 0.5s_j
                row[10+i] = 1.0
                row[10+j] = -1.0
            elif typ == 'R': # i right of j: -cx_i + cx_j + 0.5s_i + 0.5s_j <= 0
                row[10+i] = -1.0
                row[10+j] = 1.0
            elif typ == 'B': # i below j: cy_i + 0.5s_i <= cy_j - 0.5s_j
                row[20+i] = 1.0
                row[20+j] = -1.0
            elif typ == 'T': # i above j: -cy_i + cy_j + 0.5s_i + 0.5s_j <= 0
                row[20+i] = -1.0
                row[20+j] = 1.0
                
            A_ub.append(row)
            b_ub.append(0.0)
            
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        bounds = [(0, 1.0)] * 30
        
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.success:
            s = res.x[0:10]
            cx = res.x[10:20]
            cy = res.x[20:30]
            centers = np.column_stack([cx, cy])
            return centers, s
        return None

    def _combinatorial_search():
        """
        Search for valid packings by assigning relative orientations.
        Uses a greedy/backtracking approach with LP relaxation.
        """
        best_score = -1.0
        best_result = None
        
        # Heuristic: Start with a specific ordering or random
        # To make it efficient, we fix the relative order of a subset or use a known good topology.
        # Given 10 squares, 45 pairs. 4^45 is too large.
        # We will use a "shelf" heuristic to generate a base topology, then perturb.
        
        # Generate a few base topologies
        topologies = []
        
        # Topology 1: 2x5 Grid (Shelves)
        # Squares 0-4 in row 0, 5-9 in row 1.
        # Row 0: 0L1, 1L2, 2L3, 3L4
        # Row 1: 5L6, 6L7, 7L8, 8L9
        # Row 0 below Row 1: 0B5, 1B6, 2B7, 3B8, 4B9 (Approximate, actually 0B5, 1B5? No, 0B5, 1B6 is better for alignment)
        # Let's define constraints for a standard grid.
        
        def add_grid_constraints(constraints, rows, cols):
            # indices in row r, col c
            # r in 0..rows-1, c in 0..cols-1
            # idx = r*cols + c
            for r in range(rows):
                for c in range(cols):
                    i = r*cols + c
                    if c < cols - 1:
                        j = r*cols + c + 1
                        constraints.append((i, j, 'L')) # i left of j
                    if r < rows - 1:
                        j = (r+1)*cols + c
                        constraints.append((i, j, 'B')) # i below j
                        
        # 2x5 Grid
        cons = []
        add_grid_constraints(cons, 2, 5)
        topologies.append(cons)
        
        # 5x2 Grid
        cons = []
        add_grid_constraints(cons, 5, 2)
        topologies.append(cons)
        
        # 3x3 Grid + 1 (Centered)
        # 0 1 2
        # 3 4 5
        # 6 7 8
        # 9 is in the middle? Or just a 3x3 with one missing?
        # Let's do 3x3 with square 9 floating.
        cons = []
        # 0L1, 1L2
        # 3L4, 4L5
        # 6L7, 7L8
        # 0B3, 1B4, 2B5
        # 3B6, 4B7, 5B8
        for r in range(3):
            for c in range(3):
                i = r*3 + c
                if c < 2:
                    j = r*3 + c + 1
                    cons.append((i, j, 'L'))
                if r < 2:
                    j = (r+1)*3 + c
                    cons.append((i, j, 'B'))
        # Square 9: Let's put it to the right of 8? Or above 8?
        # Let's just add it as "Right of 8"
        cons.append((8, 9, 'L'))
        topologies.append(cons)
        
        # 4x3 Grid (Missing one)
        cons = []
        # 0 1 2
        # 3 4 5
        # 6 7 8
        # 9 _ _
        for r in range(4):
            for c in range(3):
                if r == 3 and c > 0: continue
                i = r*3 + c
                if c < 2 and (r != 3 or c == 0): # Check if next col exists
                    # Actually, let's just build a 10-node graph
                    pass
                    
        # Simpler: 10 squares in a line? No, too small.
        
        # Let's try a "spiral" or "snake"
        # 0 1 2 3 4
        # 9 8 7 6 5
        cons = []
        # Row 0: 0L1, 1L2, 2L3, 3L4
        for i in range(4):
            cons.append((i, i+1, 'L'))
        # Row 1: 5R6, 6R7, 7R8, 8R9 (i.e. 5 right of 6? No, 5 is at index 5. Let's say 5 is left of 6? No, snake goes back)
        # Positions:
        # 0: (0,0) 1:(0,1) 2:(0,2) 3:(0,3) 4:(0,4)
        # 9: (1,0) 8:(1,1) 7:(1,2) 6:(1,3) 5:(1,4)
        # Constraints:
        # 4 is above 5: (4, 5, 'T')
        # 5 is left of 6? No, 5 is at x=4, 6 is at x=3. So 6 is left of 5. (6, 5, 'L')
        # 6 is left of 7. (6, 7, 'L') -> Wait, 6 is at x=3, 7 is at x=2. So 7 is left of 6. (7, 6, 'L')
        # 7 is left of 8. (8, 7, 'L')
        # 8 is left of 9. (9, 8, 'L')
        # 9 is below 0? No, 9 is at (1,0), 0 is at (0,0). So 9 is below 0. (9, 0, 'B')
        
        snake_cons = [
            (0, 1, 'L'), (1, 2, 'L'), (2, 3, 'L'), (3, 4, 'L'),
            (4, 5, 'T'),
            (7, 6, 'L'), (8, 7, 'L'), (9, 8, 'L'),
            (9, 0, 'B')
        ]
        topologies.append(snake_cons)
        
        # Evaluate each topology
        for cons in topologies:
            res = _solve_layout_lp(cons)
            if res:
                centers, sides = res
                if _is_valid(centers, sides):
                    score = np.sum(sides)
                    if score > best_score:
                        best_score = score
                        best_result = (centers, sides)
                        
        return best_result

    def _optimize_centers(centers, max_time=30.0, method='SLSQP'):
        """
        Optimize centers to maximize sum of optimal sides.
        """
        x0 = np.concatenate([centers[:, 0], centers[:, 1]])
        bounds = [(0.0, 1.0)] * 20
        
        t0 = time.time()
        
        def _objective_centers(x):
            centers = np.column_stack([x[0:10], x[10:20]])
            s = _compute_optimal_sides(centers)
            return -np.sum(s)
        
        def callback(xk):
            if time.time() - t0 > max_time:
                raise RuntimeError("Time limit")
        
        try:
            if method == 'SLSQP':
                res = minimize(_objective_centers, x0, method='SLSQP', bounds=bounds,
                               options={'maxiter': 300, 'ftol': 1e-12, 'maxfun': 2000},
                               callback=callback)
            else:
                res = minimize(_objective_centers, x0, method='COBYLA',
                               options={'maxiter': 300, 'rhobeg': 0.05},
                               callback=callback)
            
            if res.success or (time.time() - t0 < max_time):
                x_opt = res.x
                new_centers = np.column_stack([x_opt[0:10], x_opt[10:20]])
                new_centers = np.clip(new_centers, 0.0, 1.0)
                new_sides = _compute_optimal_sides(new_centers)
                
                if _is_valid(new_centers, new_sides):
                    return new_centers, new_sides
        except Exception:
            pass
        
        return None

    def _repair_and_clamp(centers, sides):
        """
        Ensure validity by shrinking sides if necessary.
        """
        centers = centers.copy()
        sides = sides.copy()
        
        for i in range(10):
            h = sides[i] / 2.0
            centers[i, 0] = np.clip(centers[i, 0], h, 1.0 - h)
            centers[i, 1] = np.clip(centers[i, 1], h, 1.0 - h)
        
        for _ in range(300):
            overlap_found = False
            for i in range(10):
                for j in range(i+1, 10):
                    t = (sides[i] + sides[j]) / 2.0
                    dx = abs(centers[i,0] - centers[j,0])
                    dy = abs(centers[i,1] - centers[j,1])
                    if dx < t and dy < t:
                        overlap_found = True
                        if sides[i] < sides[j]:
                            sides[i] *= 0.85
                        else:
                            sides[j] *= 0.85
                        idx = i if sides[i] < sides[j] else j
                        h = sides[idx] / 2.0
                        centers[idx, 0] = np.clip(centers[idx, 0], h, 1.0 - h)
                        centers[idx, 1] = np.clip(centers[idx, 1], h, 1.0 - h)
            if not overlap_found:
                break
        
        return centers, sides

    def _update_best(c, s, best_holder):
        if _is_valid(c, s):
            score = np.sum(s)
            if score > best_holder[0]:
                best_holder[0] = score
                best_holder[1] = c.copy()
                best_holder[2] = s.copy()

    best_score = -1.0
    best_centers = None
    best_sides = None
    best_holder = [best_score, best_centers, best_sides]

    # 1. Combinatorial Search (New Structural Component)
    comb_res = _combinatorial_search()
    if comb_res:
        _update_best(comb_res[0], comb_res[1], best_holder)

    # 2. Heuristic Initializations
    def init_grid_33():
        c = np.zeros((10, 2))
        s = np.zeros(10)
        coords = [0.15, 0.5, 0.85]
        idx = 0
        for r in range(3):
            for col in range(3):
                c[idx, 0] = coords[col]
                c[idx, 1] = coords[r]
                s[idx] = 0.3
                idx += 1
        c[9, 0] = 0.5
        c[9, 1] = 0.5
        s[9] = 0.05
        return c, s

    def init_corner():
        c = np.zeros((10, 2))
        s = np.zeros(10)
        c[0] = [0.2, 0.2]; s[0] = 0.4
        c[1] = [0.8, 0.2]; s[1] = 0.4
        c[2] = [0.2, 0.8]; s[2] = 0.4
        c[3] = [0.8, 0.8]; s[3] = 0.4
        c[4] = [0.5, 0.1]; s[4] = 0.2
        c[5] = [0.1, 0.5]; s[5] = 0.2
        c[6] = [0.9, 0.5]; s[6] = 0.2
        c[7] = [0.5, 0.9]; s[7] = 0.2
        c[8] = [0.35, 0.35]; s[8] = 0.1
        c[9] = [0.65, 0.65]; s[9] = 0.1
        return c, s

    def init_stripe():
        c = np.zeros((10, 2))
        s = np.zeros(10)
        for i in range(10):
            row = i // 5
            col = i % 5
            c[i, 0] = 0.1 + col * 0.2
            c[i, 1] = 0.25 + row * 0.5
            s[i] = 0.18
        return c, s

    def init_diagonal():
        c = np.zeros((10, 2))
        s = np.zeros(10)
        for i in range(10):
            t = (i + 0.5) / 10.0
            c[i, 0] = t
            c[i, 1] = t
            s[i] = 0.25
        return c, s

    def init_random(seed):
        rng = np.random.default_rng(seed)
        c = rng.uniform(0.1, 0.9, size=(10, 2))
        s = rng.uniform(0.1, 0.4, size=10)
        return c, s

    def init_grid_44():
        c = np.zeros((10, 2))
        s = np.zeros(10)
        coords = [0.125, 0.375, 0.625, 0.875]
        idx = 0
        for r in range(4):
            for col in range(4):
                if idx < 10:
                    c[idx, 0] = coords[col]
                    c[idx, 1] = coords[r]
                    s[idx] = 0.24
                    idx += 1
        return c, s

    def init_2x5():
        c = np.zeros((10, 2))
        s = np.zeros(10)
        for i in range(10):
            row = i // 5
            col = i % 5
            c[i, 0] = 0.1 + col * 0.2
            c[i, 1] = 0.25 + row * 0.5
            s[i] = 0.19
        return c, s

    def init_5x2():
        c = np.zeros((10, 2))
        s = np.zeros(10)
        for i in range(10):
            col = i // 2
            row = i % 2
            c[i, 0] = 0.1 + col * 0.2
            c[i, 1] = 0.25 + row * 0.5
            s[i] = 0.19
        return c, s

    inits = [
        init_grid_33,
        init_corner,
        init_stripe,
        init_diagonal,
        init_grid_44,
        init_2x5,
        init_5x2,
        lambda: init_random(1),
        lambda: init_random(42),
        lambda: init_random(123),
        lambda: init_random(999),
        lambda: init_random(777),
        lambda: init_random(2024),
        lambda: init_random(555),
        lambda: init_random(314),
    ]
    
    for i, init_fn in enumerate(inits):
        if time.time() > deadline:
            break
        
        c, s = init_fn()
        c, s = _repair_and_clamp(c, s)
        _update_best(c, s, best_holder)
        
        # Optimize centers (20-dim)
        result = _optimize_centers(c, max_time=35.0, method='SLSQP')
        if result is not None:
            _update_best(result[0], result[1], best_holder)
        
        if time.time() > deadline:
            break
        
        # Second pass: COBYLA on centers
        if result is not None:
            c2, s2 = result
        else:
            c2, s2 = c, s
        
        result2 = _optimize_centers(c2, max_time=35.0, method='COBYLA')
        if result2 is not None:
            _update_best(result2[0], result2[1], best_holder)
        
        # Perturbation-based local search
        if best_holder[1] is not None and time.time() < deadline - 60:
            rng = np.random.default_rng(i * 100 + 1)
            c3 = best_holder[1] + rng.normal(0, 0.04, size=(10, 2))
            c3 = np.clip(c3, 0.0, 1.0)
            s3 = _compute_optimal_sides(c3)
            _update_best(c3, s3, best_holder)
            
            result4 = _optimize_centers(c3, max_time=25.0, method='SLSQP')
            if result4 is not None:
                _update_best(result4[0], result4[1], best_holder)

    # Final polish
    if best_holder[1] is not None and time.time() < deadline - 30:
        result_final = _optimize_centers(best_holder[1], max_time=25.0, method='SLSQP')
        if result_final is not None:
            _update_best(result_final[0], result_final[1], best_holder)
        
        # One more COBYLA polish
        if time.time() < deadline - 15:
            result_final2 = _optimize_centers(best_holder[1], max_time=15.0, method='COBYLA')
            if result_final2 is not None:
                _update_best(result_final2[0], result_final2[1], best_holder)

    if best_holder[1] is None:
        best_centers = np.zeros((10, 2))
        best_sides = np.zeros(10)
        for i in range(10):
            r = i // 5
            col = i % 5
            best_centers[i, 0] = 0.1 + col * 0.2
            best_centers[i, 1] = 0.1 + r * 0.2
            best_sides[i] = 0.1
        best_centers, best_sides = _repair_and_clamp(best_centers, best_sides)
        best_score = np.sum(best_sides)
    else:
        best_score = best_holder[0]
        best_centers = best_holder[1]
        best_sides = best_holder[2]

    return best_centers, best_sides

# EVOLVE-BLOCK-END
