import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
import math


def _is_feasible(centers, sides, tol=1e-7):
    """Check if the configuration is feasible."""
    n = len(sides)
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if s <= 0:
            return False
        if not (np.isfinite(cx) and np.isfinite(cy) and np.isfinite(s)):
            return False
        if cx - s/2 < -tol or cx + s/2 > 1 + tol:
            return False
        if cy - s/2 < -tol or cy + s/2 > 1 + tol:
            return False
    for i in range(n):
        for j in range(i+1, n):
            t = (sides[i] + sides[j])/2 - tol
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            if dx < t and dy < t:
                return False
    return True


class GuillotineNode:
    """A node in the guillotine cut tree."""
    def __init__(self, x, y, w, h, depth=0):
        self.x = x  # left
        self.y = y  # bottom
        self.w = w  # width
        self.h = h  # height
        self.depth = depth
        self.is_leaf = True
        self.square_idx = None
        self.side = 0.0
        self.cx = 0.0
        self.cy = 0.0
        self.left = None
        self.right = None
        self.split_axis = None  # 0 for vertical, 1 for horizontal
        self.split_pos = 0.0
    
    def area(self):
        return self.w * self.h
    
    def max_square_side(self):
        return min(self.w, self.h)
    
    def split(self, axis, pos):
        """Split this node into left/right (axis=0) or bottom/top (axis=1)."""
        if self.is_leaf:
            self.is_leaf = False
            self.left = GuillotineNode(self.x, self.y, pos, self.h, self.depth + 1)
            self.right = GuillotineNode(self.x + pos, self.y, self.w - pos, self.h, self.depth + 1)
            self.split_axis = 0
            self.split_pos = pos
        else:
            self.is_leaf = False
            self.left = GuillotineNode(self.x, self.y, self.w, pos, self.depth + 1)
            self.right = GuillotineNode(self.x, self.y + pos, self.w, self.h - pos, self.depth + 1)
            self.split_axis = 1
            self.split_pos = pos
        return self.left, self.right
    
    def assign_square(self, idx):
        """Assign a square to this leaf node."""
        if not self.is_leaf:
            raise ValueError("Cannot assign square to non-leaf node")
        self.square_idx = idx
        self.side = self.max_square_side()
        self.cx = self.x + self.w / 2
        self.cy = self.y + self.h / 2
    
    def get_squares(self, result):
        """Collect all squares from this subtree."""
        if self.is_leaf:
            if self.square_idx is not None:
                result[self.square_idx] = (self.cx, self.cy, self.side)
        else:
            self.left.get_squares(result)
            self.right.get_squares(result)
        return result


def _make_2big_8small():
    """Construct 2 large (0.5) + 8 small (0.25) tiling. Sum = 3.0."""
    centers = np.array([
        # 2 large squares at top
        [0.25, 0.75],  # large 1: [0,0.5]x[0.5,1]
        [0.75, 0.75],  # large 2: [0.5,1]x[0.5,1]
        # 8 small squares in 2x4 grid below
        [0.125, 0.125], [0.375, 0.125], [0.625, 0.125], [0.875, 0.125],
        [0.125, 0.375], [0.375, 0.375], [0.625, 0.375], [0.875, 0.375]
    ])
    sides = np.array([0.5, 0.5] + [0.25]*8)
    return centers, sides


def _guillotine_search(n_squares=10, deadline=None, max_depth=15):
    """
    Search over guillotine cut tree topologies to maximize sum of square sides.
    Uses branch-and-bound with heuristic pruning.
    """
    best_sum = 0.0
    best_tree = None
    
    def _search(node, remaining_squares, current_sum, depth):
        nonlocal best_sum, best_tree
        
        if deadline is not None and time.time() > deadline:
            return
        
        if not remaining_squares:
            return
        
        if depth >= max_depth:
            return
        
        # If only one square left, assign it to this node
        if len(remaining_squares) == 1:
            idx = remaining_squares[0]
            node.assign_square(idx)
            total = current_sum + node.side
            if total > best_sum:
                best_sum = total
                best_tree = node
            node.is_leaf = True
            node.square_idx = None
            return
        
        # Try both split axes
        for axis in [0, 1]:
            # Try various split positions
            if axis == 0:
                # Vertical split: left gets some width
                for frac in np.linspace(0.2, 0.8, 7):
                    pos = node.w * frac
                    left_w = pos
                    right_w = node.w - pos
                    
                    if left_w < 0.01 or right_w < 0.01:
                        continue
                    
                    # Heuristic: estimate max possible sum from this split
                    # Left can fit at most min(left_w, node.h) per square
                    # Right can fit at most min(right_w, node.h) per square
                    est_left_max = min(left_w, node.h) * len(remaining_squares)
                    est_right_max = min(right_w, node.h) * len(remaining_squares)
                    est_total = current_sum + est_left_max + est_right_max
                    
                    if est_total <= best_sum:
                        continue
                    
                    # Actually create the split
                    left, right = node.split(axis, pos)
                    
                    # Recursively search left and right
                    # We need to partition remaining squares between left and right
                    # Try different partitions
                    for left_count in range(1, len(remaining_squares)):
                        if deadline is not None and time.time() > deadline:
                            break
                        
                        # Try assigning first left_count squares to left, rest to right
                        left_sq = list(remaining_squares[:left_count])
                        right_sq = list(remaining_squares[left_count:])
                        
                        # Recurse on left
                        _search(left, left_sq, current_sum, depth + 1)
                        
                        # Recurse on right
                        _search(right, right_sq, current_sum, depth + 1)
                    
                    # Reset node
                    node.is_leaf = True
                    node.left = None
                    node.right = None
                    node.square_idx = None
                    node.side = 0.0
            else:
                # Horizontal split: bottom gets some height
                for frac in np.linspace(0.2, 0.8, 7):
                    pos = node.h * frac
                    bottom_h = pos
                    top_h = node.h - pos
                    
                    if bottom_h < 0.01 or top_h < 0.01:
                        continue
                    
                    est_bottom_max = min(node.w, bottom_h) * len(remaining_squares)
                    est_top_max = min(node.w, top_h) * len(remaining_squares)
                    est_total = current_sum + est_bottom_max + est_top_max
                    
                    if est_total <= best_sum:
                        continue
                    
                    left, right = node.split(axis, pos)
                    
                    for left_count in range(1, len(remaining_squares)):
                        if deadline is not None and time.time() > deadline:
                            break
                        
                        left_sq = list(remaining_squares[:left_count])
                        right_sq = list(remaining_squares[left_count:])
                        
                        _search(left, left_sq, current_sum, depth + 1)
                        _search(right, right_sq, current_sum, depth + 1)
                    
                    node.is_leaf = True
                    node.left = None
                    node.right = None
                    node.square_idx = None
                    node.side = 0.0
    
    # Start with the full unit square
    root = GuillotineNode(0, 0, 1, 1)
    all_squares = list(range(n_squares))
    _search(root, all_squares, 0.0, 0)
    
    if best_tree is None:
        return None, None
    
    # Extract squares from the best tree
    result = {}
    best_tree.get_squares(result)
    
    centers = np.zeros((n_squares, 2))
    sides = np.zeros(n_squares)
    
    for i in range(n_squares):
        if i in result:
            cx, cy, s = result[i]
            centers[i] = [cx, cy]
            sides[i] = s
        else:
            # Should not happen, but just in case
            centers[i] = [0.5, 0.5]
            sides[i] = 0.01
    
    return centers, sides


def _make_3big_7small():
    """Try 3 large squares + 7 smaller ones. Sum potentially > 3.0."""
    # 3 squares of size ~0.45 in a row at bottom, 7 smaller above
    s_big = 0.45
    s_small = 0.32
    centers = np.array([
        [0.225, 0.225], [0.5, 0.225], [0.775, 0.225],  # 3 big at bottom
        [0.15, 0.65], [0.35, 0.65], [0.55, 0.65], [0.75, 0.65],  # 4 small top row
        [0.25, 0.88], [0.5, 0.88], [0.75, 0.88]  # 3 small top-top
    ])
    sides = np.array([s_big]*3 + [s_small]*7)
    return centers, sides


def _guillotine_refine(centers, sides, deadline=None, n_iter=100):
    """
    Refine a guillotine-based packing by adjusting split positions and square assignments.
    """
    if deadline is None:
        return centers, sides
    
    n = len(sides)
    best_centers = centers.copy()
    best_sides = sides.copy()
    best_val = np.sum(sides)
    
    rng = np.random.default_rng(42)
    
    for iteration in range(n_iter):
        if time.time() > deadline:
            break
        
        # Try moving one square slightly and growing it
        i = rng.integers(n)
        s = best_sides[i]
        cx, cy = best_centers[i]
        
        # Try small perturbations
        for _ in range(30):
            dx = rng.uniform(-0.02, 0.02)
            dy = rng.uniform(-0.02, 0.02)
            ds = rng.uniform(-0.01, 0.03)
            
            new_s = np.clip(s + ds, 0.01, 1.0)
            new_cx = np.clip(cx + dx, new_s/2, 1 - new_s/2)
            new_cy = np.clip(cy + dy, new_s/2, 1 - new_s/2)
            
            new_centers = best_centers.copy()
            new_sides = best_sides.copy()
            new_centers[i] = [new_cx, new_cy]
            new_sides[i] = new_s
            
            if _is_feasible(new_centers, new_sides):
                new_val = np.sum(new_sides)
                if new_val > best_val + 1e-10:
                    best_centers = new_centers
                    best_sides = new_sides
                    best_val = new_val
    
    return best_centers, best_sides


def _make_mixed_packing():
    """Mixed packing with varied sizes."""
    # Try a more organic arrangement
    centers = np.array([
        [0.2, 0.2], [0.8, 0.2], [0.5, 0.15],  # bottom row
        [0.2, 0.6], [0.8, 0.6], [0.5, 0.55],  # middle row  
        [0.2, 0.85], [0.5, 0.85], [0.8, 0.85], [0.5, 0.7]  # top
    ])
    sides = np.array([0.35, 0.35, 0.3, 0.35, 0.35, 0.3, 0.3, 0.3, 0.3, 0.25])
    return centers, sides


def _try_guillotine_config(deadline):
    """Try to build a good guillotine packing and return it if feasible."""
    if deadline is not None and time.time() > deadline:
        return None, None
    
    # Try guillotine search
    centers, sides = _guillotine_search(10, deadline=min(deadline, time.time() + 30))
    
    if centers is None:
        return None, None
    
    # Check feasibility
    if not _is_feasible(centers, sides):
        return None, None
    
    # Refine
    centers, sides = _guillotine_refine(centers, sides, deadline=deadline)
    
    if _is_feasible(centers, sides):
        return centers, sides
    
    return None, None


def _move_and_grow(centers, sides, tol=1e-7, deadline=None, max_iter=500):
    """
    Local search that can both move and grow squares.
    Tries random perturbations of position and size.
    """
    n = len(sides)
    best_centers = centers.copy()
    best_sides = sides.copy()
    best_val = np.sum(sides)
    
    rng = np.random.default_rng(42)
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        # Try modifying one square at a time
        for i in range(n):
            # Generate several random perturbations
            for attempt in range(20):
                # Random size change
                size_factor = 1.0 + rng.uniform(-0.05, 0.15)
                new_s = best_sides[i] * size_factor
                new_s = np.clip(new_s, 1e-6, 1.0)
                
                # Random position change
                dx = rng.uniform(-0.05, 0.05)
                dy = rng.uniform(-0.05, 0.05)
                
                new_centers = best_centers.copy()
                new_sides = best_sides.copy()
                new_sides[i] = new_s
                new_cx = np.clip(best_centers[i, 0] + dx, new_s/2, 1 - new_s/2)
                new_cy = np.clip(best_centers[i, 1] + dy, new_s/2, 1 - new_s/2)
                new_centers[i] = [new_cx, new_cy]
                
                if _is_feasible(new_centers, new_sides, tol):
                    new_val = np.sum(new_sides)
                    if new_val > best_val + 1e-10:
                        best_centers = new_centers
                        best_sides = new_sides
                        best_val = new_val
                        improved = True
                        break
        
        if not improved and iteration > 100:
            # Reduce perturbation size
            break
        if not improved and iteration > 300:
            break
    
    return best_centers, best_sides


def _simulated_annealing(centers, sides, tol=1e-7, deadline=None):
    """
    Simulated annealing to escape local optima.
    """
    n = len(sides)
    current_centers = centers.copy()
    current_sides = sides.copy()
    current_val = np.sum(sides)
    
    best_centers = current_centers.copy()
    best_sides = current_sides.copy()
    best_val = current_val
    
    rng = np.random.default_rng(123)
    
    T_start = 0.1
    T_end = 0.001
    n_iter = 2000
    
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        T = T_start * (T_end / T_start) ** (it / n_iter)
        
        # Pick a random square to modify
        i = rng.integers(n)
        
        # Random perturbation
        size_change = rng.uniform(-0.03, 0.05)
        dx = rng.uniform(-0.03, 0.03)
        dy = rng.uniform(-0.03, 0.03)
        
        new_s = np.clip(current_sides[i] + size_change, 1e-6, 1.0)
        new_cx = np.clip(current_centers[i, 0] + dx, new_s/2, 1 - new_s/2)
        new_cy = np.clip(current_centers[i, 1] + dy, new_s/2, 1 - new_s/2)
        
        new_centers = current_centers.copy()
        new_sides = current_sides.copy()
        new_sides[i] = new_s
        new_centers[i] = [new_cx, new_cy]
        
        if _is_feasible(new_centers, new_sides, tol):
            new_val = np.sum(new_sides)
            delta_val = new_val - current_val
            
            # Accept if better or with probability exp(-delta/T) for worse
            if delta_val > 0 or rng.random() < np.exp(-delta_val / max(T, 1e-10)):
                current_centers = new_centers
                current_sides = new_sides
                current_val = new_val
                
                if current_val > best_val:
                    best_centers = current_centers.copy()
                    best_sides = current_sides.copy()
                    best_val = current_val
    
    return best_centers, best_sides


def _coordinate_ascent_v2(centers, sides, tol=1e-7, deadline=None):
    """
    Improved coordinate ascent that considers moving squares.
    """
    n = len(sides)
    best_centers = centers.copy()
    best_sides = sides.copy()
    
    for iteration in range(200):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        
        for i in range(n):
            current_s = best_sides[i]
            cx, cy = best_centers[i]
            
            # Try to grow square i by finding max possible size
            max_s = 2 * min(cx, 1 - cx, cy, 1 - cy)
            
            for j in range(n):
                if i == j:
                    continue
                oj_cx, oj_cy = best_centers[j]
                oj_s = best_sides[j]
                
                dx = abs(cx - oj_cx)
                dy = abs(cy - oj_cy)
                
                # Max size before overlapping in x or y
                # If we grow i, the half-width becomes s/2
                # Overlap in x when: cx_i - s_i/2 < oj_cx + oj_s/2 AND cx_i + s_i/2 > oj_cx - oj_s/2
                # i.e., s_i > 2*dx - oj_s (if cx_i > oj_cx) or similar
                
                # Simpler: just check if growing to max_s would cause overlap
                # and reduce accordingly
                
                # Distance from edge of i to edge of j in x
                if cx >= oj_cx:
                    sep_x = cx - oj_cx - (current_s + oj_s)/2
                    # Can grow by at most sep_x in x direction
                    max_from_j_x = current_s + 2*sep_x if sep_x > 0 else current_s
                else:
                    sep_x = oj_cx - cx - (current_s + oj_s)/2
                    max_from_j_x = current_s + 2*sep_x if sep_x > 0 else current_s
                
                if cy >= oj_cy:
                    sep_y = cy - oj_cy - (current_s + oj_s)/2
                    max_from_j_y = current_s + 2*sep_y if sep_y > 0 else current_s
                else:
                    sep_y = oj_cy - cy - (current_s + oj_s)/2
                    max_from_j_y = current_s + 2*sep_y if sep_y > 0 else current_s
                
                max_s = min(max_s, max_from_j_x, max_from_j_y)
            
            # Also try moving the square to find more space
            # Try shifting in 4 directions
            for dcx in [-0.02, 0.02]:
                for dcy in [-0.02, 0.02]:
                    new_cx = np.clip(cx + dcx, current_s/2, 1 - current_s/2)
                    new_cy = np.clip(cy + dcy, current_s/2, 1 - current_s/2)
                    
                    # Recalculate max size from new position
                    max_s_try = 2 * min(new_cx, 1 - new_cx, new_cy, 1 - new_cy)
                    
                    for j in range(n):
                        if i == j:
                            continue
                        oj_cx, oj_cy = best_centers[j]
                        oj_s = best_sides[j]
                        
                        dx = abs(new_cx - oj_cx)
                        dy = abs(new_cy - oj_cy)
                        
                        if new_cx >= oj_cx:
                            sep_x = new_cx - oj_cx - (current_s + oj_s)/2
                            max_from_j_x = current_s + 2*sep_x if sep_x > 0 else current_s
                        else:
                            sep_x = oj_cx - new_cx - (current_s + oj_s)/2
                            max_from_j_x = current_s + 2*sep_x if sep_x > 0 else current_s
                        
                        if new_cy >= oj_cy:
                            sep_y = new_cy - oj_cy - (current_s + oj_s)/2
                            max_from_j_y = current_s + 2*sep_y if sep_y > 0 else current_s
                        else:
                            sep_y = oj_cy - new_cy - (current_s + oj_s)/2
                            max_from_j_y = current_s + 2*sep_y if sep_y > 0 else current_s
                        
                        max_s_try = min(max_s_try, max_from_j_x, max_from_j_y)
                    
                    if max_s_try > max_s:
                        max_s = max_s_try
                        cx, cy = new_cx, new_cy
            
            if max_s > current_s + 1e-8:
                new_s = max_s
                new_sides = best_sides.copy()
                new_sides[i] = new_s
                new_centers = best_centers.copy()
                
                new_cx = np.clip(cx, new_s/2, 1 - new_s/2)
                new_cy = np.clip(cy, new_s/2, 1 - new_s/2)
                new_centers[i] = [new_cx, new_cy]
                
                if _is_feasible(new_centers, new_sides, tol):
                    best_centers = new_centers
                    best_sides = new_sides
                    improved = True
        
        if not improved:
            break
    
    return best_centers, best_sides


def _differential_evolution_optimize(deadline):
    """
    Use differential evolution with a feasibility-based fitness.
    """
    n = 10
    
    # Bounds: cx, cy in [0,1], s in [0.01, 0.6]
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))  # cx
    for i in range(n):
        bounds.append((0.0, 1.0))  # cy
    for i in range(n):
        bounds.append((0.01, 0.6))  # s
    
    def objective(x):
        centers = np.array([[x[i], x[n+i]] for i in range(n)])
        sides = x[2*n:3*n]
        
        # Check feasibility
        if not _is_feasible(centers, sides):
            # Penalty for infeasibility
            penalty = 0
            for i in range(n):
                cx, cy = centers[i]
                s = sides[i]
                if cx - s/2 < 0:
                    penalty += abs(cx - s/2)
                if cx + s/2 > 1:
                    penalty += abs(cx + s/2 - 1)
                if cy - s/2 < 0:
                    penalty += abs(cy - s/2)
                if cy + s/2 > 1:
                    penalty += abs(cy + s/2 - 1)
            for i in range(n):
                for j in range(i+1, n):
                    t = (sides[i] + sides[j])/2
                    dx = abs(centers[i, 0] - centers[j, 0])
                    dy = abs(centers[i, 1] - centers[j, 1])
                    if dx < t and dy < t:
                        penalty += min(t - dx, t - dy)
            return -np.sum(sides) + 100 * penalty
        
        return -np.sum(sides)
    
    # Start from a good initial point
    c0, s0 = _make_2big_8small()
    x0 = np.concatenate([c0.flatten(), s0])
    
    result = differential_evolution(
        objective, bounds, 
        x0=x0,
        seed=42,
        maxiter=100,
        tol=1e-8,
        polish=True,
        workers=1,
        init='sobol'
    )
    
    if np.all(np.isfinite(result.x)):
        centers = np.array([[result.x[i], result.x[n+i]] for i in range(n)])
        sides = result.x[2*n:3*n]
        
        # Clip to valid range
        for i in range(n):
            sides[i] = max(1e-6, sides[i])
            centers[i, 0] = np.clip(centers[i, 0], sides[i]/2, 1 - sides[i]/2)
            centers[i, 1] = np.clip(centers[i, 1], sides[i]/2, 1 - sides[i]/2)
        
        if _is_feasible(centers, sides):
            return centers, sides
    
    return None, None


def squares_in_square10():
    start_time = time.time()
    deadline = start_time + 580
    
    best_centers = None
    best_sides = None
    best_val = -np.inf
    
    # Try multiple initial configurations
    configs = [
        _make_2big_8small(),
        _make_3big_7small(),
        _make_mixed_packing()
    ]
    
    for c, s in configs:
        if not _is_feasible(c, s):
            continue
        val = np.sum(s)
        if val > best_val:
            best_val = val
            best_centers = c.copy()
            best_sides = s.copy()
    
    if best_centers is None:
        best_centers, best_sides = _make_2big_8small()
        best_val = np.sum(best_sides)
    
    # Try guillotine approach
    if time.time() < deadline - 200:
        try:
            g_c, g_s = _try_guillotine_config(deadline - 150)
            if g_c is not None and _is_feasible(g_c, g_s):
                g_val = np.sum(g_s)
                if g_val > best_val:
                    best_val = g_val
                    best_centers = g_c
                    best_sides = g_s
        except Exception:
            pass
    
    # Coordinate ascent v2 (with movement)
    if time.time() < deadline - 180:
        try:
            ca_c, ca_s = _coordinate_ascent_v2(
                best_centers, best_sides, deadline=deadline - 150
            )
            if _is_feasible(ca_c, ca_s):
                ca_val = np.sum(ca_s)
                if ca_val > best_val:
                    best_val = ca_val
                    best_centers = ca_c
                    best_sides = ca_s
        except Exception:
            pass
    
    # Move and grow local search
    if time.time() < deadline - 120:
        try:
            mg_c, mg_s = _move_and_grow(
                best_centers, best_sides, deadline=deadline - 100
            )
            if _is_feasible(mg_c, mg_s):
                mg_val = np.sum(mg_s)
                if mg_val > best_val:
                    best_val = mg_val
                    best_centers = mg_c
                    best_sides = mg_s
        except Exception:
            pass
    
    # Simulated annealing
    if time.time() < deadline - 80:
        try:
            sa_c, sa_s = _simulated_annealing(
                best_centers, best_sides, deadline=deadline - 60
            )
            if _is_feasible(sa_c, sa_s):
                sa_val = np.sum(sa_s)
                if sa_val > best_val:
                    best_val = sa_val
                    best_centers = sa_c
                    best_sides = sa_s
        except Exception:
            pass
    
    # Differential evolution
    if time.time() < deadline - 40:
        try:
            de_c, de_s = _differential_evolution_optimize(deadline - 20)
            if de_c is not None and _is_feasible(de_c, de_s):
                de_val = np.sum(de_s)
                if de_val > best_val:
                    best_val = de_val
                    best_centers = de_c
                    best_sides = de_s
        except Exception:
            pass
    
    # Final coordinate ascent pass
    if time.time() < deadline - 10:
        try:
            final_c, final_s = _coordinate_ascent_v2(
                best_centers, best_sides, deadline=deadline - 5
            )
            if _is_feasible(final_c, final_s):
                final_val = np.sum(final_s)
                if final_val > best_val:
                    best_val = final_val
                    best_centers = final_c
                    best_sides = final_s
        except Exception:
            pass
    
    return best_centers, best_sides
