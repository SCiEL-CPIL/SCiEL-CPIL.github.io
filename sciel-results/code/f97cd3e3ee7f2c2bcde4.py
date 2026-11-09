import numpy as np
import time
import random
from scipy.optimize import linprog

def _is_feasible(centers, sides, tol=1e-7):
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

def _solve_topology_lp(assignment, tol=1e-7, max_time=1.0):
    """
    Solve LP for a given topology assignment.
    assignment: dict mapping (i,j) -> 0 (x-sep) or 1 (y-sep)
    Returns (centers, sides, value) or None if infeasible.
    """
    n = 10
    # Variables: x_i, y_i, s_i for i=0..9 => 30 variables
    # x_i = center x, y_i = center y, s_i = side length
    # Index: 3*i = x_i, 3*i+1 = y_i, 3*i+2 = s_i
    
    # Objective: maximize sum(s_i) => minimize -sum(s_i)
    c = np.zeros(3*n)
    for i in range(n):
        c[3*i+2] = -1.0
    
    A_ub = []
    b_ub = []
    
    # Boundary constraints:
    # x_i - s_i/2 >= 0  =>  x_i - s_i/2 >= 0
    # x_i + s_i/2 <= 1  =>  x_i + s_i/2 <= 1
    # y_i - s_i/2 >= 0  =>  y_i - s_i/2 >= 0
    # y_i + s_i/2 <= 1  =>  y_i + s_i/2 <= 1
    # s_i > 0 (handled by bounds)
    
    for i in range(n):
        # x_i - s_i/2 >= 0  =>  -x_i + s_i/2 <= 0
        row = np.zeros(3*n)
        row[3*i] = -1.0
        row[3*i+2] = 0.5
        A_ub.append(row)
        b_ub.append(0.0)
        
        # x_i + s_i/2 <= 1
        row = np.zeros(3*n)
        row[3*i] = 1.0
        row[3*i+2] = 0.5
        A_ub.append(row)
        b_ub.append(1.0)
        
        # y_i - s_i/2 >= 0  =>  -y_i + s_i/2 <= 0
        row = np.zeros(3*n)
        row[3*i+1] = -1.0
        row[3*i+2] = 0.5
        A_ub.append(row)
        b_ub.append(0.0)
        
        # y_i + s_i/2 <= 1
        row = np.zeros(3*n)
        row[3*i+1] = 1.0
        row[3*i+2] = 0.5
        A_ub.append(row)
        b_ub.append(1.0)
    
    # Non-overlap constraints based on topology
    # For pair (i,j) with x-sep (assignment=0):
    #   Either x_i - s_i/2 >= x_j + s_j/2  =>  x_i - x_j - (s_i+s_j)/2 >= 0
    #   Or   x_i + s_i/2 <= x_j - s_j/2  =>  x_j - x_i - (s_i+s_j)/2 >= 0
    # We need to choose one direction. Let's use a big-M or enumerate directions.
    # Actually, for a fixed topology, we need to fix which side each square is on.
    # Let's use auxiliary binary-like constraints via the assignment.
    # For simplicity, let's assume a specific ordering. We'll use:
    # For x-sep: x_i - x_j >= (s_i+s_j)/2  (i is to the right of j)
    # OR x_j - x_i >= (s_i+s_j)/2  (j is to the right of i)
    # This requires another binary choice. Let's enumerate both or use a trick.
    # For now, let's just try one direction and if infeasible, try the other.
    # Actually, let's use the standard approach: for each pair, we have 4 possible
    # separations (x+, x-, y+, y-). The assignment picks the axis, but we also
    # need the sign. Let's extend the assignment to include sign.
    
    # Let's redefine: assignment[(i,j)] = 0: x_i > x_j, 1: x_j > x_i, 
    #                2: y_i > y_j, 3: y_j > y_i
    
    for i in range(n):
        for j in range(i+1, n):
            sep = assignment[(i,j)]
            if sep == 0:  # x_i - x_j >= (s_i+s_j)/2
                row = np.zeros(3*n)
                row[3*i] = 1.0
                row[3*j] = -1.0
                row[3*i+2] = -0.5
                row[3*j+2] = -0.5
                A_ub.append(-row)  # -x_i + x_j + (s_i+s_j)/2 <= 0
                b_ub.append(0.0)
            elif sep == 1:  # x_j - x_i >= (s_i+s_j)/2
                row = np.zeros(3*n)
                row[3*j] = 1.0
                row[3*i] = -1.0
                row[3*i+2] = -0.5
                row[3*j+2] = -0.5
                A_ub.append(-row)  # -x_j + x_i + (s_i+s_j)/2 <= 0
                b_ub.append(0.0)
            elif sep == 2:  # y_i - y_j >= (s_i+s_j)/2
                row = np.zeros(3*n)
                row[3*i+1] = 1.0
                row[3*j+1] = -1.0
                row[3*i+2] = -0.5
                row[3*j+2] = -0.5
                A_ub.append(-row)
                b_ub.append(0.0)
            elif sep == 3:  # y_j - y_i >= (s_i+s_j)/2
                row = np.zeros(3*n)
                row[3*j+1] = 1.0
                row[3*i+1] = -1.0
                row[3*i+2] = -0.5
                row[3*j+2] = -0.5
                A_ub.append(-row)
                b_ub.append(0.0)
    
    if A_ub:
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
    else:
        A_ub = np.empty((0, 3*n))
        b_ub = np.empty(0)
    
    # Bounds: 0 <= x_i, y_i <= 1, 0 <= s_i <= 1
    bounds = []
    for i in range(n):
        bounds.append((0, 1))  # x_i
        bounds.append((0, 1))  # y_i
        bounds.append((1e-9, 1))  # s_i
    
    start_t = time.time()
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if time.time() - start_t > max_time:
        return None
    
    if res.status != 0:
        return None
    
    x = res.x
    centers = np.array([[x[3*i], x[3*i+1]] for i in range(n)])
    sides = np.array([x[3*i+2] for i in range(n)])
    
    if not _is_feasible(centers, sides):
        return None
    
    return centers, sides, np.sum(sides)

def _max_side_given_others(centers, sides, i, tol=1e-7):
    """Calculate max side for square i given others are fixed."""
    n = len(sides)
    cx, cy = centers[i]
    s = sides[i]
    
    # Boundary constraints
    max_s_boundary = min(2*cx, 2*(1-cx), 2*cy, 2*(1-cy))
    
    max_s = max_s_boundary
    for j in range(n):
        if j == i: continue
        cx_j, cy_j = centers[j]
        s_j = sides[j]
        
        # Distance to center of j
        dx = abs(cx - cx_j)
        dy = abs(cy - cy_j)
        
        # We need |cx - cx_j| >= (s+s_j)/2 OR |cy - cy_j| >= (s+s_j)/2
        # This means either s <= 2*dx - s_j OR s <= 2*dy - s_j
        # So s <= min(2*dx - s_j, 2*dy - s_j) is NOT correct.
        # It is s <= (2*dx - s_j) IF we separate in x.
        # OR s <= (2*dy - s_j) IF we separate in y.
        # We can choose the better separation.
        # So the constraint is s <= max(2*dx - s_j, 2*dy - s_j) ? No.
        # If we separate in x, we need cx - s/2 >= cx_j + s_j/2 OR cx + s/2 <= cx_j - s_j/2.
        # This implies s <= 2*dx - s_j.
        # Similarly for y.
        # Since we can choose to separate in x OR y, the max allowed s is the max of the two limits?
        # Wait. If s is large, we might not be able to separate in x. But we can separate in y.
        # So the condition is: (2*dx - s_j >= s) OR (2*dy - s_j >= s).
        # This is equivalent to: s <= max(2*dx - s_j, 2*dy - s_j).
        # Let's verify.
        # If s <= 2*dx - s_j, we can separate in x.
        # If s <= 2*dy - s_j, we can separate in y.
        # If either is true, we are feasible.
        # So the max s allowed by pair (i,j) is indeed max(2*dx - s_j, 2*dy - s_j).
        
        limit = max(2*dx - s_j, 2*dy - s_j)
        if limit < max_s:
            max_s = limit
            
    return max(0.0, max_s)

def _local_refine(centers, sides, tol=1e-7, max_iter=1000, deadline=None):
    n = len(sides)
    best_centers = centers.copy()
    best_sides = sides.copy()
    best_val = np.sum(sides)
    
    if deadline is None:
        deadline = time.time() + 100
        
    for iteration in range(max_iter):
        if time.time() > deadline - 5:
            break
            
        improved = False
        
        # 1. Grow each square to max possible size
        for i in range(n):
            new_s = _max_side_given_others(best_centers, best_sides, i, tol)
            if new_s > best_sides[i] + 1e-9:
                best_sides[i] = new_s
                improved = True
        
        # 2. Try to move squares to improve packing
        for i in range(n):
            s = best_sides[i]
            cx, cy = best_centers[i]
            
            for dx in [-0.05, 0.05]:
                for dy in [-0.05, 0.05]:
                    new_cx = cx + dx
                    new_cy = cy + dy
                    if new_cx - s/2 < 0 or new_cx + s/2 > 1: continue
                    if new_cy - s/2 < 0 or new_cy + s/2 > 1: continue
                    
                    new_centers = best_centers.copy()
                    new_centers[i] = [new_cx, new_cy]
                    
                    sim_sides = best_sides.copy()
                    sim_centers = new_centers.copy()
                    
                    for k in range(n):
                        sim_sides[k] = _max_side_given_others(sim_centers, sim_sides, k, tol)
                        
                    if np.sum(sim_sides) > best_val + 1e-7:
                        best_centers = sim_centers
                        best_sides = sim_sides
                        best_val = np.sum(best_sides)
                        improved = True
                        break
                if improved: break
            if improved: break
            
        if not improved:
            break
            
    return best_centers, best_sides

def _enumerate_topologies(seed=0, max_topologies=100):
    """
    Generate diverse valid topologies by random assignment and checking acyclicity.
    Returns list of assignment dicts.
    """
    random.seed(seed)
    n = 10
    pairs = [(i,j) for i in range(n) for j in range(i+1, n)]
    topologies = []
    seen = set()
    
    for _ in range(10000):
        if len(topologies) >= max_topologies:
            break
        
        # Random assignment: 0=x+, 1=x-, 2=y+, 3=y-
        assignment = {}
        for (i,j) in pairs:
            assignment[(i,j)] = random.randint(0, 3)
        
        # Check for cycles in the constraint graph
        # Build directed graph: edge i->j if constraint says i is to the right/above j
        # x+: i > j in x => edge i->j
        # x-: j > i in x => edge j->i
        # y+: i > j in y => edge i->j
        # y-: j > i in y => edge j->i
        # But x and y are independent dimensions, so cycles only matter within each dimension?
        # Actually, for LP feasibility, we just need the constraints to be consistent.
        # A cycle in x-constraints would mean x_1 > x_2 > ... > x_1, which is infeasible.
        # So we need to check for cycles in the x-subgraph and y-subgraph separately.
        
        # Build x-graph and y-graph
        x_edges = []
        y_edges = []
        for (i,j) in pairs:
            sep = assignment[(i,j)]
            if sep == 0:  # x_i > x_j
                x_edges.append((i,j))
            elif sep == 1:  # x_j > x_i
                x_edges.append((j,i))
            elif sep == 2:  # y_i > y_j
                y_edges.append((i,j))
            elif sep == 3:  # y_j > y_i
                y_edges.append((j,i))
        
        # Check for cycles using DFS
        def has_cycle(edges, n):
            adj = [[] for _ in range(n)]
            for (u,v) in edges:
                adj[u].append(v)
            
            WHITE, GRAY, BLACK = 0, 1, 2
            color = [WHITE] * n
            
            def dfs(u):
                color[u] = GRAY
                for v in adj[u]:
                    if color[v] == GRAY:
                        return True
                    if color[v] == WHITE and dfs(v):
                        return True
                color[u] = BLACK
                return False
            
            for u in range(n):
                if color[u] == WHITE:
                    if dfs(u):
                        return True
            return False
        
        if has_cycle(x_edges, n) or has_cycle(y_edges, n):
            continue
        
        # Check uniqueness
        key = tuple(sorted(assignment.items()))
        if key in seen:
            continue
        seen.add(key)
        
        topologies.append(assignment)
    
    return topologies

def _gap_filling_pack(random_seed=0):
    """
    Construct a packing using a greedy gap-filling heuristic.
    Returns centers and sides.
    """
    random.seed(random_seed)
    n = 10
    centers = np.zeros((n, 2))
    sides = np.zeros(n)
    
    # List of empty rectangles: (x_min, y_min, x_max, y_max)
    empty_rects = [(0.0, 0.0, 1.0, 1.0)]
    
    for k in range(n):
        best_rect_idx = -1
        best_size = 0
        best_cx = 0
        best_cy = 0
        
        for idx, (x1, y1, x2, y2) in enumerate(empty_rects):
            w = x2 - x1
            h = y2 - y1
            s = min(w, h)
            if s > best_size:
                best_size = s
                best_rect_idx = idx
                best_cx = (x1 + x2) / 2
                best_cy = (y1 + y2) / 2
            elif s == best_size and s > 0:
                # Tie-breaking: randomize to explore different packings
                if random.random() < 0.5:
                    best_size = s
                    best_rect_idx = idx
                    best_cx = (x1 + x2) / 2
                    best_cy = (y1 + y2) / 2
        
        if best_rect_idx == -1:
            break
            
        centers[k] = [best_cx, best_cy]
        sides[k] = best_size
        
        # Update empty rectangles
        x1, y1, x2, y2 = empty_rects[best_rect_idx]
        cx, cy = centers[k]
        s = sides[k]
        
        new_rects = []
        # Left
        if cx - s/2 > x1:
            new_rects.append((x1, y1, cx - s/2, y2))
        # Right
        if cx + s/2 < x2:
            new_rects.append((cx + s/2, y1, x2, y2))
        # Bottom
        if cy - s/2 > y1:
            new_rects.append((cx - s/2, y1, cx + s/2, cy - s/2))
        # Top
        if cy + s/2 < y2:
            new_rects.append((cx - s/2, cy + s/2, cx + s/2, y2))
        
        # Replace the used rectangle with the new ones
        empty_rects[best_rect_idx] = None # Mark for removal
        empty_rects = [r for r in empty_rects if r is not None]
        empty_rects.extend(new_rects)
        
        # Optional: Merge adjacent rectangles? Not strictly necessary for N=10.
        
    return centers, sides

def squares_in_square10():
    start_time = time.time()
    deadline = start_time + 575
    
    best_centers = None
    best_sides = None
    best_val = -np.inf
    
    # 1. Try the known good 3.0 packing
    c0 = np.array([
        [0.25, 0.75],
        [0.75, 0.75],
        [0.125, 0.125], [0.375, 0.125], [0.625, 0.125], [0.875, 0.125],
        [0.125, 0.375], [0.375, 0.375], [0.625, 0.375], [0.875, 0.375]
    ])
    s0 = np.array([0.5, 0.5] + [0.25]*8)
    if _is_feasible(c0, s0):
        best_centers = c0.copy()
        best_sides = s0.copy()
        best_val = np.sum(s0)
    
    # 2. Try topology enumeration with LP
    if time.time() < deadline - 60:
        try:
            topologies = _enumerate_topologies(seed=42, max_topologies=50)
            for assignment in topologies:
                if time.time() > deadline - 30:
                    break
                result = _solve_topology_lp(assignment, max_time=2.0)
                if result is not None:
                    centers, sides, val = result
                    if val > best_val:
                        best_val = val
                        best_centers = centers.copy()
                        best_sides = sides.copy()
        except Exception:
            pass
    
    # 3. Try multiple gap-filling packings with different seeds
    num_restarts = 10
    for i in range(num_restarts):
        if time.time() > deadline - 30:
            break
            
        c, s = _gap_filling_pack(random_seed=i)
        
        # Refine this packing
        refine_deadline = min(deadline - 10, time.time() + 10)
        try:
            rc, rs = _local_refine(c, s, max_iter=200, deadline=refine_deadline)
            if _is_feasible(rc, rs):
                val = np.sum(rs)
                if val > best_val:
                    best_val = val
                    best_centers = rc.copy()
                    best_sides = rs.copy()
        except Exception:
            pass
            
    # 4. Final polish on the best found
    if best_centers is not None and time.time() < deadline - 5:
        refine_deadline = min(deadline - 2, time.time() + 20)
        try:
            rc, rs = _local_refine(best_centers, best_sides, max_iter=300, deadline=refine_deadline)
            if _is_feasible(rc, rs):
                val = np.sum(rs)
                if val > best_val:
                    best_val = val
                    best_centers = rc.copy()
                    best_sides = rs.copy()
        except Exception:
            pass
            
    return best_centers, best_sides
