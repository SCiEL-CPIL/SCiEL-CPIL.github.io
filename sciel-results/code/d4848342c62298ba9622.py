import numpy as np
import time
from itertools import product

def squares_in_square10() -> tuple:
    """
    Place 10 axis-aligned squares inside the unit square [0,1]^2 to maximize
    the sum of their side lengths.
    
    Strategy:
    1. Define a set of discrete grid topologies (partitions of the unit square).
    2. For each topology, represent the layout as a set of cells.
    3. Map the 10 squares to these cells (or subsets).
    4. Optimize the row heights and column widths to maximize the sum of min(h_i, w_j).
    5. Return the best configuration found.
    """
    n = 10
    tol = 1e-7
    start_time = time.time()
    deadline = start_time + 600  # 600s limit, keep margin
    
    best_centers = None
    best_sides = None
    best_score = 0.0

    # Helper to check time
    def time_left():
        return deadline - time.time()

    # Define topologies as (rows, cols, assignment)
    # Assignment is a list of 10 indices into the grid cells.
    # If grid has R*C cells, we need to map 10 squares.
    # Some cells might be empty or shared? No, each square gets a cell.
    # If R*C > 10, some cells are empty.
    # If R*C < 10, we can't use a simple grid. 
    # Let's stick to grids with >= 10 cells or specific composite shapes.
    
    # Topology 1: 3x4 grid (12 cells). 10 squares, 2 empty.
    # Topology 2: 2x5 grid (10 cells). 10 squares, 0 empty.
    # Topology 3: 1x10 grid (10 cells).
    # Topology 4: 4x3 grid (12 cells).
    # Topology 5: 5x2 grid (10 cells).
    # Topology 6: 10x1 grid (10 cells).
    
    # Also consider "L-shapes" or other partitions?
    # For now, let's optimize the rectangular grids first.
    
    topologies = [
        (2, 5),
        (5, 2),
        (3, 4),
        (4, 3),
        (1, 10),
        (10, 1),
    ]
    
    # For each topology, we need to assign 10 squares to R*C cells.
    # To maximize sum, we should probably assign larger squares to larger cells?
    # But we don't know sizes yet.
    # We can just optimize the boundaries for a fixed assignment.
    # The assignment of squares to cells matters for the objective function 
    # only if we have constraints linking squares. Here, squares are independent 
    # except for the shared row/col boundaries.
    # Wait, if square i is in cell (r, c), its size is min(h_r, w_c).
    # The objective is sum(min(h_r, w_c)) for all squares.
    # If two squares are in the same row r, they share h_r.
    # If two squares are in the same col c, they share w_c.
    
    # We can iterate over permutations of square indices assigned to cells?
    # That's 10! which is too big.
    # But the objective is symmetric with respect to swapping squares i and j 
    # IF they are in cells that have the same (h, w) pair? No.
    # Actually, the assignment of *which* square goes where doesn't change the 
    # set of (h_r, w_c) pairs used, unless we consider that squares are distinct 
    # only by their resulting size. The metric is sum of sizes.
    # So, for a fixed grid partition (h, w), the maximum sum is obtained by 
    # filling the cells. Since all cells are available, and we have 10 squares,
    # we just pick 10 cells out of R*C.
    # To maximize sum, we should pick the 10 largest min(h_r, w_c) values?
    # No, h and w are variables. We optimize h and w.
    
    # Let's formulate the optimization problem for a fixed (R, C) and fixed set of active cells.
    # Let A be the set of active cells.
    # Maximize sum_{(r,c) in A} min(h_r, w_c)
    # Subject to: sum(h_r) = 1, sum(w_c) = 1, h_r > 0, w_c > 0.
    
    # This is a convex optimization problem? 
    # min(h, w) is concave. Sum of concave functions is concave.
    # Maximizing a concave function is a convex optimization problem.
    # We can use scipy.optimize.minimize on the negative objective.
    
    from scipy.optimize import minimize
    
    def solve_topology(R, C, active_cells):
        """
        Solve for h (size R) and w (size C) given active cells.
        active_cells: set of (r, c) tuples.
        """
        # Variables: h[0..R-1], w[0..C-1]
        # Constraints: sum(h) = 1, sum(w) = 1, h>0, w>0.
        # We can parameterize h and w to satisfy constraints automatically?
        # Or use bounds and constraints.
        
        # Let's use simple coordinate descent or Nelder-Mead on h and w directly,
        # projecting back to simplex if needed, or just using bounds [0, 1].
        # Since sum must be 1, we can normalize after each step?
        # Better: use scipy minimize with bounds and equality constraints.
        
        def objective(vars):
            h = vars[:R]
            w = vars[R:R+C]
            # Normalize to ensure sum=1? 
            # If we enforce sum=1 via constraint, we don't need to normalize in objective.
            # But for numerical stability, let's normalize inside.
            h = np.maximum(h, 1e-9)
            w = np.maximum(w, 1e-9)
            h = h / np.sum(h)
            w = w / np.sum(w)
            
            total = 0.0
            for (r, c) in active_cells:
                total += min(h[r], w[c])
            return -total
        
        # Initial guess: uniform
        x0 = np.ones(R + C) / (R + C)
        
        # Bounds: [1e-9, 1]
        bounds = [(1e-9, 1.0)] * (R + C)
        
        # Constraints: sum(h) = 1, sum(w) = 1
        # Actually, if we normalize inside the objective, the optimizer might
        # struggle with the flat manifold.
        # Let's fix one variable to handle the sum constraint.
        # h_R-1 = 1 - sum(h_0..h_R-2)
        # w_C-1 = 1 - sum(w_0..w_C-2)
        
        R_vars = R - 1
        C_vars = C - 1
        n_vars = R_vars + C_vars
        
        def objective_reduced(vars):
            h_rest = vars[:R_vars]
            w_rest = vars[R_vars:R_vars+C_vars]
            
            # Ensure positivity and sum < 1 for the last ones
            # Clip to small positive
            h_rest = np.clip(h_rest, 1e-9, 1.0 - 1e-9)
            w_rest = np.clip(w_rest, 1e-9, 1.0 - 1e-9)
            
            sum_h_rest = np.sum(h_rest)
            sum_w_rest = np.sum(w_rest)
            
            # If sum >= 1, it's infeasible for the last variable to be positive.
            # We can penalize or clip.
            # Let's just clip the last variable to 1e-9 if sum is too high.
            h_last = max(1e-9, 1.0 - sum_h_rest)
            w_last = max(1e-9, 1.0 - sum_w_rest)
            
            h = np.append(h_rest, h_last)
            w = np.append(w_rest, w_last)
            
            # Normalize just in case of numerical drift?
            # No, the construction ensures sum=1 (approx).
            
            total = 0.0
            for (r, c) in active_cells:
                total += min(h[r], w[c])
            return -total
        
        x0_reduced = np.ones(n_vars) / n_vars
        
        res = minimize(objective_reduced, x0_reduced, method='Nelder-Mead', 
                       options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
        
        if res.success:
            x_opt = res.x
            h_rest = x_opt[:R_vars]
            w_rest = x_opt[R_vars:R_vars+C_vars]
            h_last = 1.0 - np.sum(h_rest)
            w_last = 1.0 - np.sum(w_rest)
            
            # If h_last or w_last is not positive, the solution is on the boundary
            # which Nelder-Mead might handle poorly with simple clipping.
            # Let's check validity.
            if h_last < 1e-9 or w_last < 1e-9:
                # Try to fix by redistributing? Or just accept if close.
                pass
            
            h = np.append(h_rest, h_last)
            w = np.append(w_rest, w_last)
            
            # Construct centers and sides
            centers = np.zeros((10, 2))
            sides = np.zeros(10)
            
            # Calculate cell centers
            # y coordinates: cumulative sum of h
            y_coords = np.concatenate(([0.0], np.cumsum(h)))
            x_coords = np.concatenate(([0.0], np.cumsum(w)))
            
            idx = 0
            for (r, c) in active_cells:
                cx = (x_coords[c] + x_coords[c+1]) / 2.0
                cy = (y_coords[r] + y_coords[r+1]) / 2.0
                s = min(h[r], w[c])
                
                # Ensure square fits in cell
                # It does by definition.
                centers[idx] = [cx, cy]
                sides[idx] = s
                idx += 1
                
            score = np.sum(sides)
            return centers, sides, score
        else:
            return None, None, 0.0

    # Iterate over topologies
    for R, C in topologies:
        if time_left() < 10:
            break
            
        total_cells = R * C
        if total_cells < 10:
            continue
            
        # We need to choose 10 cells out of total_cells.
        # Which 10?
        # Intuitively, we want to use cells that allow for large squares.
        # In a uniform grid, all cells are same.
        # In an optimized grid, cells vary.
        # However, the optimization determines h and w based on which cells are active.
        # So we need to try different subsets of active cells?
        # That's C(total_cells, 10) possibilities.
        # For 2x5=10, only 1 subset (all).
        # For 3x4=12, C(12,10)=66 subsets.
        # For 4x3=12, 66 subsets.
        # For 1x10=10, 1 subset.
        
        # Let's start with the "all cells" case if total_cells == 10.
        # If total_cells > 10, we can try a few heuristics:
        # 1. All cells except corners?
        # 2. All cells except random ones?
        # 3. Since the objective is symmetric in some ways, maybe just using 
        #    the first 10 cells is a good start, but we can do better.
        
        # Let's just try the case where we use all cells if possible, 
        # or a dense subset.
        
        # For 2x5 and 5x2, use all.
        if total_cells == 10:
            active_cells = [(r, c) for r in range(R) for c in range(C)]
            centers, sides, score = solve_topology(R, C, active_cells)
            if centers is not None and score > best_score:
                best_centers = centers
                best_sides = sides
                best_score = score
                
        else:
            # total_cells > 10
            # Try a few subsets.
            # Subset 1: All cells except the last 2 (arbitrary).
            # Subset 2: All cells except corners.
            
            all_cells = [(r, c) for r in range(R) for c in range(C)]
            
            # Heuristic: Try removing cells to make the grid more "square-like" for the remaining?
            # Or just try all permutations of removing 2 cells?
            # C(12, 2) = 66. That's manageable.
            from itertools import combinations
            
            cells_to_remove = total_cells - 10
            remove_combos = list(combinations(range(total_cells), cells_to_remove))
            
            # Limit number of combos if too many
            if len(remove_combos) > 50:
                remove_combos = remove_combos[:50]
                
            for combo in remove_combos:
                if time_left() < 5:
                    break
                    
                remove_set = set(combo)
                active_cells = [cell for i, cell in enumerate(all_cells) if i not in remove_set]
                
                centers, sides, score = solve_topology(R, C, active_cells)
                if centers is not None and score > best_score:
                    best_centers = centers
                    best_sides = sides
                    best_score = score

    # Fallback if no topology worked (should not happen)
    if best_centers is None:
        # Simple grid 2x5
        R, C = 2, 5
        active_cells = [(r, c) for r in range(R) for c in range(C)]
        centers, sides, score = solve_topology(R, C, active_cells)
        if centers is not None:
            best_centers = centers
            best_sides = sides
            best_score = score
        else:
            # Hardcoded fallback
            centers = np.zeros((10, 2))
            sides = np.ones(10) * 0.2
            # 2 rows, 5 cols
            for i in range(10):
                r = i // 5
                c = i % 5
                centers[i] = [c * 0.2 + 0.1, r * 0.5 + 0.25]
                sides[i] = 0.2

    return best_centers, best_sides

