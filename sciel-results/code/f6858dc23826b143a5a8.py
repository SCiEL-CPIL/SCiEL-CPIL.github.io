import numpy as np
import time
from scipy.optimize import minimize

def _is_valid(centers, sides, tol=1e-7):
    """Check all constraints: inside unit square, no overlaps, positive sides."""
    n = len(sides)
    if n != 10:
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    if np.any(centers[:, 0] - sides / 2 < -tol) or np.any(centers[:, 0] + sides / 2 > 1 + tol):
        return False
    if np.any(centers[:, 1] - sides / 2 < -tol) or np.any(centers[:, 1] + sides / 1 > 1 + tol):
        return False
    for i in range(n):
        for j in range(i + 1, n):
            t = (sides[i] + sides[j]) / 2 - tol
            if abs(centers[i, 0] - centers[j, 0]) < t and abs(centers[i, 1] - centers[j, 1]) < t:
                return False
    return True

def _objective(centers, sides):
    """Sum of side lengths."""
    return float(np.sum(sides))

def _max_possible_side(i, centers, sides, tol=1e-7):
    """
    Calculate the maximum side length square i can have, given its current center
    and the positions/sizes of all other squares.
    """
    cx, cy = centers[i]
    s_max = 1.0 # Upper bound from unit square
    
    # Boundary constraints
    s_max = min(s_max, 2 * cx, 2 * (1 - cx), 2 * cy, 2 * (1 - cy))
    
    # Overlap constraints with other squares
    for j in range(10):
        if i == j:
            continue
        cx_j, cy_j = centers[j]
        s_j = sides[j]
        
        # The condition for NO overlap is:
        # |cx - cx_j| >= (s_i + s_j)/2 - tol  OR  |cy - cy_j| >= (s_i + s_j)/2 - tol
        # We want to find max s_i such that this holds.
        # This implies:
        # s_i <= 2 * |cx - cx_j| + 2*tol - s_j  (if we separate in x)
        # s_i <= 2 * |cy - cy_j| + 2*tol - s_j  (if we separate in y)
        
        # To be safe, we must satisfy BOTH potential separation conditions? 
        # No, we just need ONE to hold. 
        # However, if we increase s_i, we might violate the current separation.
        # The "bottleneck" is the minimum distance to the other square in either axis.
        # Actually, if they are currently separated in x by dx, we can grow until 
        # (s_i + s_j)/2 = dx + tol. So s_i <= 2*dx + 2*tol - s_j.
        # If they are separated in y by dy, s_i <= 2*dy + 2*tol - s_j.
        # If they are NOT separated (overlap), this is invalid state, but we assume valid input.
        
        dx = abs(cx - cx_j)
        dy = abs(cy - cy_j)
        
        # If they are touching or separated in x, the limit is determined by x.
        # If in y, by y.
        # The actual limit is the minimum of the two potential limits, because 
        # we can't exceed the "wall" created by the other square in EITHER dimension 
        # if we want to maintain the separation that currently exists.
        # Wait, if they are separated in x, we can grow in y freely (until boundary).
        # So the limit from square j is:
        # If dx > dy: The separation is primarily in x? No.
        
        # Let's look at the constraint:
        # NOT ( |cx-cx_j| < (s_i+s_j)/2 - tol AND |cy-cy_j| < (s_i+s_j)/2 - tol )
        # This is satisfied if:
        # |cx-cx_j| >= (s_i+s_j)/2 - tol  OR  |cy-cy_j| >= (s_i+s_j)/2 - tol
        
        # Let L_x = 2*|cx-cx_j| + 2*tol - s_j
        # Let L_y = 2*|cy-cy_j| + 2*tol - s_j
        
        # If s_i <= L_x, then the x-separation condition holds.
        # If s_i <= L_y, then the y-separation condition holds.
        
        # We need (s_i <= L_x) OR (s_i <= L_y).
        # This is equivalent to s_i <= max(L_x, L_y).
        
        limit_x = 2 * dx + 2 * tol - s_j
        limit_y = 2 * dy + 2 * tol - s_j
        
        s_max = min(s_max, max(limit_x, limit_y))
        
    return max(0.0, s_max)

def _greedy_enlarge(centers, sides, tol=1e-7, max_iter=100):
    """
    Iteratively increase the side length of each square to its maximum possible value
    given the current positions of others. Repeat until convergence.
    """
    centers = centers.copy()
    sides = sides.copy()
    
    for _ in range(max_iter):
        changed = False
        for i in range(10):
            s_new = _max_possible_side(i, centers, sides, tol)
            # We can only grow, not shrink (shrinking might allow others to grow, 
            # but pure growth is a good local heuristic for max sum)
            if s_new > sides[i] + 1e-9:
                sides[i] = s_new
                changed = True
        
        if not changed:
            break
            
    return centers, sides

def _pattern_search(centers, sides, tol=1e-7, time_limit=500):
    """
    Direct search (Pattern Search) to maximize sum of sides.
    We perturb centers and sides. If valid and better, accept.
    """
    best_centers = centers.copy()
    best_sides = sides.copy()
    best_val = _objective(best_centers, best_sides)
    
    # Initial step sizes
    step_c = np.full(20, 0.05) # 10 centers * 2 coords
    step_s = np.full(10, 0.02)  # 10 sides
    
    # Combined vector for search
    # We will search in the space of centers and sides.
    # To ensure validity, we use a "repair" or just reject invalid moves.
    # Given the complexity of constraints, rejection is safer and simpler.
    
    start_time = time.time()
    no_improve_count = 0
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Generate random perturbation
        # We perturb each coordinate independently
        for k in range(20): # 20 center coords
            if time.time() - start_time > time_limit:
                break
                
            # Try positive step
            new_centers = best_centers.copy()
            new_sides = best_sides.copy()
            new_centers[k // 2, k % 2] += step_c[k]
            
            if _is_valid(new_centers, new_sides, tol):
                val = _objective(new_centers, new_sides)
                if val > best_val:
                    best_centers = new_centers
                    best_sides = new_sides
                    best_val = val
                    improved = True
            else:
                # Try negative step
                new_centers = best_centers.copy()
                new_centers[k // 2, k % 2] -= step_c[k]
                if _is_valid(new_centers, new_sides, tol):
                    val = _objective(new_centers, new_sides)
                    if val > best_val:
                        best_centers = new_centers
                        best_sides = new_sides
                        best_val = val
                        improved = True

        for k in range(10): # 10 side lengths
            if time.time() - start_time > time_limit:
                break
                
            # Try increasing side
            new_sides = best_sides.copy()
            new_sides[k] += step_s[k]
            if new_sides[k] < 1.0:
                if _is_valid(best_centers, new_sides, tol):
                    val = _objective(best_centers, new_sides)
                    if val > best_val:
                        best_centers = best_centers.copy()
                        best_sides = new_sides
                        best_val = val
                        improved = True
            else:
                # If we hit the boundary, try decreasing? No, we want to maximize.
                # If we can't increase, maybe we can move center to allow increase?
                # That's handled by center perturbations.
                pass
                
            # Try decreasing side (rarely helpful for max sum, but might allow repositioning? 
            # No, smaller side is always worse for objective. Only useful if it allows 
            # OTHER squares to grow. But pattern search handles others separately.)
            # So we skip decreasing sides unless we are stuck? 
            # Actually, decreasing a side might free up space for a much larger neighbor.
            # Let's try it.
            new_sides = best_sides.copy()
            new_sides[k] = max(0.01, new_sides[k] - step_s[k])
            if _is_valid(best_centers, new_sides, tol):
                val = _objective(best_centers, new_sides)
                if val > best_val:
                    best_centers = best_centers.copy()
                    best_sides = new_sides
                    best_val = val
                    improved = True

        if improved:
            no_improve_count = 0
            # Slightly increase step size to explore faster
            step_c *= 1.1
            step_s *= 1.1
        else:
            no_improve_count += 1
            # Decrease step size to refine
            step_c *= 0.8
            step_s *= 0.8
            
            if no_improve_count > 20:
                # Restart with smaller steps or break
                break
                
        # Greedy enlarge after each successful search step?
        # It's expensive (100 * 10 * 10 checks). 
        # Maybe do it every 5 iterations?
        if no_improve_count == 0 and no_improve_count % 5 == 0:
             c_enl, s_enl = _greedy_enlarge(best_centers, best_sides, tol, max_iter=20)
             if _is_valid(c_enl, s_enl, tol):
                 val = _objective(c_enl, s_enl)
                 if val > best_val:
                     best_centers = c_enl
                     best_sides = s_enl
                     best_val = val

    return best_centers, best_sides

def _generate_seeds():
    """Generate diverse initial configurations."""
    seeds = []
    
    # 1. Grid
    c = np.array([[0.1, 0.1], [0.3, 0.1], [0.5, 0.1], [0.7, 0.1], [0.9, 0.1],
                  [0.1, 0.3], [0.3, 0.3], [0.5, 0.3], [0.7, 0.3], [0.9, 0.3]])
    s = np.full(10, 0.18)
    seeds.append((c, s))
    
    # 2. Heterogeneous: 1 large, 9 small
    c = np.array([[0.25, 0.25],
                  [0.75, 0.25], [0.5, 0.25], [0.25, 0.5],
                  [0.75, 0.5], [0.5, 0.5], [0.25, 0.75],
                  [0.75, 0.75], [0.5, 0.75], [0.5, 0.9]])
    s = np.array([0.4, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.1])
    seeds.append((c, s))
    
    # 3. Random
    rng = np.random.default_rng(42)
    s = rng.uniform(0.1, 0.3, size=10)
    c = np.zeros((10, 2))
    for i in range(10):
        c[i, 0] = rng.uniform(s[i]/2, 1 - s[i]/2)
        c[i, 1] = rng.uniform(s[i]/2, 1 - s[i]/2)
    seeds.append((c, s))
    
    # 4. Corner packing
    c = np.array([[0.2, 0.2], [0.8, 0.2], [0.2, 0.8], [0.8, 0.8],
                  [0.5, 0.2], [0.5, 0.8], [0.2, 0.5], [0.8, 0.5],
                  [0.5, 0.5], [0.5, 0.6]])
    s = np.array([0.3, 0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2, 0.1])
    seeds.append((c, s))
    
    return seeds

def squares_in_square10() -> tuple:
    """
    Place 10 axis-aligned squares in the unit square to maximize sum of side lengths.
    Uses Pattern Search with Greedy Enlargement.
    """
    n = 10
    tol = 1e-7
    start_time = time.time()
    deadline = start_time + 590 # Safety margin
    
    seeds = _generate_seeds()
    
    best_centers = None
    best_sides = None
    best_val = -float('inf')
    
    for seed_idx, (centers, sides) in enumerate(seeds):
        if time.time() > deadline - 100:
            break
            
        # Ensure seed is valid
        if not _is_valid(centers, sides, tol):
            # Simple projection: shrink sides until valid
            # This is a fallback, seeds should be valid
            continue
        
        # 1. Greedy Enlargement
        c_enl, s_enl = _greedy_enlarge(centers, sides, tol, max_iter=50)
        
        if _is_valid(c_enl, s_enl, tol):
            val = _objective(c_enl, s_enl)
            if val > best_val:
                best_val = val
                best_centers = c_enl.copy()
                best_sides = s_enl.copy()
        
        # 2. Pattern Search
        if time.time() > deadline - 50:
            continue
            
        c_ps, s_ps = _pattern_search(c_enl, s_enl, tol, time_limit=50)
        
        if _is_valid(c_ps, s_ps, tol):
            val = _objective(c_ps, s_ps)
            if val > best_val:
                best_val = val
                best_centers = c_ps.copy()
                best_sides = s_ps.copy()
                
    # If no valid result, fall back to a simple valid grid
    if best_centers is None:
        best_centers = np.array([[0.1, 0.1], [0.3, 0.1], [0.5, 0.1], [0.7, 0.1], [0.9, 0.1],
                                 [0.1, 0.3], [0.3, 0.3], [0.5, 0.3], [0.7, 0.3], [0.9, 0.3]], dtype=np.float64)
        best_sides = np.full(10, 0.15, dtype=np.float64)
        
    return best_centers, best_sides
