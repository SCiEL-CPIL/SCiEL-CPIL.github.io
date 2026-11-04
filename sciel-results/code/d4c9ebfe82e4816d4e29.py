"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import fsolve
import random

def _eval_packing(xy, r):
    """
    Evaluate a packing: returns (total_radius, is_valid).
    Valid means all circles inside [0,1]^2 and disjoint interiors.
    """
    n = len(xy)
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if ri < -1e-9:
            return 0.0, False
        if x - ri < -1e-9 or 1.0 - x - ri < -1e-9 or y - ri < -1e-9 or 1.0 - y - ri < -1e-9:
            return 0.0, False
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-9:
                return 0.0, False
    return float(np.sum(r)), True

def _grid_init(side, n=N):
    """Create a grid of up to n circles."""
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) < n:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.48 / side
                positions.append([x, y, r])
    while len(positions) < n:
        positions.append([0.5, 0.5, 0.01])
    return np.array(positions[:n])

def _random_init(n, seed=0):
    """Random initialization with small radii."""
    rng = np.random.RandomState(seed)
    xy = rng.uniform(0.15, 0.85, size=(n, 2))
    r = rng.uniform(0.02, 0.04, size=n)
    return xy, r

def _solve_constraint_system(xy, r, constraints, deadline):
    """
    Solve for exact positions and radii given a set of active constraints.
    constraints: list of tuples.
      ('cc', i, j): circle i and j are tangent.
      ('cw', i, wall): circle i touches wall 0,1,2,3 (L, R, B, T).
    
    Returns (xy, r, success).
    """
    if time.time() > deadline:
        return xy, r, False

    # Variables: x0, y0, r0, x1, y1, r1, ...
    # We only solve for variables involved in constraints to save time? 
    # No, fsolve needs a square system. 
    # Strategy: Fix the positions/radii of circles NOT involved in the current "active" set?
    # Actually, a simpler robust approach for this specific "structural" change:
    # Pick a subset of circles (e.g., 3-5) that form a cluster.
    # Define constraints for that cluster.
    # Solve for their exact geometry.
    # Keep the rest fixed.
    # This is a "local cluster refinement" strategy.
    
    # Let's refine a specific subset of circles.
    # Identify indices involved in constraints.
    involved_indices = set()
    for c in constraints:
        if c[0] == 'cc':
            involved_indices.add(c[1])
            involved_indices.add(c[2])
        elif c[0] == 'cw':
            involved_indices.add(c[1])
    
    if len(involved_indices) < 2:
        return xy, r, False
        
    indices = list(involved_indices)
    n_vars = len(indices) * 3  # x, y, r for each
    
    # Initial guess
    x0 = np.zeros(n_vars)
    for k, idx in enumerate(indices):
        x0[k*3] = xy[idx, 0]
        x0[k*3+1] = xy[idx, 1]
        x0[k*3+2] = r[idx]
        
    def equations(vars):
        eqs = []
        # First, unpack vars into local xy, r
        local_xy = np.zeros((n_vars//3, 2))
        local_r = np.zeros(n_vars//3)
        for k in range(len(indices)):
            local_xy[k, 0] = vars[k*3]
            local_xy[k, 1] = vars[k*3+1]
            local_r[k] = vars[k*3+2]
            
        # Enforce wall constraints for involved circles
        for c in constraints:
            if c[0] == 'cw':
                idx = c[1]
                wall = c[2]
                k = indices.index(idx)
                x, y, r = local_xy[k, 0], local_xy[k, 1], local_r[k]
                if wall == 0: # Left
                    eqs.append(x - r)
                elif wall == 1: # Right
                    eqs.append(1.0 - x - r)
                elif wall == 2: # Bottom
                    eqs.append(y - r)
                elif wall == 3: # Top
                    eqs.append(1.0 - y - r)
                    
        # Enforce circle-circle constraints
        for c in constraints:
            if c[0] == 'cc':
                i_idx, j_idx = c[1], c[2]
                ki = indices.index(i_idx)
                kj = indices.index(j_idx)
                xi, yi, ri = local_xy[ki, 0], local_xy[ki, 1], local_r[ki]
                xj, yj, rj = local_xy[kj, 0], local_xy[kj, 1], local_r[kj]
                dist = np.sqrt((xi-xj)**2 + (yi-yj)**2)
                eqs.append(dist - (ri + rj))
                
        # We have fewer equations than variables if we don't fix some.
        # To make it a square system, we need to fix some variables or add more constraints.
        # Simple strategy: Fix the center of one circle in the cluster? No, that's arbitrary.
        # Better: Only solve for the variables that are "free".
        # Actually, a standard trick: If we have N circles and M constraints, and M < 3N, it's underdetermined.
        # We can use least squares? No, fsolve needs square.
        
        # Let's try a different approach: 
        # We will only solve for the radii and positions of a SMALL cluster (e.g. 3 circles) 
        # where we assume they are all tangent to each other and to walls.
        # If we pick 3 circles, we have 9 variables. 
        # 3 mutual tangencies + 3 wall contacts = 6 equations. Still underdetermined.
        
        # Alternative Structural Strategy: 
        # Instead of solving a sparse system, use a "Best Fit" approach but with 
        # EXACT tangency enforcement.
        
        # Let's pivot the implementation to a simpler, robust structural change:
        # "Tangency-Forcing Local Search".
        # 1. Identify the most overlapping pair.
        # 2. Force them to be tangent by adjusting positions/radii.
        # 3. Check if this improves the sum of radii.
        
        # This is getting complicated to implement correctly in fsolve with variable constraints.
        # Let's implement a simpler structural change: 
        # **Simulated Annealing with Exact Overlap Resolution**.
        # The current code uses physics (forces). 
        # The inspiration suggests SA.
        # Let's implement SA where the move is: Pick 2 circles, swap their positions? No.
        # Move: Pick 1 circle, move it to a random location.
        # Cost: Sum of overlaps.
        # But we want to maximize sum of radii.
        
        # Let's look at Inspiration 2: "simulated annealing approach".
        # And Inspiration 1/3: "gradient polish".
        
        # Let's try a **Greedy Swap and Resize** strategy.
        # 1. Start with a valid packing.
        # 2. For each circle, try to increase its radius.
        # 3. If it overlaps, try to move it or shrink a neighbor.
        
        # Actually, the most effective structural change from the inspirations that isn't just "tuning" is 
        # **using scipy.optimize.minimize with a different objective formulation**.
        # The current code uses physics. 
        # Let's use `scipy.optimize.differential_evolution` or `dual_annealing`? 
        # No, those are global.
        
        # Let's go with **Iterative Closest Pair Resolution**.
        # This is a different algorithmic family: Geometric Conflict Resolution.
        
        # For now, let's return the input to avoid breaking the program, 
        # and I will implement the real logic in the main construct loop.
        
        # To satisfy the "Square System" requirement of fsolve, I will fix the positions of all 
        # circles EXCEPT the ones in the cluster, and only solve for the cluster's radii and positions 
        # if the number of constraints matches 3*len(cluster).
        # This is rare.
        
        # Let's abandon the complex fsolve cluster solver for this step and implement 
        # **Simulated Annealing** as suggested by Inspiration 2, which is a distinct paradigm 
        # from the current Physics-based relaxation.
        
        return xy, r, False # Placeholder to avoid error, real logic in construct

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    # Phase 1: Initial valid packing using Grid + Local Search (kept from seed for stability)
    # We need a good starting point.
    best_xy = None
    best_r = None
    best_total = 0.0
    
    # Use the existing grid init and a quick local search to get a valid base
    # We'll reuse the logic from the seed but wrap it in the new strategy.
    
    # Let's implement the SA strategy directly.
    # State: xy (N, 2), r (N,)
    # Objective: Maximize sum(r) subject to non-overlap.
    # Penalty Method for SA:
    # Score = sum(r) - Penalty * Total_Overlap_Area
    
    def _get_score(xy, r):
        n = len(xy)
        total_r = np.sum(r)
        penalty = 0.0
        # Wall penalties
        for i in range(n):
            x, y, ri = xy[i, 0], xy[i, 1], r[i]
            if x < ri: penalty += (ri - x)**2
            if 1.0 - x < ri: penalty += (1.0 - x - ri)**2
            if y < ri: penalty += (ri - y)**2
            if 1.0 - y < ri: penalty += (1.0 - y - ri)**2
        # Pairwise penalties
        for i in range(n):
            for j in range(i+1, n):
                dx = xy[i, 0] - xy[j, 0]
                dy = xy[i, 1] - xy[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    overlap = min_dist - dist
                    penalty += overlap**2
        return total_r - 100.0 * penalty

    def _is_valid(xy, r):
        n = len(xy)
        for i in range(n):
            x, y, ri = xy[i, 0], xy[i, 1], r[i]
            if ri < -1e-9: return False
            if x - ri < -1e-9 or 1.0 - x - ri < -1e-9 or y - ri < -1e-9 or 1.0 - y - ri < -1e-9:
                return False
        for i in range(n):
            for j in range(i + 1, n):
                dx = xy[i, 0] - xy[j, 0]
                dy = xy[i, 1] - xy[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < r[i] + r[j] - 1e-9:
                    return False
        return True

    # Initialize
    # Start with a grid to ensure validity
    xy, r = _random_init(N, seed=42)
    # Quick physics relaxation to get a valid start (using seed code logic simplified)
    # We'll just run a few steps of the seed's physics to get a valid state.
    # For brevity in this structural change, we'll use a simple "push apart" loop.
    for step in range(100):
        if time.time() > deadline: break
        moved = False
        for i in range(N):
            for j in range(i+1, N):
                dx = xy[i, 0] - xy[j, 0]
                dy = xy[i, 1] - xy[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    if dist < 1e-6:
                        dx, dy = 0.01, 0.01
                        dist = np.sqrt(dx*dx+dy*dy)
                    overlap = min_dist - dist
                    fx = overlap * dx / dist
                    fy = overlap * dy / dist
                    xy[i, 0] += fx * 0.5
                    xy[i, 1] += fy * 0.5
                    xy[j, 0] -= fx * 0.5
                    xy[j, 1] -= fy * 0.5
                    moved = True
        # Wall push
        for i in range(N):
            ri = r[i]
            if xy[i, 0] < ri: xy[i, 0] = ri
            if xy[i, 0] > 1-ri: xy[i, 0] = 1-ri
            if xy[i, 1] < ri: xy[i, 1] = ri
            if xy[i, 1] > 1-ri: xy[i, 1] = 1-ri
            
    # Ensure valid
    if not _is_valid(xy, r):
        # Fallback to grid
        pos = _grid_init(5)
        xy = pos[:, :2].copy()
        r = pos[:, 2].copy()
        
    best_xy = xy.copy()
    best_r = r.copy()
    best_total = np.sum(r)
    best_valid = _is_valid(best_xy, best_r)
    
    # Simulated Annealing
    current_xy = best_xy.copy()
    current_r = best_r.copy()
    current_score = _get_score(current_xy, current_r)
    
    T = 0.01
    T_min = 1e-5
    cooling = 0.995
    
    rng = np.random.RandomState(123)
    
    while time.time() < deadline - 5.0:
        # Generate move
        # 1. Pick a circle
        i = rng.randint(N)
        
        # 2. Move it slightly or change radius
        move_type = rng.rand()
        new_xy = current_xy.copy()
        new_r = current_r.copy()
        
        if move_type < 0.5:
            # Move position
            dx = rng.normal(0, 0.05)
            dy = rng.normal(0, 0.05)
            new_xy[i, 0] += dx
            new_xy[i, 1] += dy
        else:
            # Change radius
            dr = rng.normal(0, 0.02)
            new_r[i] += dr
            if new_r[i] < 0.01: new_r[i] = 0.01
            
        # Clamp to box
        ri = new_r[i]
        new_xy[i, 0] = np.clip(new_xy[i, 0], ri, 1.0-ri)
        new_xy[i, 1] = np.clip(new_xy[i, 1], ri, 1.0-ri)
        
        new_score = _get_score(new_xy, new_r)
        delta = new_score - current_score
        
        # Accept if better, or with probability exp(delta/T)
        if delta > 0 or rng.rand() < np.exp(delta / T):
            current_xy = new_xy
            current_r = new_r
            current_score = new_score
            
            # Check validity and update best
            if _is_valid(current_xy, current_r):
                total_r = np.sum(current_r)
                if total_r > best_total:
                    best_total = total_r
                    best_xy = current_xy.copy()
                    best_r = current_r.copy()
                    best_valid = True
        
        T *= cooling
        if T < T_min:
            T = T_min

    # Final Polish: Local search on best valid
    if best_valid:
        # Simple local search: try to increase radius of each circle if possible
        for _ in range(50): # Iterations
            if time.time() > deadline - 2.0: break
            improved = False
            for i in range(N):
                # Try to increase r[i]
                old_r = best_r[i]
                max_possible = min(best_xy[i, 0], 1.0-best_xy[i, 0], best_xy[i, 1], 1.0-best_xy[i, 1])
                # Check neighbors
                min_neighbor_clear = max_possible
                for j in range(N):
                    if i == j: continue
                    dx = best_xy[i, 0] - best_xy[j, 0]
                    dy = best_xy[i, 1] - best_xy[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    clear = dist - best_r[j]
                    if clear < min_neighbor_clear:
                        min_neighbor_clear = clear
                
                new_r = min(max_possible, min_neighbor_clear)
                if new_r > old_r + 1e-6:
                    best_r[i] = new_r
                    improved = True
            if not improved:
                break

    # Final validation
    if best_valid:
        result = np.zeros((N, 3))
        for i in range(N):
            x, y, ri = best_xy[i, 0], best_xy[i, 1], best_r[i]
            ri = min(ri, x, 1.0 - x, y, 1.0 - y)
            ri = max(ri, 0.001)
            result[i, 2] = ri
            result[i, 0] = np.clip(x, ri, 1.0 - ri)
            result[i, 1] = np.clip(y, ri, 1.0 - ri)
            
        # One last check
        if _is_valid(result[:, :2], result[:, 2]):
            return result

    # Fallback
    side = int(np.ceil(np.sqrt(N)))
    result = np.zeros((N, 3))
    for i in range(N):
        gx = i % side
        gy = i // side
        result[i, 0] = (gx + 0.5) / side
        result[i, 1] = (gy + 0.5) / side
        result[i, 2] = 0.4 / side
    return result
# EVOLVE-BLOCK-END
