# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _evaluate(centers, sides, tol=1e-7):
    """
    Vectorized evaluation of the packing.
    Returns:
        objective: sum of side lengths
        violation: maximum violation of constraints (0 if valid)
    """
    n = len(sides)
    cx = centers[:, 0]
    cy = centers[:, 1]
    
    obj = np.sum(sides)
    
    # Boundary violations
    # Squares must be inside [-tol, 1+tol]
    # cx - s/2 >= -tol  => s/2 - cx <= tol
    # cx + s/2 <= 1+tol => cx + s/2 - 1 <= tol
    lo_x = cx - sides * 0.5
    hi_x = cx + sides * 0.5
    lo_y = cy - sides * 0.5
    hi_y = cy + sides * 0.5
    
    # Violation is how much it exceeds bounds
    bv_x = np.maximum(0, -lo_x) + np.maximum(0, hi_x - 1.0)
    bv_y = np.maximum(0, -lo_y) + np.maximum(0, hi_y - 1.0)
    max_bv = np.max(bv_x + bv_y)
    
    # Pairwise overlap violations
    # Overlap if |cx_i - cx_j| < (s_i + s_j)/2 - tol AND |cy_i - cy_j| < (s_i + s_j)/2 - tol
    # We want to measure the "depth" of overlap if any.
    # dx = |cx_i - cx_j|, dy = |cy_i - cy_j|
    # t_x = (s_i + s_j)/2 - tol
    # t_y = (s_i + s_j)/2 - tol
    # If dx < t_x and dy < t_y, there is a violation.
    # The amount of violation can be defined as max(0, t_x - dx) + max(0, t_y - dy) or similar.
    # For a smooth penalty, we can use:
    # pen = max(0, t_x - dx) * max(0, t_y - dy) is non-smooth.
    # Let's use a simpler violation metric: 
    # If they overlap, the violation is the minimum distance we need to push them apart to separate them.
    # This is min(t_x - dx, t_y - dy) if both are positive? No.
    # Actually, to fix an overlap, we can move x or y.
    # The "cost" of overlap is often defined as the overlap area or the min separation needed.
    # For local search, a simple penalty that is >0 when overlapping is sufficient.
    # Let's use: max(0, (s_i+s_j)/2 - tol - |cx_i-cx_j|) * max(0, (s_i+s_j)/2 - tol - |cy_i-cy_j|)
    # This is 0 if they don't overlap in either dimension.
    # It is >0 if they overlap in both.
    # This is a good metric for "penalty".
    
    dx = np.abs(cx[:, None] - cx[None, :])
    dy = np.abs(cy[:, None] - cy[None, :])
    
    # t is the threshold for overlap
    t = 0.5 * (sides[:, None] + sides[None, :]) - tol
    
    # Overlap in x direction
    ov_x = np.maximum(0, t - dx)
    # Overlap in y direction
    ov_y = np.maximum(0, t - dy)
    
    # Product is non-zero only if overlapping in BOTH dimensions
    # This effectively measures the "area" of the overlap region (roughly)
    ov_area = ov_x * ov_y
    
    # We only care about i < j
    # Mask upper triangle
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    max_ov = np.max(ov_area[mask])
    
    # Total violation is max of boundary and overlap
    violation = max(max_bv, max_ov)
    
    return obj, violation

def _local_search(centers, sides, start_time, deadline, rng, tol=1e-7):
    """
    Adaptive local search with feasibility penalty.
    """
    n = len(sides)
    dim = 2 * n  # n centers (x,y) + n sides
    
    current_centers = centers.copy()
    current_sides = sides.copy()
    
    curr_obj, curr_viol = _evaluate(current_centers, current_sides, tol)
    
    best_centers = current_centers.copy()
    best_sides = current_sides.copy()
    best_obj = curr_obj if curr_viol < 1e-7 else -np.inf
    best_viol = curr_viol
    
    step = 0.05
    min_step = 1e-6
    max_step = 0.2
    
    # Penalty weight for violations
    # If we use a penalty, we can accept moves that are slightly invalid if the objective gain is huge,
    # but we must end up valid.
    # A safer approach: only accept valid moves, but use step size adaptation to find them.
    # However, if we get stuck in a local minimum of valid moves, we might miss better regions.
    # Given the strict "invalid result scores 0" rule, we MUST return a valid result.
    # So we track the best VALID solution found.
    
    while time.time() < deadline:
        # Choose a random dimension to perturb
        d = rng.integers(0, dim)
        
        new_centers = current_centers
        new_sides = current_sides
        
        # Create copies only if we modify them
        # To save time, we can modify in place and revert, but copying is safer and fast enough for N=10
        new_centers = current_centers.copy()
        new_sides = current_sides.copy()
        
        if d < n:
            # Perturb center x or y
            axis = rng.integers(0, 2)
            delta = rng.normal(0, step)
            new_centers[d, axis] += delta
        elif d < 2 * n:
            # Perturb side
            delta = rng.normal(0, step)
            new_sides[d - n] += delta
        else:
            # Perturb both x and y of a random center
            i = d - 2 * n
            new_centers[i, 0] += rng.normal(0, step)
            new_centers[i, 1] += rng.normal(0, step)
            
        # Ensure sides are positive
        if new_sides[d - n] < 0.01:
             new_sides[d - n] = 0.01
             
        # Evaluate
        new_obj, new_viol = _evaluate(new_centers, new_sides, tol)
        
        # Acceptance criteria
        # We want to improve the objective, but we must maintain validity for the "best" record.
        # For the "current" state, we can allow some flexibility?
        # No, let's stick to: Accept if it's valid and better.
        # If it's invalid, reject.
        # This is a strict hill climber.
        
        accepted = False
        if new_viol < 1e-7:
            if new_obj > curr_obj:
                accepted = True
                step *= 1.05 # Increase step on success
            else:
                # Even if not better, we might accept to explore?
                # Standard hill climbing rejects. 
                # Let's add a small probability of accepting worse valid moves to escape local minima?
                # Or just rely on step size changes.
                # Let's try: Accept if valid, regardless of objective? No, that's random walk.
                # Accept if valid and (new_obj > curr_obj OR rng.random() < 0.05)
                # Actually, simple hill climbing with adaptive step is often robust.
                if rng.random() < 0.1:
                    accepted = True
                    step *= 0.95
                else:
                    step *= 0.95
        else:
            # Invalid move
            step *= 0.9
            # If step gets too small, we might get stuck. 
            # But we are checking time.
            
        if accepted:
            current_centers = new_centers
            current_sides = new_sides
            curr_obj = new_obj
            curr_viol = new_viol
            
            if curr_viol < 1e-7 and curr_obj > best_obj:
                best_obj = curr_obj
                best_centers = current_centers.copy()
                best_sides = current_sides.copy()
                best_viol = curr_viol
        
        # Update step size
        step = np.clip(step, min_step, max_step)
        
        # Periodic restart? Or just let it run?
        # If step < min_step for a long time, we might be stuck.
        # We can reset step if we haven't improved in a while.
        # Simple heuristic: if step < 1e-4, increase it a bit.
        if step < 1e-4 and rng.random() < 0.01:
            step = 0.05

    return best_centers, best_sides, best_obj

def _generate_initial_config(n=10, method='heuristic'):
    """
    Generate initial configurations.
    Method 'heuristic': 2 large (0.5), 8 small (0.25) tiling. Sum = 3.0.
    Method 'grid': 10 squares of 0.316? No, 0.316 doesn't tile.
    Let's try to generate a config close to the optimal equal-size packing.
    """
    if method == 'heuristic':
        centers = np.zeros((n, 2))
        sides = np.zeros(n)
        
        # 2 Large
        centers[0] = [0.25, 0.25]
        sides[0] = 0.5
        centers[1] = [0.75, 0.25]
        sides[1] = 0.5
        
        # 8 Small
        # Top-left quadrant (x:0-0.5, y:0.5-1.0)
        for k in range(4):
            col = k % 2
            row = k // 2
            centers[2+k, 0] = (col + 0.5) * 0.25
            centers[2+k, 1] = 0.5 + (row + 0.5) * 0.25
            sides[2+k] = 0.25
            
        # Top-right quadrant (x:0.5-1.0, y:0.5-1.0)
        for k in range(4):
            col = k % 2
            row = k // 2
            centers[6+k, 0] = 0.5 + (col + 0.5) * 0.25
            centers[6+k, 1] = 0.5 + (row + 0.5) * 0.25
            sides[6+k] = 0.25
            
        return centers, sides
        
    elif method == 'equal':
        # Try to pack 10 squares of similar size.
        # 3x3 grid of 0.316 is too big for 10.
        # Let's try 0.3.
        # 0.3 * 3 = 0.9.
        # We can fit 3 in x and 3 in y.
        # Positions:
        # 0.15, 0.45, 0.75
        # This gives 9 squares.
        # Where does the 10th go?
        # Maybe shrink them all?
        # If s=0.3, sum = 3.0.
        # If s=0.31, sum = 3.1.
        # Can we fit 10 squares of 0.31?
        # Area = 10 * 0.0961 = 0.961 < 1.
        # It might be possible.
        # Let's initialize with a jittered grid of 0.31.
        centers = np.zeros((n, 2))
        sides = np.full(n, 0.31)
        
        # 3x3 grid
        for i in range(9):
            col = i % 3
            row = i // 3
            centers[i, 0] = 0.155 + col * 0.31
            centers[i, 1] = 0.155 + row * 0.31
            
        # 10th square?
        # This leaves no room.
        # So this init is likely invalid.
        # We need a valid init.
        # Let's stick to the heuristic which is known valid.
        return _generate_initial_config(n, 'heuristic')

def squares_in_square10() -> tuple:
    """
    Returns:
        centers: np.ndarray of shape (10, 2)
        sides:   np.ndarray of shape (10,)
    """
    n = 10
    tol = 1e-7
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin
    
    best_centers = None
    best_sides = None
    best_val = -1.0
    
    # Try multiple seeds/restarts
    # Start with the known good heuristic
    c_init, s_init = _generate_initial_config(n, 'heuristic')
    
    # Verify it's valid
    obj, viol = _evaluate(c_init, s_init, tol)
    if viol < 1e-7:
        best_val = obj
        best_centers = c_init.copy()
        best_sides = s_init.copy()
    else:
        # Fallback to a trivial valid config if heuristic fails (shouldn't happen)
        best_centers = np.zeros((n, 2))
        best_sides = np.full(n, 0.1)
        for i in range(n):
            best_centers[i, 0] = 0.1 + 0.1 * (i % 10)
            best_centers[i, 1] = 0.1
        best_val = np.sum(best_sides)
        
    # Run local search from this start
    rng = np.random.default_rng(42)
    bc, bs, bv = _local_search(c_init, s_init, start_time, deadline, rng, tol)
    if bv > best_val:
        best_val = bv
        best_centers = bc
        best_sides = bs
        
    # Try a few more random restarts if time permits
    # The heuristic is a local optimum. Can we beat 3.0?
    # Let's try starting from a "noisy" version of the heuristic.
    for i in range(5):
        if time.time() > deadline - 30:
            break
            
        c_rand = best_centers + rng.normal(0, 0.02, size=(n, 2))
        s_rand = np.clip(best_sides + rng.normal(0, 0.01, size=n), 0.01, 1.0)
        
        # Ensure validity of start?
        # _local_search handles invalid starts by rejecting moves, but it starts from curr state.
        # If curr state is invalid, best_obj is -inf, so it will accept any valid move.
        # So it's safe to start from invalid states.
        
        bc, bs, bv = _local_search(c_rand, s_rand, start_time, deadline, rng, tol)
        if bv > best_val:
            best_val = bv
            best_centers = bc
            best_sides = bs
            
    # Final check
    if best_centers is None:
        return _generate_initial_config(n, 'heuristic')
        
    # Ensure final result is strictly valid
    final_obj, final_viol = _evaluate(best_centers, best_sides, tol)
    if final_viol > 1e-7:
        # If the best found is invalid (shouldn't happen as we only track valid bests),
        # fall back to the initial heuristic.
        return _generate_initial_config(n, 'heuristic')
        
    return best_centers, best_sides

# EVOLVE-BLOCK-END
