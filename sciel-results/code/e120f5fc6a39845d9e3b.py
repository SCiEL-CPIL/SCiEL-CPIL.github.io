import numpy as np
import time

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # 1. Structured Initialization: 1 center + 8 inner ring + 17 outer ring
    # This mimics the optimal dense packing structure better than a grid.
    # The grid leaves corners empty and has inefficient spacing.
    # We will define positions and initial radii.
    
    # Center
    center = np.array([0.5, 0.5])
    
    # Inner ring (8 circles)
    r1_dist = 0.18
    angles1 = np.linspace(0, 2*np.pi, 8, endpoint=False)
    ring1_pos = center + r1_dist * np.column_stack([np.cos(angles1), np.sin(angles1)])
    
    # Outer ring (17 circles)
    # 17 is 26 - 1 - 8. 
    # We distribute them roughly evenly on a larger circle, but we might need to split 
    # or adjust. Actually, 17 is a lot for one ring. 
    # Let's try 1 center, 8 inner, 17 outer.
    # The outer ring will be crowded. 
    # Alternative: 1 center, 6 inner, 9 middle, 10 outer?
    # Let's stick to a simple radial layout for initialization, then let the optimizer fix it.
    
    # Better init: 
    # 1 center
    # 8 around it
    # 17 around that? That's 26.
    # The outer ring of 17 will have radius ~ 0.45.
    # Circumference ~ 2.8. 17 circles -> spacing ~ 0.16. Radius ~ 0.08.
    # Inner 8: radius ~ 0.18. Spacing ~ 0.14. Radius ~ 0.07.
    # Center: limited by inner ring. Dist 0.18. R_center ~ 0.09.
    
    # Let's refine the radii estimates for a valid start.
    # If we start with valid non-overlapping circles, the local search can grow them.
    
    # Let's use a slightly more robust init: 
    # Generate 26 points using a spiral or grid, but perturbed.
    # Actually, the "ring" idea is good for the *topology* of the solution.
    # Let's create the points:
    
    pts = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # Center
    pts[0] = [0.5, 0.5]
    
    # Ring 1: 8 points
    r1 = 0.2
    for i in range(8):
        angle = i * np.pi / 4
        pts[i+1] = [0.5 + r1*np.cos(angle), 0.5 + r1*np.sin(angle)]
        
    # Ring 2: 17 points
    # We need to fit 17 points. 
    # Let's distribute them on a circle of radius 0.42
    r2 = 0.42
    # To avoid symmetry issues that might trap the local search, add a small random offset to angles
    angles2 = np.linspace(0, 2*np.pi, 17, endpoint=False) + np.random.uniform(-0.1, 0.1, 17)
    for i in range(17):
        pts[i+9] = [0.5 + r2*np.cos(angles2[i]), 0.5 + r2*np.sin(angles2[i])]
        
    # Initial radii: very small to ensure validity
    radii[:] = 0.01
    
    # 2. Local Search (Simulated Annealing / Gradient Ascent)
    # Objective: Maximize sum(radii)
    # Constraints:
    #   1. x - r >= 0, x + r <= 1
    #   2. y - r >= 0, y + r <= 1
    #   3. dist(i, j) >= r_i + r_j
    
    # We will use a penalty method or a projection method.
    # Given the non-linearity, a simple random search with acceptance criteria (SA) is robust.
    # But we can be smarter: 
    # At each step, pick a circle, move it slightly, or grow it slightly.
    # If constraints are violated, revert or shrink.
    
    # Let's use a "Grow and Push" strategy.
    # 1. Try to increase all radii by a small epsilon.
    # 2. If overlap occurs, push circles apart.
    # This is complex.
    
    # Alternative: Standard Simulated Annealing on (x, y, r)
    # State: (N, 3)
    # Move: Perturb one circle's x, y, or r.
    # Score: Sum(r) - Penalty(Violations)
    
    state = np.hstack([pts, radii[:, None]])
    best_state = state.copy()
    best_score = np.sum(state[:, 2])
    
    # Precompute indices for pairs
    i_idx, j_idx = np.triu_indices(N, k=1)
    pairs = np.column_stack([i_idx, j_idx])
    
    def get_score(s):
        # s is (N, 3)
        x = s[:, 0]
        y = s[:, 1]
        r = s[:, 2]
        
        # Boundary violations
        # x < r, x > 1-r, y < r, y > 1-r
        # We want r <= min(x, 1-x, y, 1-y)
        # So max_r_allowed = min(x, 1-x, y, 1-y)
        # Violation = max(0, r - max_r_allowed)
        max_r_bound = np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y))
        bound_viol = np.sum(np.maximum(0, r - max_r_bound))
        
        # Pairwise violations
        # dist(i, j) < r_i + r_j
        dx = x[i_idx] - x[j_idx]
        dy = y[i_idx] - y[j_idx]
        dist = np.sqrt(dx*dx + dy*dy)
        sum_r = r[i_idx] + r[j_idx]
        pair_viol = np.sum(np.maximum(0, sum_r - dist))
        
        total_viol = bound_viol + pair_viol
        
        # Score is sum of radii, but we need to ensure validity.
        # The evaluator returns 0 if invalid.
        # So we must return a VALID configuration.
        # Therefore, the search must stay in the valid space or project back.
        
        # Strategy: 
        # Keep a "current" valid state.
        # Propose a new state.
        # If new state is valid, accept if better.
        # If new state is invalid, reject (or use soft penalty to guide back, but simpler to just reject invalid moves).
        
        # However, rejecting all invalid moves can get stuck.
        # Better: Use a penalty in the *search* score, but only return *valid* states.
        # If we end with an invalid state, we must project it.
        
        # Let's define a "search_score" that penalizes violations heavily.
        penalty_weight = 10.0
        return np.sum(r) - penalty_weight * total_viol, total_viol

    current_score, _ = get_score(state)
    
    T = 0.05 # Initial temperature
    cooling = 0.999
    
    # Time limit check
    while time.time() < deadline:
        # Pick a random circle
        k = np.random.randint(N)
        
        # Pick a random move type:
        # 0: Move x
        # 1: Move y
        # 2: Grow/Shrink r
        move_type = np.random.randint(3)
        
        new_state = state.copy()
        
        # Step size depends on temperature
        step = T * 0.1
        
        if move_type == 0:
            new_state[k, 0] += np.random.normal(0, step)
        elif move_type == 1:
            new_state[k, 1] += np.random.normal(0, step)
        else:
            # Grow radius
            # Only allow growing if it seems likely to be valid?
            # Or just try and check.
            new_state[k, 2] += np.random.normal(0, step)
            
        # Ensure radius is positive
        new_state[k, 2] = max(0.001, new_state[k, 2])
        
        new_score, new_viol = get_score(new_state)
        
        # Acceptance criteria
        # If new is valid (new_viol == 0) and better, accept.
        # If new is invalid, we might still accept if it's "less invalid" and T is high?
        # But we need to return a valid state.
        # Let's stick to: Only accept if valid. If invalid, reject.
        # This is slow but safe.
        
        # Optimization: Check validity faster?
        # get_score is vectorized, so it's fast enough for 26 circles.
        
        if new_viol < 1e-6: # Valid
            if new_score > current_score:
                state = new_state
                current_score = new_score
                if current_score > best_score:
                    best_state = state.copy()
                    best_score = current_score
            else:
                # Accept worse valid moves with probability exp((new-old)/T) to escape local minima?
                # Since we are maximizing, and all valid moves are "good", 
                # moving to a slightly worse valid position might allow a bigger grow later.
                delta = new_score - current_score
                if np.random.random() < np.exp(delta / T):
                    state = new_state
                    current_score = new_score
        else:
            # Invalid move. Reject.
            # Could also use a "repair" step, but for now, reject.
            pass
        
        T *= cooling
        if T < 0.001:
            T = 0.001 # Keep some noise to avoid freezing completely, or restart?
            
    # Final check: ensure best_state is valid
    # If best_state is somehow invalid (shouldn't be, as we only track valid improvements), project it.
    # Projection: reduce radii until valid.
    # Simple projection:
    # Sort by radius, reduce largest ones?
    # Or just scale down all radii by a factor until valid.
    
    # Let's verify best_state validity
    x = best_state[:, 0]
    y = best_state[:, 1]
    r = best_state[:, 2]
    
    # Check bounds
    max_r_bound = np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y))
    # If any r > max_r_bound, it's invalid.
    if np.any(r > max_r_bound + 1e-9):
        # Scale down
        # Find the max violation ratio
        # Actually, just reduce all radii by a small factor iteratively?
        # Or set r = min(r, max_r_bound) and then fix overlaps.
        # This is getting complex. 
        # Since we only accept valid states in the loop, best_state SHOULD be valid.
        # But floating point errors might creep in.
        # Let's add a safety projection.
        
        # Reduce radii until valid
        scale = 1.0
        for _ in range(10):
            r_check = r * scale
            max_r_bound = np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y))
            if np.any(r_check > max_r_bound + 1e-9):
                scale *= 0.9
                continue
            
            # Check overlaps
            dx = x[i_idx] - x[j_idx]
            dy = y[i_idx] - y[j_idx]
            dist = np.sqrt(dx*dx + dy*dy)
            sum_r = r_check[i_idx] + r_check[j_idx]
            if np.any(sum_r > dist + 1e-9):
                scale *= 0.9
                continue
            
            # Valid
            break
            
        r = r * scale
        best_state[:, 2] = r
    
    return best_state

# EVOLVE-BLOCK-END
