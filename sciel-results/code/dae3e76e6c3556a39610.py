import numpy as np
import time
from scipy.optimize import minimize

def _is_valid(centers, sides, tol=1e-7):
    """Check if the packing is valid."""
    n = len(sides)
    if centers.shape != (n, 2) or sides.shape != (n,):
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    
    # Check inside unit square
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s/2 < -tol or cx + s/2 > 1 + tol:
            return False
        if cy - s/2 < -tol or cy + s/2 > 1 + tol:
            return False
            
    # Check no overlaps
    for i in range(n):
        for j in range(i+1, n):
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            min_dx = (sides[i] + sides[j])/2 - tol
            min_dy = (sides[i] + sides[j])/2 - tol
            if dx < min_dx and dy < min_dy:
                return False
    return True

def _objective(centers, sides):
    """Return sum of sides if valid, else a large penalty."""
    n = len(sides)
    tol = 1e-7
    
    # Check bounds
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s/2 < -tol or cx + s/2 > 1 + tol:
            return -1e6
        if cy - s/2 < -tol or cy + s/2 > 1 + tol:
            return -1e6
        if s <= 0:
            return -1e6
    
    # Check overlaps
    for i in range(n):
        for j in range(i+1, n):
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            min_dx = (sides[i] + sides[j])/2 - tol
            min_dy = (sides[i] + sides[j])/2 - tol
            if dx < min_dx and dy < min_dy:
                return -1e6
    
    return np.sum(sides)

def _smooth_penalty(centers, sides, tol=1e-7, beta=50.0):
    """
    Smooth surrogate for the constraint violations.
    Uses soft-min (log-sum-exp) to approximate the minimum separation.
    Returns a scalar penalty that is 0 when all constraints are satisfied (approximately).
    """
    n = len(sides)
    penalty = 0.0
    
    # Boundary penalties: cx - s/2 >= 0 and cx + s/2 <= 1
    # Violation: max(0, -(cx - s/2)) + max(0, (cx + s/2 - 1))
    # Smooth approx: -1/beta * log(sum(exp(-beta * margin)))
    # But simpler: just use softplus on the violation amount.
    # Let's use a simple quadratic penalty for boundaries for stability in the smooth phase,
    # or stick to the log-sum-exp for overlaps only.
    
    # Overlap penalty:
    # For each pair (i,j), we need dx >= (si+sj)/2 OR dy >= (si+sj)/2.
    # Violation is if dx < (si+sj)/2 AND dy < (si+sj)/2.
    # Let gap_x = dx - (si+sj)/2
    # Let gap_y = dy - (si+sj)/2
    # We want to penalize when min(gap_x, gap_y) < 0.
    # min(a, b) approx = -1/beta * log(exp(-beta*a) + exp(-beta*b))
    # If min < 0, we want penalty > 0.
    # So penalty = -1/beta * log(exp(-beta*gap_x) + exp(-beta*gap_y))
    # If gap_x, gap_y > 0, this is small negative (close to 0).
    # If gap_x, gap_y < 0, this is large positive.
    
    for i in range(n):
        for j in range(i+1, n):
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            t = (sides[i] + sides[j])/2
            
            gap_x = dx - t
            gap_y = dy - t
            
            # Soft min of gap_x and gap_y
            # To avoid overflow, use the stable log-sum-exp
            m = min(gap_x, gap_y)
            if m > 500 / beta:
                # Both gaps are large positive, penalty is negligible
                continue
            
            # log(exp(-beta*gap_x) + exp(-beta*gap_y))
            # = -beta*m + log(exp(-beta*(gap_x - m)) + exp(-beta*(gap_y - m)))
            val = -beta * m + np.log(np.exp(-beta * (gap_x - m)) + np.exp(-beta * (gap_y - m)))
            penalty += val / beta
    
    # Boundary penalties
    for i in range(n):
        s = sides[i]
        # Left: cx - s/2 >= 0 -> gap = cx - s/2
        # Right: 1 - (cx + s/2) >= 0 -> gap = 1 - cx - s/2
        # Top/Bottom similar
        
        gaps = [
            centers[i, 0] - s/2,
            1 - centers[i, 0] - s/2,
            centers[i, 1] - s/2,
            1 - centers[i, 1] - s/2
        ]
        
        # We want to penalize if any gap < 0.
        # Use soft-min of all gaps.
        m = min(gaps)
        if m > 500 / beta:
            continue
            
        val = -beta * m + np.log(np.sum(np.exp(-beta * (np.array(gaps) - m))))
        penalty += val / beta
        
    return penalty

def _local_search_smooth(centers, sides, time_limit=10.0, beta_start=10.0, beta_end=100.0):
    """
    Uses scipy.optimize.minimize with a smooth penalty function.
    We minimize -sum(sides) + penalty(centers, sides).
    """
    start_time = time.time()
    n = len(sides)
    
    # Flatten variables: [cx0, cy0, s0, cx1, cy1, s1, ...]
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = centers[i, 0]
        x0[3*i+1] = centers[i, 1]
        x0[3*i+2] = sides[i]
    
    best_x = x0.copy()
    best_score = _objective(centers, sides)
    
    if best_score < 0:
        return centers, sides, best_score
    
    def unpack(x):
        c = np.zeros((n, 2))
        s = np.zeros(n)
        for i in range(n):
            c[i, 0] = x[3*i]
            c[i, 1] = x[3*i+1]
            s[i] = x[3*i+2]
        return c, s
    
    def objective_fn(x):
        if time.time() - start_time > time_limit:
            return 1e6
        c, s = unpack(x)
        
        # Hard constraints for bounds to keep search space sane
        # If sides are negative or too large, return large penalty
        if np.any(s < 0.01) or np.any(s > 1.0):
            return 1e6
            
        # Clamp centers to keep them in bounds roughly
        # But let the penalty handle the exact constraint
        # Actually, clamping might break the smoothness. 
        # Let's rely on the penalty.
        
        pen = _smooth_penalty(c, s, beta=beta_start)
        val = -np.sum(s) + pen
        return val
    
    # Run multiple steps with increasing beta (annealing the smoothness)
    betas = np.linspace(beta_start, beta_end, 5)
    
    for beta in betas:
        if time.time() - start_time > time_limit:
            break
        
        # Update the objective function to use current beta
        # We need to pass beta to _smooth_penalty
        # Since objective_fn captures beta_start, we need to refactor or use a closure that reads a mutable variable.
        # For simplicity, let's just create a new function for each beta.
        
        def obj(x, b=beta):
            if time.time() - start_time > time_limit:
                return 1e6
            c, s = unpack(x)
            if np.any(s < 0.01) or np.any(s > 1.0):
                return 1e6
            pen = _smooth_penalty(c, s, beta=b)
            return -np.sum(s) + pen
        
        try:
            res = minimize(obj, x0, method='L-BFGS-B', options={'maxiter': 200, 'disp': False})
            if res.success or res.fun < objective_fn(x0):
                x0 = res.x
        except Exception:
            pass
        
        # Update best
        c, s = unpack(x0)
        # Clip sides to be positive
        s = np.clip(s, 0.01, 1.0)
        # Clip centers
        for i in range(n):
            c[i, 0] = np.clip(c[i, 0], s[i]/2, 1 - s[i]/2)
            c[i, 1] = np.clip(c[i, 1], s[i]/2, 1 - s[i]/2)
        
        score = _objective(c, s)
        if score > best_score:
            best_score = score
            best_x = x0.copy()
            
    c, s = unpack(best_x)
    return c, s, best_score

def _generate_initial_config():
    """Generate the known 3.0 packing."""
    n = 10
    centers = np.zeros((n, 2))
    sides = np.zeros(n)
    
    # Bottom half: 2 squares of side 0.5
    sides[0] = 0.5
    sides[1] = 0.5
    centers[0, 0] = 0.25
    centers[0, 1] = 0.25
    centers[1, 0] = 0.75
    centers[1, 1] = 0.25
    
    # Top half: 8 squares of side 0.25
    sides[2] = 0.25
    centers[2, 0] = 0.125
    centers[2, 1] = 0.625
    
    sides[3] = 0.25
    centers[3, 0] = 0.375
    centers[3, 1] = 0.625
    
    sides[4] = 0.25
    centers[4, 0] = 0.125
    centers[4, 1] = 0.875
    
    sides[5] = 0.25
    centers[5, 0] = 0.375
    centers[5, 1] = 0.875
    
    sides[6] = 0.25
    centers[6, 0] = 0.625
    centers[6, 1] = 0.625
    
    sides[7] = 0.25
    centers[7, 0] = 0.875
    centers[7, 1] = 0.625
    
    sides[8] = 0.25
    centers[8, 0] = 0.625
    centers[8, 1] = 0.875
    
    sides[9] = 0.25
    centers[9, 0] = 0.875
    centers[9, 1] = 0.875
    
    return centers, sides

def squares_in_square10() -> tuple:
    """
    Returns:
        centers: np.ndarray of shape (10, 2)
        sides:   np.ndarray of shape (10,)
    
    Uses a hybrid approach:
    1. Start with the known 3.0 packing.
    2. Use smooth optimization (L-BFGS-B with soft-min penalty) to refine and potentially improve.
    3. Use simulated annealing for broader exploration if time permits.
    """
    deadline = time.time() + 600  # Safety margin
    
    # Start with known 3.0 packing
    centers, sides = _generate_initial_config()
    best_centers = centers.copy()
    best_sides = sides.copy()
    best_score = _objective(best_centers, best_sides)
    
    # Phase 1: Smooth local search (Gradient based)
    # Try a few different starting points (perturbed)
    rng = np.random.RandomState(42)
    
    for attempt in range(5):
        if time.time() > deadline - 30:
            break
        
        # Perturb the initial config
        if attempt == 0:
            c_init, s_init = centers.copy(), sides.copy()
        else:
            c_init = centers + rng.normal(0, 0.05, (10, 2))
            s_init = sides * (1 + rng.normal(0, 0.1, 10))
            s_init = np.clip(s_init, 0.05, 0.6)
            # Clamp centers
            for i in range(10):
                c_init[i, 0] = np.clip(c_init[i, 0], s_init[i]/2, 1 - s_init[i]/2)
                c_init[i, 1] = np.clip(c_init[i, 1], s_init[i]/2, 1 - s_init[i]/2)
        
        # Run smooth local search
        time_left = deadline - time.time() - 30
        if time_left < 5:
            break
            
        c_new, s_new, score_new = _local_search_smooth(
            c_init, s_init, 
            time_limit=min(20.0, time_left), 
            beta_start=5.0, 
            beta_end=50.0
        )
        
        if score_new > best_score:
            best_score = score_new
            best_centers = c_new.copy()
            best_sides = s_new.copy()
            
    # Phase 2: Simulated Annealing (if time remains)
    # This helps escape local minima found by the gradient method
    if time.time() < deadline - 10:
        # Implement a quick SA loop
        current_centers = best_centers.copy()
        current_sides = best_sides.copy()
        current_score = best_score
        
        T = 0.02
        T_min = 1e-6
        cooling = 0.999
        rng_sa = np.random.RandomState(123)
        
        while T > T_min and time.time() < deadline - 10:
            idx = rng_sa.randint(10)
            pert_type = rng_sa.random()
            
            new_centers = current_centers.copy()
            new_sides = current_sides.copy()
            
            if pert_type < 0.5:
                # Move
                new_centers[idx, 0] += rng_sa.normal(0, 0.01)
                new_centers[idx, 1] += rng_sa.normal(0, 0.01)
            elif pert_type < 0.8:
                # Resize
                new_sides[idx] *= (1 + rng_sa.normal(0, 0.05))
                new_sides[idx] = np.clip(new_sides[idx], 0.01, 0.6)
            else:
                # Swap
                idx2 = rng_sa.randint(10)
                if idx2 != idx:
                    new_centers[idx], new_centers[idx2] = new_centers[idx2].copy(), new_centers[idx].copy()
                    new_sides[idx], new_sides[idx2] = new_sides[idx2], new_sides[idx]
            
            # Clamp
            for i in range(10):
                s = new_sides[i]
                new_centers[i, 0] = np.clip(new_centers[i, 0], s/2, 1 - s/2)
                new_centers[i, 1] = np.clip(new_centers[i, 1], s/2, 1 - s/2)
            
            new_score = _objective(new_centers, new_sides)
            
            if new_score >= current_score:
                current_centers = new_centers
                current_sides = new_sides
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_centers = new_centers.copy()
                    best_sides = new_sides.copy()
            else:
                delta = new_score - current_score
                if delta > -100: # Ignore large penalties
                    prob = np.exp(delta / T)
                    if rng_sa.random() < prob:
                        current_centers = new_centers
                        current_sides = new_sides
                        current_score = new_score
                        
            T *= cooling
            
    return best_centers, best_sides
