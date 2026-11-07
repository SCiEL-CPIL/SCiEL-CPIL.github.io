import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        d = self.dim
        budget = self.budget
        best_x = np.zeros(d)
        best_f = np.inf

        def _time_left():
            return deadline - time.time()

        try:
            # Phase 1: Coarse grid to find the step boundary (enter the ellipsoidal region)
            # The step ellipsoidal function has a flat plateau outside the ellipsoid and a 
            # smooth ellipsoidal function inside. We need to cross the boundary early.
            
            # Use a stratified sample that covers the box systematically
            # We'll use a coarse grid in each dimension and evaluate all combinations
            # For dim=5, a grid of size 4 per dim gives 4^5 = 1024 points, too many
            # A grid of size 3 per dim gives 3^5 = 243 points, which fits in budget
            # But we need to be smarter. Let's use a Halton-like sequence or stratified sampling.
            
            # Actually, let's use a deterministic approach: evaluate at the center first,
            # then expand outward in a structured way.
            
            # First, evaluate at the center
            center = np.zeros(d)
            f_center = func(center)
            budget -= 1
            best_f = f_center
            best_x = center.copy()
            
            # Check if we're on the plateau (flat region)
            # Sample a few points at increasing distances to detect the boundary
            # If the function value changes, we've found the boundary
            
            # Use a spiral-like or concentric sampling to find the boundary
            # Start with small radius and expand
            
            radii = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
            found_inside = False
            
            for r in radii:
                if budget <= 0 or _time_left() < 5.0:
                    break
                    
                # Sample points on a sphere of radius r in each coordinate direction
                # and also at random directions
                n_samples = min(20, budget)
                if n_samples <= 0:
                    break
                    
                # Deterministic directions: coordinate axes and diagonals
                directions = []
                # Coordinate axes
                for i in range(d):
                    for sign in [1, -1]:
                        dir_vec = np.zeros(d)
                        dir_vec[i] = sign
                        directions.append(dir_vec)
                # Diagonals
                for sign_pattern in [[1]*d, [-1]*d, [1,-1,1,-1,1][:d], [-1,1,-1,1,-1][:d]]:
                    directions.append(np.array(sign_pattern))
                
                # Add some random directions
                rng = np.random.default_rng(42)
                for _ in range(min(10, len(directions))):
                    dir_vec = rng.standard_normal(d)
                    norm = np.linalg.norm(dir_vec)
                    if norm > 1e-10:
                        dir_vec /= norm
                        directions.append(dir_vec)
                
                for dir_vec in directions:
                    if budget <= 0 or _time_left() < 5.0:
                        break
                    x = center + r * dir_vec
                    x = np.clip(x, -5, 5)
                    fx = func(x)
                    budget -= 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    if fx != f_center:
                        found_inside = True
                        break
                if found_inside:
                    break
            
            # Phase 2: Local optimization using a (1+lambda)-ES with adaptive sigma
            # Start from the best point found
            x = best_x.copy()
            f = best_f
            
            # If we found the inside, use a smaller sigma; otherwise, use a larger one
            if found_inside:
                sigma = 0.5
            else:
                sigma = 2.0
            
            # (1+10)-ES with plateau escape
            lam = 10
            no_improve = 0
            max_no_improve = 15
            rng = np.random.default_rng(123)
            
            while budget > 0 and _time_left() > 2.0:
                # Generate population
                pop = []
                f_pop = []
                for i in range(lam):
                    if budget <= 0 or _time_left() < 2.0:
                        break
                    y = x + sigma * rng.standard_normal(d)
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    budget -= 1
                    pop.append(y)
                    f_pop.append(fy)
                    if fy < f:
                        f = fy
                        x = y.copy()
                        best_f = f
                        best_x = x.copy()
                
                if not pop:
                    break
                
                # Check for improvement
                best_in_pop = min(f_pop)
                if best_in_pop < f:
                    no_improve = 0
                    sigma *= 1.1
                else:
                    no_improve += 1
                    sigma *= 0.9
                    if no_improve > max_no_improve:
                        # Plateau escape: try a large jump
                        if budget > 0 and _time_left() > 2.0:
                            y_jump = x + 3.0 * rng.standard_normal(d)
                            y_jump = np.clip(y_jump, -5, 5)
                            fy_jump = func(y_jump)
                            budget -= 1
                            if fy_jump < f:
                                x = y_jump.copy()
                                f = fy_jump
                                best_f = f
                                best_x = x.copy()
                                sigma = 1.0
                                no_improve = 0
                            else:
                                sigma = max(0.1, sigma * 0.5)
                                no_improve = 0
            
        except Exception as e:
            pass
        
        return best_x, best_f
