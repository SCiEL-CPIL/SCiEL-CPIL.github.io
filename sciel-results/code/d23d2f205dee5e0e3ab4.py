"""Multi-Start Nelder-Mead optimizer for BBOB-style Schaffers F7 (f17)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-Start Nelder-Mead with deterministic starting points."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s kill

        dim = self.dim
        budget = self.budget
        
        # 1. Generate 10 distinct deterministic starting points
        # Using a simple deterministic pattern to cover the space
        # We want points in [-4, 4]^5 as per the prompt's hint about minimizer location
        # But we should probably cover the whole box [-5, 5] or at least [-4, 4]
        # Let's use a grid-like or random-but-deterministic spread.
        # Since we have 10 starts, we can just use hardcoded diverse points 
        # or a deterministic LHS (Latin Hypercube) style.
        
        # Let's create 10 points manually or via a simple deterministic rule
        # that spreads them out.
        # Point 0: Center
        # Points 1-9: Distributed around
        
        # A robust way: Use a fixed set of points that are well-spread.
        # Or generate them deterministically based on index.
        
        starts = []
        # Point 0: Origin
        starts.append(np.zeros(dim))
        
        # For the other 9, let's use a deterministic pseudo-random sequence 
        # seeded by the index to ensure they are distinct and spread.
        # Or better, just hardcode a diverse set.
        # Let's use a simple grid for the first few and then random-ish.
        
        # Actually, for dim=5, 10 starts is quite few.
        # Let's use a deterministic set of points that covers the corners and mid-points.
        
        # 1. Center
        # 2. (4, 0, 0, 0, 0)
        # 3. (-4, 0, 0, 0, 0)
        # 4. (0, 4, 0, 0, 0)
        # 5. (0, -4, 0, 0, 0)
        # 6. (0, 0, 4, 0, 0)
        # 7. (0, 0, -4, 0, 0)
        # 8. (0, 0, 0, 4, 0)
        # 9. (0, 0, 0, -4, 0)
        
        # This covers the axes. What about diagonal?
        # Let's mix it.
        
        # Alternative: Use a deterministic Halton sequence or similar low-discrepancy sequence.
        # But simple is better for reliability.
        
        # Let's just use 10 fixed points that are spread out.
        # p0: [0, 0, 0, 0, 0]
        # p1: [4, 4, 4, 4, 4]
        # p2: [-4, -4, -4, -4, -4]
        # p3: [4, -4, 4, -4, 4]
        # p4: [-4, 4, -4, 4, -4]
        # p5: [4, 4, -4, -4, 4]
        # p6: [-4, -4, 4, 4, -4]
        # p7: [4, -4, -4, 4, -4]
        # p8: [-4, 4, 4, -4, 4]
        # p9: [0, 0, 4, 0, -4] # Just a random distinct one
        
        # Let's generate them programmatically to be safe and clean.
        # We'll use a simple deterministic rule:
        # For i in range(10):
        #   x_j = 4 * cos(i * j * 0.5) # Something like that?
        
        # Let's stick to the axis-aligned + corners approach for simplicity and spread.
        
        points = [
            np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([4.0, 4.0, 4.0, 4.0, 4.0]),
            np.array([-4.0, -4.0, -4.0, -4.0, -4.0]),
            np.array([4.0, -4.0, 4.0, -4.0, 4.0]),
            np.array([-4.0, 4.0, -4.0, 4.0, -4.0]),
            np.array([4.0, 4.0, -4.0, -4.0, 4.0]),
            np.array([-4.0, -4.0, 4.0, 4.0, -4.0]),
            np.array([4.0, -4.0, -4.0, 4.0, -4.0]),
            np.array([-4.0, 4.0, 4.0, -4.0, 4.0]),
            np.array([2.0, -2.0, 2.0, -2.0, 2.0])
        ]
        
        # Ensure they are within bounds
        points = [np.clip(p, -5, 5) for p in points]
        
        n_starts = len(points)
        calls_per_start = budget // n_starts
        # If there's a remainder, give it to the first few starts
        remainder = budget % n_starts
        
        best_x_global = points[0].copy()
        best_f_global = float('inf')
        calls_made = 0
        
        # Nelder-Mead parameters
        # Initial simplex size
        initial_step = 0.5
        
        for i in range(n_starts):
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break
                
            # Determine budget for this start
            current_budget = calls_per_start + (1 if i < remainder else 0)
            
            # Initialize simplex
            x0 = points[i]
            f0 = func(x0)
            calls_made += 1
            if f0 < best_f_global:
                best_f_global = f0
                best_x_global = x0.copy()
            
            # Create initial simplex
            simplex = [x0.copy()]
            fs = [f0]
            
            for j in range(dim):
                xj = x0.copy()
                xj[j] += initial_step
                xj = np.clip(xj, -5, 5)
                fj = func(xj)
                calls_made += 1
                if fj < best_f_global:
                    best_f_global = fj
                    best_x_global = xj.copy()
                simplex.append(xj)
                fs.append(fj)
                
                if calls_made >= budget:
                    break
            
            if calls_made >= budget:
                break
                
            # Nelder-Mead iteration
            # We need to keep track of the simplex and its fitnesses
            # and update them until we run out of budget for this start.
            
            while calls_made < budget and time.time() < deadline:
                # Sort simplex by fitness
                order = np.argsort(fs)
                simplex = [simplex[idx] for idx in order]
                fs = [fs[idx] for idx in order]
                
                # Calculate centroid of all but the worst point
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                alpha = 1.0
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = func(x_ref)
                calls_made += 1
                if f_ref < best_f_global:
                    best_f_global = f_ref
                    best_x_global = x_ref.copy()
                    
                if f_ref < fs[0]:
                    # Expansion
                    gamma = 2.0
                    x_exp = centroid + gamma * (centroid - simplex[-1])
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = func(x_exp)
                    calls_made += 1
                    if f_exp < best_f_global:
                        best_f_global = f_exp
                        best_x_global = x_exp.copy()
                        
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        fs[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        fs[-1] = f_ref
                elif f_ref < fs[-2]:
                    # Accept reflection
                    simplex[-1] = x_ref
                    fs[-1] = f_ref
                else:
                    # Contraction
                    rho = 0.5
                    x_con = centroid + rho * (simplex[-1] - centroid)
                    x_con = np.clip(x_con, -5, 5)
                    f_con = func(x_con)
                    calls_made += 1
                    if f_con < best_f_global:
                        best_f_global = f_con
                        best_x_global = x_con.copy()
                        
                    if f_con < f_ref:
                        simplex[-1] = x_con
                        fs[-1] = f_con
                    else:
                        # Shrink
                        for k in range(1, dim + 1):
                            simplex[k] = simplex[0] + 0.5 * (simplex[k] - simplex[0])
                            # Re-evaluate? 
                            # Standard NM shrinks all points except the best.
                            # We need to evaluate the new points.
                            # This consumes budget.
                            # If we don't have enough budget, we might break.
                            if calls_made >= budget:
                                break
                            f_shrunk = func(simplex[k])
                            calls_made += 1
                            if f_shrunk < best_f_global:
                                best_f_global = f_shrunk
                                best_x_global = simplex[k].copy()
                            fs[k] = f_shrunk
                            if calls_made >= budget:
                                break
            
            # Move to next start
            if calls_made >= budget:
                break
                
        return best_x_global, best_f_global

# EVOLVE-BLOCK-END
