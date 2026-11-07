import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.start_time = None
        self.best_x = None
        self.best_f = np.inf

    def __call__(self, func):
        self.start_time = time.time()
        d = self.dim
        N = self.budget
        deadline = self.start_time + 285.0  # Safety margin for 300s limit

        def time_check():
            return time.time() < deadline

        def safe_eval(x):
            if self.calls >= N or not time_check():
                return np.inf
            self.calls += 1
            try:
                # Ensure x is a 1D array of floats
                x = np.asarray(x, dtype=float).flatten()
                # Clip to box [-5, 5] to be safe, though func should do it
                x = np.clip(x, -5.0, 5.0)
                val = float(func(x))
                return val
            except Exception:
                return np.inf

        def update_best(x, f):
            if f < self.best_f:
                self.best_f = f
                self.best_x = x.copy()

        # Initial evaluation at origin
        x0 = np.zeros(d)
        f0 = safe_eval(x0)
        update_best(x0, f0)

        rng = np.random.default_rng(42)
        
        # Generate diverse starting points for global exploration
        # 1. Origin
        # 2. Random points in [-4, 4]
        # 3. Points along axes to help find the valley direction
        start_points = [x0.copy()]
        
        # Add random points
        for _ in range(8):
            start_points.append(rng.uniform(-4, 4, d))
            
        # Add points along axes (helps if valley is aligned with an axis)
        for i in range(d):
            for sign in [-1, 1]:
                pt = np.zeros(d)
                pt[i] = 3.0 * sign
                start_points.append(pt)

        # Phase 1: Global Exploration with Adaptive Local Refinement
        # For each start point, run a local optimizer that adapts step sizes
        # based on local curvature (Quasi-Newton lite).
        
        def adaptive_local_optimize(x_start, f_start, max_evals):
            """
            Custom local optimizer using adaptive coordinate descent.
            It estimates local curvature per dimension to scale steps,
            effectively handling the ill-conditioned Bent Cigar landscape.
            """
            if not time_check():
                return x_start, f_start
            
            x = x_start.copy()
            f = f_start
            
            # Initial step sizes: moderate
            step_sizes = np.ones(d) * 0.5
            # History for curvature estimation
            last_x = x.copy()
            last_f = f
            # Curvature estimates (approx second derivative)
            # We'll update these based on change in slope
            curvatures = np.ones(d)
            
            for _ in range(max_evals):
                if not time_check() or self.calls >= N:
                    break
                
                improved = False
                for i in range(d):
                    if not time_check() or self.calls >= N:
                        break
                    
                    # Adaptive step size based on curvature
                    # If curvature is high (steep), step is small.
                    # If curvature is low (flat), step is larger.
                    # Bent Cigar has high curvature in some dims, low in others.
                    # We want to take large steps in low curvature dirs, small in high.
                    # Approx: step ~ 1/sqrt(curv)
                    # But we need to be careful. Let's use a simple scaling.
                    
                    # Try a step in positive and negative direction
                    # Use a line search to find the best step
                    best_step_val = 0
                    best_f_val = f
                    best_x_val = x.copy()
                    
                    # Try a few candidate step sizes
                    # Base step size scaled by inverse curvature estimate
                    base_step = step_sizes[i]
                    
                    # Candidates: base, base/2, base*2
                    candidates = [base_step, base_step * 0.5, base_step * 2.0]
                    
                    for sign in [1, -1]:
                        for s in candidates:
                            x_trial = x.copy()
                            x_trial[i] += sign * s
                            x_trial = np.clip(x_trial, -5.0, 5.0)
                            
                            f_trial = safe_eval(x_trial)
                            
                            if f_trial < best_f_val:
                                best_f_val = f_trial
                                best_step_val = sign * s
                                best_x_val = x_trial.copy()
                    
                    if best_f_val < f:
                        # Update curvature estimate
                        # Change in slope approx: (f_new - f_old) / step
                        # Change in step approx: step - prev_step (if we had it)
                        # Simple heuristic: if we improved, curvature might be lower than thought?
                        # Or just adjust step size based on success.
                        if abs(best_step_val) > 0:
                            # If we took a large step and improved, maybe we can go larger?
                            # If we took a small step and improved, maybe we can go larger?
                            # Let's adjust step size to be the successful step size
                            step_sizes[i] = abs(best_step_val)
                            
                        x = best_x_val
                        f = best_f_val
                        update_best(x, f)
                        improved = True
                    else:
                        # No improvement, shrink step size
                        step_sizes[i] *= 0.5

                if not improved:
                    # Stagnation in this local run
                    # Try to escape by randomizing a bit? Or just break.
                    # For Bent Cigar, if we are in the valley, we should be improving.
                    # If not, we might be stuck in a flat region or at the minimum.
                    break

            return x, f

        # Run the adaptive local optimizer from each start point
        # Allocate budget dynamically
        # Reserve some budget for a final polish
        reserve_budget = 50
        remaining = N - self.calls - reserve_budget
        
        if remaining > 0:
            # Distribute remaining budget among start points
            # Give more budget to better start points?
            # For simplicity, give equal budget to first few, then stop if budget runs out.
            evals_per_start = max(10, remaining // len(start_points))
            
            for sp in start_points:
                if self.calls >= N - reserve_budget or not time_check():
                    break
                
                f_sp = safe_eval(sp)
                update_best(sp, f_sp)
                
                if np.isinf(f_sp):
                    continue
                
                # Run local optimization
                x_opt, f_opt = adaptive_local_optimize(sp, f_sp, evals_per_start)
                update_best(x_opt, f_opt)

        # Phase 2: Final Polish with very fine adaptive search
        if self.calls < N and time_check():
            x_current = self.best_x.copy()
            f_current = self.best_f
            
            # Use very small step sizes for final convergence
            step_sizes = np.ones(d) * 0.01
            
            for _ in range(20): # Max 20 sweeps
                if not time_check() or self.calls >= N:
                    break
                
                improved = False
                for i in range(d):
                    if not time_check() or self.calls >= N:
                        break
                    
                    # Try small steps
                    best_f_val = f_current
                    best_x_val = x_current.copy()
                    
                    for sign in [1, -1]:
                        for s in [step_sizes[i], step_sizes[i]*0.1]:
                            x_trial = x_current.copy()
                            x_trial[i] += sign * s
                            x_trial = np.clip(x_trial, -5.0, 5.0)
                            
                            f_trial = safe_eval(x_trial)
                            
                            if f_trial < best_f_val:
                                best_f_val = f_trial
                                best_x_val = x_trial.copy()
                    
                    if best_f_val < f_current:
                        x_current = best_x_val
                        f_current = best_f_val
                        update_best(x_current, f_current)
                        improved = True
                        # Keep step size if it worked
                    else:
                        step_sizes[i] *= 0.5

                if not improved:
                    break

        return self.best_x, self.best_f
