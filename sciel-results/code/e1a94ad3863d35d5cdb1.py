"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

class Optimizer:
    """
    Optimizer using Multistart Nelder-Mead.
    """
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        # Parameters for Nelder-Mead
        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5    # contraction
        sigma = 0.5  # shrink
        
        def evaluate_simplex(simplex, fvals):
            """Evaluate a simplex, tracking calls and best."""
            nonlocal calls_made, best_f, best_x
            for i in range(len(simplex)):
                if calls_made >= budget:
                    break
                f = func(simplex[i])
                fvals[i] = f
                calls_made += 1
                if f < best_f:
                    best_f = f
                    best_x = simplex[i].copy()

        def create_simplex(center, step_size):
            """Create a simplex around center with given step size."""
            simplex = [center.copy() for _ in range(dim + 1)]
            for i in range(1, dim + 1):
                simplex[i][i-1] += step_size
            return simplex

        # Strategy: Use multiple phases
        # Phase 1: Coarse search with larger steps
        # Phase 2: Fine search around best points found
        
        # First, do a quick random sampling to get a rough idea of the landscape
        sample_points = []
        sample_fvals = []
        n_samples = min(20, budget // 10)
        
        for _ in range(n_samples):
            if calls_made >= budget:
                break
            x = np.random.uniform(-4, 4, dim)
            f = func(x)
            calls_made += 1
            sample_points.append(x)
            sample_fvals.append(f)
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        # Sort samples by function value
        if len(sample_fvals) > 0:
            sorted_idx = np.argsort(sample_fvals)
            sample_points = [sample_points[i] for i in sorted_idx]
            sample_fvals = [sample_fvals[i] for i in sorted_idx]
        
        # Now run Nelder-Mead from multiple starting points
        # Start from the best few sample points and some random points
        start_points = []
        
        # Add top 3 sample points
        for i in range(min(3, len(sample_points))):
            start_points.append(sample_points[i].copy())
        
        # Add a few more random points
        n_extra_starts = min(5, (budget - calls_made) // 10)
        for _ in range(n_extra_starts):
            start_points.append(np.random.uniform(-4, 4, dim))
        
        # For each start point, run Nelder-Mead with adaptive step size
        for start_idx, center in enumerate(start_points):
            if calls_made >= budget:
                break
            
            # Determine initial step size based on how early we are
            # Early: larger steps for exploration, later: smaller for exploitation
            progress = calls_made / budget
            if start_idx < 3:
                # Starting from good samples: use moderate steps
                initial_step = 1.5 * (1.0 - 0.5 * progress)
            else:
                # Starting from random: use larger steps initially
                initial_step = 2.0 * (1.0 - 0.3 * progress)
            
            # Cap step size
            initial_step = max(0.3, min(initial_step, 2.5))
            
            # Create and evaluate initial simplex
            simplex = create_simplex(center, initial_step)
            fvals = [np.inf] * (dim + 1)
            evaluate_simplex(simplex, fvals)
            
            if calls_made >= budget:
                break
            
            # Sort simplex by function value
            sorted_indices = np.argsort(fvals)
            simplex = [simplex[i] for i in sorted_indices]
            fvals = [fvals[i] for i in sorted_indices]
            
            # Local best for this run
            local_best_f = fvals[0]
            local_best_x = simplex[0].copy()
            no_improvement_count = 0
            run_evals = 0
            max_evals_per_run = max(30, min(60, (budget - calls_made) // 5))
            
            # Nelder-Mead iterations
            while calls_made < budget and run_evals < max_evals_per_run:
                # Check for convergence
                if len(fvals) > 1 and (fvals[-1] - fvals[0]) < 1e-12 * max(1.0, abs(fvals[0])):
                    break
                
                # Identify worst point
                worst_idx = dim  # Last element is worst after sorting
                
                # Calculate centroid of all but worst
                centroid = np.mean(simplex[:dim], axis=0)
                
                # Reflection
                x_ref = centroid + alpha * (centroid - simplex[worst_idx])
                x_ref = np.clip(x_ref, -5, 5)
                
                if calls_made >= budget:
                    break
                
                f_ref = func(x_ref)
                calls_made += 1
                run_evals += 1
                
                if f_ref < best_f:
                    best_f = f_ref
                    best_x = x_ref.copy()
                
                if f_ref < fvals[0]:
                    # Expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp = np.clip(x_exp, -5, 5)
                    
                    if calls_made >= budget:
                        break
                        
                    f_exp = func(x_exp)
                    calls_made += 1
                    run_evals += 1
                    
                    if f_exp < best_f:
                        best_f = f_exp
                        best_x = x_exp.copy()
                        
                    if f_exp < f_ref:
                        simplex[worst_idx] = x_exp
                        fvals[worst_idx] = f_exp
                    else:
                        simplex[worst_idx] = x_ref
                        fvals[worst_idx] = f_ref
                        
                elif f_ref < fvals[dim - 1]:
                    # Replace worst with reflected point
                    simplex[worst_idx] = x_ref
                    fvals[worst_idx] = f_ref
                else:
                    # Contraction
                    x_con = centroid + rho * (x_ref - centroid)
                    x_con = np.clip(x_con, -5, 5)
                    
                    if calls_made >= budget:
                        break
                        
                    f_con = func(x_con)
                    calls_made += 1
                    run_evals += 1
                    
                    if f_con < best_f:
                        best_f = f_con
                        best_x = x_con.copy()
                        
                    if f_con < f_ref:
                        simplex[worst_idx] = x_con
                        fvals[worst_idx] = f_con
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            
                            if calls_made >= budget:
                                break
                                
                            f_shr = func(simplex[i])
                            calls_made += 1
                            run_evals += 1
                            
                            if f_shr < best_f:
                                best_f = f_shr
                                best_x = simplex[i].copy()
                                
                            fvals[i] = f_shr
                        
                        if calls_made >= budget:
                            break

                # Re-sort simplex
                sorted_indices = np.argsort(fvals)
                simplex = [simplex[i] for i in sorted_indices]
                fvals = [fvals[i] for i in sorted_indices]
                
                # Check if we made progress
                if fvals[0] < local_best_f - 1e-10 * max(1.0, abs(local_best_f)):
                    local_best_f = fvals[0]
                    local_best_x = simplex[0].copy()
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                    
                # If stuck for too many steps, break to restart
                if no_improvement_count > 15:
                    break
            
            # After each NM run, if we have budget left, do a local refinement
            # around the best point found so far with a smaller step
            if calls_made < budget and best_x is not None:
                remaining_budget = budget - calls_made
                if remaining_budget >= 10:
                    # Do a fine NM around best_x
                    fine_center = best_x.copy()
                    fine_step = max(0.1, initial_step * 0.2)
                    
                    fine_simplex = create_simplex(fine_center, fine_step)
                    fine_fvals = [np.inf] * (dim + 1)
                    evaluate_simplex(fine_simplex, fine_fvals)
                    
                    if calls_made < budget:
                        sorted_indices = np.argsort(fine_fvals)
                        fine_simplex = [fine_simplex[i] for i in sorted_indices]
                        fine_fvals = [fine_fvals[i] for i in sorted_indices]
                        
                        fine_local_best = fine_fvals[0]
                        fine_no_improve = 0
                        fine_run_evals = 0
                        fine_max_evals = min(30, remaining_budget - 5)
                        
                        while calls_made < budget and fine_run_evals < fine_max_evals:
                            if len(fine_fvals) > 1 and (fine_fvals[-1] - fine_fvals[0]) < 1e-12 * max(1.0, abs(fine_fvals[0])):
                                break
                            
                            worst_idx = dim
                            centroid = np.mean(fine_simplex[:dim], axis=0)
                            
                            x_ref = centroid + alpha * (centroid - fine_simplex[worst_idx])
                            x_ref = np.clip(x_ref, -5, 5)
                            
                            if calls_made >= budget:
                                break
                            
                            f_ref = func(x_ref)
                            calls_made += 1
                            fine_run_evals += 1
                            
                            if f_ref < best_f:
                                best_f = f_ref
                                best_x = x_ref.copy()
                            
                            if f_ref < fine_fvals[0]:
                                x_exp = centroid + gamma * (x_ref - centroid)
                                x_exp = np.clip(x_exp, -5, 5)
                                
                                if calls_made >= budget:
                                    break
                                    
                                f_exp = func(x_exp)
                                calls_made += 1
                                fine_run_evals += 1
                                
                                if f_exp < best_f:
                                    best_f = f_exp
                                    best_x = x_exp.copy()
                                    
                                if f_exp < f_ref:
                                    fine_simplex[worst_idx] = x_exp
                                    fine_fvals[worst_idx] = f_exp
                                else:
                                    fine_simplex[worst_idx] = x_ref
                                    fine_fvals[worst_idx] = f_ref
                                    
                            elif f_ref < fine_fvals[dim - 1]:
                                fine_simplex[worst_idx] = x_ref
                                fine_fvals[worst_idx] = f_ref
                            else:
                                x_con = centroid + rho * (x_ref - centroid)
                                x_con = np.clip(x_con, -5, 5)
                                
                                if calls_made >= budget:
                                    break
                                    
                                f_con = func(x_con)
                                calls_made += 1
                                fine_run_evals += 1
                                
                                if f_con < best_f:
                                    best_f = f_con
                                    best_x = x_con.copy()
                                    
                                if f_con < f_ref:
                                    fine_simplex[worst_idx] = x_con
                                    fine_fvals[worst_idx] = f_con
                                else:
                                    for i in range(1, dim + 1):
                                        fine_simplex[i] = fine_simplex[0] + sigma * (fine_simplex[i] - fine_simplex[0])
                                        fine_simplex[i] = np.clip(fine_simplex[i], -5, 5)
                                        
                                        if calls_made >= budget:
                                            break
                                            
                                        f_shr = func(fine_simplex[i])
                                        calls_made += 1
                                        fine_run_evals += 1
                                        
                                        if f_shr < best_f:
                                            best_f = f_shr
                                            best_x = fine_simplex[i].copy()
                                        
                                        fine_fvals[i] = f_shr
                                    
                                    if calls_made >= budget:
                                        break

                            sorted_indices = np.argsort(fine_fvals)
                            fine_simplex = [fine_simplex[i] for i in sorted_indices]
                            fine_fvals = [fine_fvals[i] for i in sorted_indices]
                            
                            if fine_fvals[0] < fine_local_best - 1e-10 * max(1.0, abs(fine_local_best)):
                                fine_local_best = fine_fvals[0]
                                fine_no_improve = 0
                            else:
                                fine_no_improve += 1
                            
                            if fine_no_improve > 10:
                                break

        return best_x, best_f
# EVOLVE-BLOCK-END
