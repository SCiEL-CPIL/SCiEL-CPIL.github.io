"""Rewritten program: Robust Adaptive Nelder-Mead for Rosenbrock valley."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Adaptive Nelder-Mead optimizer tailored for Rosenbrock (f8)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # --- Phase 0: Initialization ---
        # Start at origin, which is a common reference point.
        # Rosenbrock minimum is at 0. Hidden shift is in [-4, 4].
        # Starting at 0 is a good bet. If we are far, NM will move us.
        
        # We will maintain a list of active simplexes or just one robust one.
        # Let's try a multi-start strategy but with a very cheap check.
        
        # Candidate starts: 0, and a few points aligned with the valley direction if we suspect shift.
        # However, NM from 0 is usually sufficient if the step size is large enough initially.
        
        # Let's use a single robust NM run.
        
        # Initial Simplex
        # Base point
        x0 = np.zeros(dim)
        
        # Create simplex vertices
        # Standard NM uses a step size. For Rosenbrock, a step of 1.0 or 2.0 is often needed 
        # to escape the flat regions or find the valley if shifted.
        # Let's use 1.0 as a safe bet, but we'll adapt it.
        
        simplex = [x0.copy()]
        base_step = 1.0
        
        for i in range(dim):
            v = x0.copy()
            v[i] += base_step
            v = np.clip(v, -5, 5)
            simplex.append(v)
        
        # Evaluate initial simplex
        f_simplex = []
        calls = 0
        for v in simplex:
            f_simplex.append(func(v))
            calls += 1
            
        # Track best
        min_idx = np.argmin(f_simplex)
        x_best = simplex[min_idx].copy()
        fx_best = f_simplex[min_idx]
        
        # NM Parameters
        alpha = 1.0   # Reflection
        gamma = 2.0   # Expansion
        rho = 0.5     # Contraction
        sigma = 0.5   # Shrink
        
        # Adaptive parameters
        # We will monitor the improvement. If no improvement for X iterations, 
        # we might restart from a different point or increase step size.
        
        no_improve_count = 0
        max_no_improve = 20
        current_step = base_step
        
        # Main Loop
        while calls < budget:
            # Sort simplex by function value
            order = np.argsort(f_simplex)
            simplex = [simplex[i] for i in order]
            f_simplex = [f_simplex[i] for i in order]
            
            # Check for convergence (simplex size is very small)
            # Calculate diameter
            diameter = np.linalg.norm(simplex[0] - simplex[-1])
            if diameter < 1e-6:
                # Converged or stuck. 
                # If we have budget left, try a restart with larger step or random kick.
                if calls < budget - 10:
                    # Try a random kick to escape local minimum
                    kick_dir = rng.normal(0, 1, dim)
                    kick_dir /= np.linalg.norm(kick_dir) + 1e-10
                    kick_size = current_step * 2
                    x_kick = x_best + kick_dir * kick_size
                    x_kick = np.clip(x_kick, -5, 5)
                    f_kick = func(x_kick)
                    calls += 1
                    
                    if f_kick < fx_best:
                        x_best = x_kick
                        fx_best = f_kick
                        # Rebuild simplex around new best
                        simplex = [x_best.copy()]
                        current_step *= 1.5 # Increase step for next simplex
                        for i in range(dim):
                            v = x_best.copy()
                            v[i] += current_step
                            v = np.clip(v, -5, 5)
                            simplex.append(v)
                        f_simplex = []
                        for v in simplex:
                            f_simplex.append(func(v))
                            calls += 1
                        min_idx = np.argmin(f_simplex)
                        if f_simplex[min_idx] < fx_best:
                            fx_best = f_simplex[min_idx]
                            x_best = simplex[min_idx].copy()
                        no_improve_count = 0
                    else:
                        # Shrink step and try again? Or stop if step is tiny.
                        current_step *= 0.5
                        if current_step < 1e-4:
                            break
                        # Rebuild simplex with smaller step
                        simplex = [x_best.copy()]
                        for i in range(dim):
                            v = x_best.copy()
                            v[i] += current_step
                            v = np.clip(v, -5, 5)
                            simplex.append(v)
                        f_simplex = []
                        for v in simplex:
                            f_simplex.append(func(v))
                            calls += 1
                        min_idx = np.argmin(f_simplex)
                        if f_simplex[min_idx] < fx_best:
                            fx_best = f_simplex[min_idx]
                            x_best = simplex[min_idx].copy()
                        no_improve_count = 0
                else:
                    break

            # Standard NM Iteration
            if calls >= budget - 1:
                break
                
            centroid = np.mean(simplex[:-1], axis=0)
            worst = simplex[-1]
            f_worst = f_simplex[-1]
            f_best = f_simplex[0]
            
            # Reflection
            x_ref = centroid + alpha * (centroid - worst)
            x_ref = np.clip(x_ref, -5, 5)
            
            # Check if x_ref is too close to centroid (degenerate)
            if np.allclose(x_ref, centroid, atol=1e-10):
                # Degenerate, shrink
                for i in range(1, dim + 1):
                    if calls >= budget:
                        break
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], -5, 5)
                    f_simplex[i] = func(simplex[i])
                    calls += 1
                continue
            
            f_ref = func(x_ref)
            calls += 1
            
            if f_ref < f_best:
                # Try Expansion
                x_exp = centroid + gamma * (centroid - worst)
                x_exp = np.clip(x_exp, -5, 5)
                if np.allclose(x_exp, centroid, atol=1e-10):
                    simplex[-1] = x_ref
                    f_simplex[-1] = f_ref
                else:
                    f_exp = func(x_exp)
                    calls += 1
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        f_simplex[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        f_simplex[-1] = f_ref
            elif f_ref < f_simplex[-2]:
                # Accept Reflection
                simplex[-1] = x_ref
                f_simplex[-1] = f_ref
            else:
                # Try Contraction
                # Outer contraction (standard)
                x_con = centroid + rho * (worst - centroid)
                x_con = np.clip(x_con, -5, 5)
                if np.allclose(x_con, centroid, atol=1e-10):
                    # Shrink
                    for i in range(1, dim + 1):
                        if calls >= budget:
                            break
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        f_simplex[i] = func(simplex[i])
                        calls += 1
                    continue
                
                f_con = func(x_con)
                calls += 1
                
                if f_con < f_ref:
                    simplex[-1] = x_con
                    f_simplex[-1] = f_con
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        if calls >= budget:
                            break
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        f_simplex[i] = func(simplex[i])
                        calls += 1
            
            # Update Best
            min_idx = np.argmin(f_simplex)
            if f_simplex[min_idx] < fx_best:
                fx_best = f_simplex[min_idx]
                x_best = simplex[min_idx].copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            # If stuck for too long, force a restart or step adjustment
            if no_improve_count >= max_no_improve:
                # Reduce step size and rebuild
                current_step *= 0.5
                if current_step < 1e-5:
                    break
                simplex = [x_best.copy()]
                for i in range(dim):
                    v = x_best.copy()
                    v[i] += current_step
                    v = np.clip(v, -5, 5)
                    simplex.append(v)
                f_simplex = []
                for v in simplex:
                    f_simplex.append(func(v))
                    calls += 1
                min_idx = np.argmin(f_simplex)
                if f_simplex[min_idx] < fx_best:
                    fx_best = f_simplex[min_idx]
                    x_best = simplex[min_idx].copy()
                no_improve_count = 0

        return x_best, fx_best
# EVOLVE-BLOCK-END
