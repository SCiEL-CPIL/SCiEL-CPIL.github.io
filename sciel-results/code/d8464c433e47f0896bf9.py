"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic multi-scale coordinate descent for Büche-Rastrigin f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin

        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = None
        best_fx = float('inf')

        def record(x, fx):
            nonlocal best_x, best_fx, calls
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        def evaluate(x):
            nonlocal calls
            if calls >= budget or time.time() > deadline:
                return None
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            fx = func(x)
            calls += 1
            record(x, fx)
            return fx

        # Phase 0: Initial diverse probes
        # Büche-Rastrigin has global min at x=2 (unshifted) and local mins at integers.
        # We probe a grid of likely candidates to find a good starting basin.
        init_candidates = []
        # The known unshifted minimum
        init_candidates.append(np.full(dim, 2.0))
        # Origin
        init_candidates.append(np.zeros(dim))
        # Some structured points around the expected range [-4, 4]
        # We'll add a few random-ish deterministic points
        rng_init = np.random.default_rng(42)
        for _ in range(10):
            init_candidates.append(rng_init.uniform(-4, 4, size=dim))
        
        # Evaluate initial candidates
        for p in init_candidates:
            fx = evaluate(p)
            if fx is None:
                break

        if best_x is None:
            # Fallback if all failed (shouldn't happen)
            best_x = np.zeros(dim)
            best_fx = func(best_x)
            calls = 1

        # Phase 1: Multi-scale Coordinate Descent
        # Strategy: 
        # 1. Start with a large step size.
        # 2. For each dimension, try moving +step and -step.
        # 3. If improvement, keep it.
        # 4. If no improvement in a full pass, reduce step size by a factor (e.g., 0.5).
        # 5. Repeat until step size is very small or budget/time runs out.
        
        current_x = best_x.copy()
        current_fx = best_fx
        
        # Initial step size: large enough to jump between local minima
        # Local minima are roughly 1 unit apart. So start with 2.0 or 1.0.
        step_size = 2.0
        min_step = 1e-6
        shrink_factor = 0.5
        
        # To ensure progress early, we might want to do a few full passes with large steps.
        # However, simple coordinate descent can get stuck in local minima if steps are too small.
        # Büche-Rastrigin is multimodal. 
        # A "jump" strategy might be better: try larger steps if stuck.
        # Let's stick to a robust shrinking scheme but with a "restart" if stuck too long.
        
        stuck_count = 0
        last_improvement_fx = current_fx
        pass_count = 0
        
        while calls < budget and time.time() < deadline:
            improved = False
            for d in range(dim):
                if time.time() > deadline or calls >= budget:
                    break
                
                # Try positive step
                x_plus = current_x.copy()
                x_plus[d] += step_size
                fx_plus = evaluate(x_plus)
                if fx_plus is None:
                    break
                
                if fx_plus < current_fx:
                    current_x = x_plus
                    current_fx = fx_plus
                    improved = True
                    stuck_count = 0
                    continue # Try to improve further in this dimension? 
                    # Standard CD usually takes the best of +/- or just the first improvement.
                    # Let's check negative too if we didn't improve? 
                    # Actually, if we improved, we might want to check if we overshot.
                    # But simple CD: take the move.
                
                # Try negative step
                x_minus = current_x.copy()
                x_minus[d] -= step_size
                fx_minus = evaluate(x_minus)
                if fx_minus is None:
                    break
                
                if fx_minus < current_fx:
                    current_x = x_minus
                    current_fx = fx_minus
                    improved = True
                    stuck_count = 0
            
            pass_count += 1
            
            if improved:
                # Reset stuck counter if we improved
                # But if we improved slightly, we might still be in a local min.
                # Let's shrink step size to refine.
                step_size *= shrink_factor
                if step_size < min_step:
                    step_size = min_step
            else:
                # No improvement in this pass
                stuck_count += 1
                if stuck_count > 5:
                    # Stuck. Try a larger step? Or restart from best?
                    # Büche-Rastrigin has many local minima. 
                    # If stuck, maybe we are in a local min. 
                    # Let's try a random jump to escape?
                    # Or just shrink further to ensure we are at the bottom of the current valley.
                    # If we shrink too much without improvement, we are at a local minimum.
                    # We should try to escape.
                    if step_size > 0.1:
                        # Try a larger step to escape local min
                        step_size = max(step_size * 2.0, 1.0)
                        stuck_count = 0
                    else:
                        # We are stuck at a small scale. 
                        # Let's try a random perturbation to escape.
                        rng_escape = np.random.default_rng(int(time.time() * 1000) % 10000)
                        x_escape = current_x + rng_escape.standard_normal(dim) * 1.0
                        fx_escape = evaluate(x_escape)
                        if fx_escape is not None and fx_escape < current_fx:
                            current_x = x_escape
                            current_fx = fx_escape
                            step_size = 1.0
                            stuck_count = 0
                        else:
                            # Still stuck. Shrink step size to converge.
                            step_size *= shrink_factor
                            if step_size < min_step:
                                step_size = min_step
                            # If we've been stuck for a long time at small steps, 
                            # maybe we should restart from the global best found so far?
                            # But current_x is already the global best if we track it.
                            # Let's force a restart from best_x if we are far from it?
                            # Actually, current_x IS the best we found in this trajectory.
                            # If we are stuck, we might be in a local min that is not global.
                            # Let's try a "random restart" logic:
                            # If stuck_count > 20, pick a random point near best_x with larger variance.
                            if stuck_count > 20:
                                rng_restart = np.random.default_rng(12345 + calls)
                                x_restart = best_x + rng_restart.standard_normal(dim) * 2.0
                                fx_restart = evaluate(x_restart)
                                if fx_restart is not None:
                                    if fx_restart < current_fx:
                                        current_x = x_restart
                                        current_fx = fx_restart
                                        step_size = 1.0
                                        stuck_count = 0
                                    else:
                                        # Keep current, but reset stuck counter to avoid infinite loop
                                        stuck_count = 0
                                        step_size = min_step # Force convergence on current point
                                else:
                                    break
                else:
                    # Just stuck a few passes. Shrink step size.
                    step_size *= shrink_factor
                    if step_size < min_step:
                        step_size = min_step

            # Update global best
            if current_fx < best_fx:
                best_fx = current_fx
                best_x = current_x.copy()

        # Phase 2: Final fine-tuning
        # Use very small steps to polish the result
        if calls < budget and time.time() < deadline:
            current_x = best_x.copy()
            current_fx = best_fx
            step_size = 0.01
            for _ in range(10): # A few passes of fine-tuning
                for d in range(dim):
                    if time.time() > deadline or calls >= budget:
                        break
                    x_plus = current_x.copy()
                    x_plus[d] += step_size
                    fx_plus = evaluate(x_plus)
                    if fx_plus is None: break
                    if fx_plus < current_fx:
                        current_x = x_plus
                        current_fx = fx_plus
                        continue
                    x_minus = current_x.copy()
                    x_minus[d] -= step_size
                    fx_minus = evaluate(x_minus)
                    if fx_minus is None: break
                    if fx_minus < current_fx:
                        current_x = x_minus
                        current_fx = fx_minus
                if current_fx < best_fx:
                    best_fx = current_fx
                    best_x = current_x.copy()
                step_size *= 0.5

        return best_x, best_fx
# EVOLVE-BLOCK-END
