"""Combined optimizer for BBOB f12 (Bent cigar): multi-start Hooke-Jeeves with adaptive coordinate steps."""
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285.0  # safety margin before 330s limit

        rng = np.random.default_rng(42)
        call_count = 0
        best_x = None
        best_f = np.inf

        def evaluate(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return np.inf
            x = np.clip(x, -5.0, 5.0)
            f_val = func(x)
            call_count += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # --- Phase 1: Multi-start initialization ---
        # Evaluate origin first (common for BBOB), then random points
        evaluate(np.zeros(dim))
        
        n_init = 15
        for _ in range(n_init):
            if call_count >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            evaluate(x)

        if best_x is None:
            evaluate(np.zeros(dim))
            best_x = np.zeros(dim)

        # --- Phase 2: Hooke-Jeeves pattern search with adaptive steps ---
        # Hooke-Jeeves is particularly effective for anisotropic valleys like bent cigar
        current_x = best_x.copy()
        current_f = best_f
        steps = np.ones(dim) * 1.0
        min_step = 1e-7
        max_step = 5.0
        step_multiplier = 2.0
        step_shrink = 0.5
        shrink_threshold = 3  # number of full cycles without improvement before shrinking

        no_improve_cycles = 0
        restart_cooldown = 0

        while call_count < budget and time.time() < deadline:
            improved = False

            # Try each coordinate in sequence with line search
            for i in range(dim):
                if call_count >= budget or time.time() >= deadline:
                    break

                # Try positive direction
                trial_x = current_x.copy()
                trial_x[i] += steps[i]
                trial_x = np.clip(trial_x, -5, 5)
                f_trial = evaluate(trial_x)

                if f_trial < current_f:
                    current_x = trial_x.copy()
                    current_f = f_trial
                    improved = True
                    no_improve_cycles = 0
                    # Continue in same direction (line search)
                    while call_count < budget and time.time() < deadline:
                        next_x = current_x.copy()
                        next_x[i] += steps[i]
                        next_x = np.clip(next_x, -5, 5)
                        f_next = evaluate(next_x)
                        if f_next < current_f:
                            current_x = next_x.copy()
                            current_f = f_next
                        else:
                            break
                    # Increase step size for this coordinate
                    steps[i] = min(steps[i] * step_multiplier, max_step)
                else:
                    # Try negative direction
                    trial_x = current_x.copy()
                    trial_x[i] -= steps[i]
                    trial_x = np.clip(trial_x, -5, 5)
                    f_trial = evaluate(trial_x)

                    if f_trial < current_f:
                        current_x = trial_x.copy()
                        current_f = f_trial
                        improved = True
                        no_improve_cycles = 0
                        # Continue in same direction (line search)
                        while call_count < budget and time.time() < deadline:
                            next_x = current_x.copy()
                            next_x[i] -= steps[i]
                            next_x = np.clip(next_x, -5, 5)
                            f_next = evaluate(next_x)
                            if f_next < current_f:
                                current_x = next_x.copy()
                                current_f = f_next
                            else:
                                break
                        # Increase step size for this coordinate
                        steps[i] = min(steps[i] * step_multiplier, max_step)

            # If no coordinate improved in this cycle, shrink steps
            if not improved:
                no_improve_cycles += 1
                steps *= step_shrink

                # If steps are too small, try a local perturbation from best point
                if np.all(steps < min_step):
                    # Small random perturbation around best point
                    noise = rng.standard_normal(dim) * 0.1
                    new_x = best_x + noise
                    new_x = np.clip(new_x, -5, 5)
                    f_new = evaluate(new_x)
                    if f_new < current_f:
                        current_x = new_x.copy()
                        current_f = f_new
                    # Reset steps to a moderate value
                    steps = np.ones(dim) * 0.5
                    no_improve_cycles = 0
                elif no_improve_cycles >= shrink_threshold:
                    # If we've shrunk multiple times without improvement, 
                    # try a larger jump from best point (local restart)
                    if restart_cooldown <= 0:
                        jump_scale = np.max(steps) * 2.0
                        jump = rng.standard_normal(dim) * jump_scale
                        new_x = best_x + jump
                        new_x = np.clip(new_x, -5, 5)
                        f_new = evaluate(new_x)
                        if f_new < current_f:
                            current_x = new_x.copy()
                            current_f = f_new
                        # Reset steps
                        steps = np.ones(dim) * 1.0
                        no_improve_cycles = 0
                        restart_cooldown = 5  # wait 5 cycles before another restart
                    else:
                        restart_cooldown -= 1

            else:
                # Reset cooldown on improvement
                if restart_cooldown > 0:
                    restart_cooldown = max(0, restart_cooldown - 1)

            # Periodic global exploration: occasionally jump to a new random point
            # This helps escape local minima in the anisotropic landscape
            if call_count % 80 == 0 and call_count > 0 and call_count < budget - 50:
                if rng.random() < 0.15:
                    # Jump to a new random location
                    jump_x = rng.uniform(-5, 5, dim)
                    f_jump = evaluate(jump_x)
                    if f_jump < current_f:
                        current_x = jump_x.copy()
                        current_f = f_jump
                    # Reset steps for fresh exploration
                    steps = np.ones(dim) * 1.0
                    no_improve_cycles = 0

        return best_x, best_f
