"""Simulated Annealing with adaptive cooling for BBOB f7 (Step ellipsoidal)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Simulated Annealing with adaptive cooling schedule, initialized from a broad uniform scout."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(123)
        dim = self.dim
        budget = self.budget

        # Initialize with a broad uniform scout
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # Adaptive parameters
        T = 10.0
        T_min = 1e-12
        step_size = 2.0
        plateau_threshold = 1e-6  # Treat very small changes as plateau

        # Tracking for adaptation
        stall_counter = 0
        stall_limit = 25
        recent_accepts = 0
        recent_total = 0
        window_len = 20

        # Track best improvement magnitude for scaling
        last_improvement = 0.0
        improvement_momentum = 0.0

        # Phase-based cooling: start with high exploration, end with exploitation
        total_steps = budget - 1

        for i in range(total_steps):
            progress = i / total_steps
            
            # Adaptive temperature based on progress and recent performance
            base_T = T * (1.0 - 0.7 * progress)  # Gradual cooling
            
            # Generate candidate with occasional larger jumps for plateau escape
            if rng.random() < 0.08:
                # Larger exploration step to escape plateaus
                direction = rng.standard_normal(dim)
                norm = np.linalg.norm(direction)
                if norm > 1e-10:
                    direction /= norm
                y = np.clip(x + step_size * 3.0 * direction, -5, 5)
            else:
                direction = rng.standard_normal(dim)
                y = np.clip(x + step_size * direction, -5, 5)
            
            fy = func(y)
            
            accepted = False
            delta_f = fy - fx
            
            if delta_f < -plateau_threshold:
                # Clear improvement
                accepted = True
            elif abs(delta_f) <= plateau_threshold:
                # Plateau move - accept with moderate probability to explore flat regions
                if rng.random() < 0.5:
                    accepted = True
            else:
                # Uphill move - use annealing acceptance
                if delta_f < base_T * 3:
                    prob = np.exp(-delta_f / base_T)
                    if rng.random() < prob:
                        accepted = True
            
            if accepted:
                x = y
                fx = fy
                recent_accepts += 1
                
                if fx < best_fx - plateau_threshold:
                    improvement = best_fx - fx
                    last_improvement = improvement
                    improvement_momentum = 0.8 * improvement_momentum + 0.2 * improvement
                    best_x = x.copy()
                    best_fx = fx
                    stall_counter = 0
                    # Cool down faster on significant improvement
                    T *= 0.96
                elif fx < best_fx:
                    # Tiny improvement, still cool slightly
                    T *= 0.995
                else:
                    # Accepted but not better than best - maintain temperature
                    T *= 0.998
            else:
                stall_counter += 1
                T *= 0.990

            recent_total += 1

            # Adaptive Step Size based on acceptance rate
            if recent_total >= window_len:
                accept_ratio = recent_accepts / recent_total
                if accept_ratio > 0.6:
                    step_size *= 1.15
                elif accept_ratio < 0.3:
                    step_size *= 0.85
                # Reset window
                recent_accepts = 0
                recent_total = 0
                # Clamp step size
                step_size = np.clip(step_size, 0.02, 5.0)

            # Restart mechanism for stagnation - more aggressive in early phase
            if stall_counter > stall_limit:
                # In early phase, do a full restart; in late phase, local perturbation
                if progress < 0.5:
                    x = best_x.copy()
                    fx = best_fx
                    T = max(T, 5.0)
                    step_size = max(step_size, 1.0)
                else:
                    # Local perturbation around best
                    direction = rng.standard_normal(dim)
                    x = np.clip(best_x + 0.5 * step_size * direction, -5, 5)
                    fx = func(x)
                    T = max(T, 2.0)
                    step_size = max(step_size, 0.5)
                
                stall_counter = 0
                last_improvement = 0.0
                improvement_momentum = 0.0

            if T < T_min:
                T = T_min

        return best_x, best_fx
# EVOLVE-BLOCK-END
