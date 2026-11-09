"""Simulated Annealing with adaptive cooling for BBOB f7 (Step ellipsoidal)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """SA with faster cooling and plateau-escape jumps for step ellipsoidal functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(0)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def time_left():
            return deadline - time.time()

        # Phase 1: Broad uniform scout
        n_initial = min(25, budget // 4)
        for i in range(n_initial):
            if calls_made >= budget or time_left() < 10.0:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: SA with adaptive cooling and plateau escape
        # Faster cooling (0.97) and lower initial temp (5.0) to converge
        # on the sharp drop; occasional 2x jumps to escape flat plateaus.
        
        n_sa_runs = 10
        calls_per_run = max(15, (budget - calls_made) // n_sa_runs)
        
        for run in range(n_sa_runs):
            if calls_made >= budget or time_left() < 15.0:
                break
            
            # Choose starting point
            if run == 0 and best_x is not None:
                x = best_x.copy()
                fx = best_f
            elif run % 2 == 0 and best_x is not None:
                x = best_x + 2.0 * rng.standard_normal(dim)
                x = np.clip(x, -5, 5)
                fx = func(x)
                calls_made += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                x = rng.uniform(-5, 5, dim)
                fx = func(x)
                calls_made += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            
            # SA parameters: faster cooling, lower initial temp
            T_init = 5.0
            T_min = 0.005
            cooling_rate = 0.97
            T = T_init
            
            sigma = 1.5
            accept_count = 0
            total_count = 0
            plateau_steps = 0
            max_plateau = 30
            
            for step in range(calls_per_run):
                if calls_made >= budget or time_left() < 5.0:
                    break
                
                # 10% chance of a 2x step jump to escape plateaus
                step_mult = 2.0 if rng.random() < 0.1 else 1.0
                
                y = x + sigma * step_mult * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                
                fy = func(y)
                calls_made += 1
                total_count += 1
                
                delta_f = fy - fx
                
                if delta_f < 0:
                    x = y
                    fx = fy
                    accept_count += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    plateau_steps = 0
                    sigma *= 1.05
                else:
                    if T > T_min:
                        prob = np.exp(-delta_f / T)
                        if rng.random() < prob:
                            x = y
                            fx = fy
                            accept_count += 1
                            plateau_steps = 0
                        else:
                            plateau_steps += 1
                    else:
                        plateau_steps += 1
                
                T *= cooling_rate
                
                # Adapt step size
                if total_count > 10:
                    accept_rate = accept_count / total_count
                    if accept_rate > 0.6:
                        sigma *= 1.1
                    elif accept_rate < 0.1:
                        sigma *= 0.85
                    if total_count % 20 == 0:
                        accept_count = 0
                        total_count = 0
                
                # Plateau escape: large jump
                if plateau_steps > max_plateau:
                    y = x + 3.0 * rng.standard_normal(dim)
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    calls_made += 1
                    if fy < fx:
                        x = y
                        fx = fy
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                        plateau_steps = 0
                        sigma = max(sigma, 1.5)
                    else:
                        y = x + 5.0 * rng.standard_normal(dim)
                        y = np.clip(y, -5, 5)
                        fy = func(y)
                        calls_made += 1
                        if fy < fx:
                            x = y
                            fx = fy
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()
                        plateau_steps = 0
                        sigma = max(sigma, 2.0)
                
                sigma = np.clip(sigma, 0.05, 8.0)

        # Phase 3: Fine-grained local search
        if best_x is not None and calls_made < budget and time_left() > 10.0:
            x = best_x.copy()
            fx = best_f
            sigma = 0.2
            polish_steps = min(80, budget - calls_made)
            for _ in range(polish_steps):
                if calls_made >= budget or time_left() < 3.0:
                    break
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < fx:
                    x = y
                    fx = fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.05
                else:
                    sigma *= 0.92
                sigma = np.clip(sigma, 0.005, 1.5)

        return best_x, best_f
# EVOLVE-BLOCK-END
