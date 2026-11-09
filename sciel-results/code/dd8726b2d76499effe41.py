"""Coarse-to-fine 1D grid search exploiting Rastrigin separability, then Nelder-Mead polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Coarse-to-fine 1D coordinate search + Nelder-Mead polish for Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 330s kill

        # Budget tracking
        calls_made = 0

        def call_func(x):
            nonlocal calls_made
            if calls_made >= self.budget:
                raise ValueError("Budget exceeded")
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            val = float(func(x))
            calls_made += 1
            return val

        dim = self.dim

        # Phase 1: Coarse-to-fine 1D grid search along each axis
        # Rastrigin: f(x) = 10*n + sum(x_i^2 - 10*cos(2*pi*x_i))
        # Global minimum at each x_i = 0 (in local coords), so we search for the
        # basin by finding the best 1D slice for each coordinate.

        # Initialize with center
        x = np.zeros(dim)
        fx = call_func(x)

        # Coarse 1D scan for each axis
        # 21 points per axis to better capture Rastrigin's period-1 oscillations
        coarse_points = np.linspace(-5, 5, 21)

        for axis in range(dim):
            best_val = fx
            best_pt = x[axis]
            for pt in coarse_points:
                x[axis] = pt
                val = call_func(x)
                if val < best_val:
                    best_val = val
                    best_pt = pt
            x[axis] = best_pt
            fx = best_val

        # Medium grid: 25 points in a window of width 1.5 around current best for each axis
        for axis in range(dim):
            center = x[axis]
            window = np.linspace(max(-5, center - 1.5), min(5, center + 1.5), 25)
            best_val = fx
            best_pt = x[axis]
            for pt in window:
                x[axis] = pt
                val = call_func(x)
                if val < best_val:
                    best_val = val
                    best_pt = pt
            x[axis] = best_pt
            fx = best_val

        # Check time
        if time.time() > deadline:
            return x, fx

        # Fine grid: 33 points in a window of width 0.6
        for axis in range(dim):
            center = x[axis]
            window = np.linspace(max(-5, center - 0.3), min(5, center + 0.3), 33)
            best_val = fx
            best_pt = x[axis]
            for pt in window:
                x[axis] = pt
                val = call_func(x)
                if val < best_val:
                    best_val = val
                    best_pt = pt
            x[axis] = best_pt
            fx = best_val

        # Ultra-fine grid: 41 points in a window of width 0.1
        for axis in range(dim):
            center = x[axis]
            window = np.linspace(max(-5, center - 0.05), min(5, center + 0.05), 41)
            best_val = fx
            best_pt = x[axis]
            for pt in window:
                x[axis] = pt
                val = call_func(x)
                if val < best_val:
                    best_val = val
                    best_pt = pt
            x[axis] = best_pt
            fx = best_val

        if time.time() > deadline:
            return x, fx

        # Phase 2: Nelder-Mead polish
        # Use scipy's minimize with Nelder-Mead
        from scipy.optimize import minimize

        best_x = x.copy()
        best_f = fx

        # First Nelder-Mead pass with moderate tolerance
        try:
            result = minimize(
                call_func,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 150, 'xatol': 1e-5, 'fatol': 1e-7, 'adaptive': True}
            )
            if result.fun < best_f:
                best_x = result.x
                best_f = result.fun
        except Exception:
            pass

        # Check time
        if time.time() > deadline:
            return best_x, best_f

        # Second Nelder-Mead pass with tighter tolerance
        if calls_made < self.budget - 15:
            try:
                result = minimize(
                    call_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True}
                )
                if result.fun < best_f:
                    best_x = result.x
                    best_f = result.fun
            except Exception:
                pass

        # Check time
        if time.time() > deadline:
            return best_x, best_f

        # Third Nelder-Mead pass with very tight tolerance for final polish
        if calls_made < self.budget - 5:
            try:
                result = minimize(
                    call_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 50, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
                )
                if result.fun < best_f:
                    best_x = result.x
                    best_f = result.fun
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
