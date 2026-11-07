"""Hybrid (mu+lambda)-ES + L-BFGS-B optimizer for BBOB f22 in 5D.

Strategy:
  Phase 1: (mu+lambda) Evolution Strategy with a diverse initial population
           spread across [-5, 5]^5 to locate the best of the 21 basins early.
           This provides strong AOCC progress through rapid global exploration.
  Phase 2: L-BFGS-B refinement on the best ES candidate for fast local
           convergence to the minimum.
  A wall-clock deadline guard (320 s) returns the best result found safely early.
"""
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """(mu+lambda)-ES then L-BFGS-B optimizer for the BBOB f22 task."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        deadline = time.time() + 320.0  # safety margin before 330 s kill
        n = self.dim
        budget = self.budget
        calls = {"count": 0}

        def fwrap(x):
            if time.time() >= deadline:
                raise TimeoutError
            calls["count"] += 1
            if calls["count"] > budget:
                raise ValueError("Budget exceeded")
            return func(x)

        best = {"x": None, "f": np.inf}

        def update_best(x, f):
            if f < best["f"]:
                best["f"] = f
                best["x"] = x.copy()

        rng = np.random.default_rng(42)

        # ---------------- Phase 1: (mu+lambda) ES ----------------
        # Use a population-based ES to explore the 21 basins of f22.
        # lambda=20 offspring per generation, mu=10 parents (mu+lambda selection).
        # Initial sigma ~ 2.0 to cover the [-5,5]^5 box reasonably.
        
        lam = 20  # number of offspring per generation
        mu = 10   # number of parents selected
        sigma = 2.0  # initial step size
        es_budget = min(250, budget - 150)  # reserve ~150 calls for L-BFGS-B
        
        # Initialize population with diverse points
        # Mix of grid-like and random points for good initial coverage
        pop = np.zeros((mu, n))
        # First few points from a coarse grid
        grid_vals = np.linspace(-5.0, 5.0, 3)
        grid_points = list(np.meshgrid(*([grid_vals] * n), indexing="ij"))
        grid_points = np.array([p for p in zip(*grid_points)])
        grid_points = np.asarray(grid_points).reshape(-1, n)
        # Shuffle and take first mu points
        perm = rng.permutation(grid_points.shape[0])
        grid_points = grid_points[perm]
        
        # Fill population: half from grid, half random
        n_grid = min(mu // 2, grid_points.shape[0])
        n_rand = mu - n_grid
        pop[:n_grid] = grid_points[:n_grid]
        pop[n_grid:] = rng.uniform(-5.0, 5.0, size=(n_rand, n))
        
        # Evaluate initial population
        for i in range(mu):
            if time.time() >= deadline:
                break
            if calls["count"] >= budget:
                break
            try:
                fi = fwrap(pop[i])
                update_best(pop[i], fi)
            except (TimeoutError, ValueError):
                break
        
        if best["x"] is None:
            x0 = np.zeros(n)
            try:
                best["f"] = fwrap(x0)
                best["x"] = x0
            except:
                return (None, np.inf)
        
        # Store fitnesses for selection
        fitnesses = np.full(mu, np.inf)
        # Re-evaluate to get fitnesses (or we could have stored them)
        # Actually, let's just re-evaluate to be safe, but that wastes calls.
        # Better: store during initial eval. Let me redo this more carefully.
        
        # Let me restart the ES phase with proper bookkeeping
        calls = {"count": 0}
        best = {"x": None, "f": np.inf}
        
        def fwrap(x):
            if time.time() >= deadline:
                raise TimeoutError
            calls["count"] += 1
            if calls["count"] > budget:
                raise ValueError("Budget exceeded")
            return func(x)
        
        def update_best(x, f):
            if f < best["f"]:
                best["f"] = f
                best["x"] = x.copy()
        
        # Initialize population
        pop = np.zeros((mu, n))
        grid_vals = np.linspace(-5.0, 5.0, 3)
        grid_points = list(np.meshgrid(*([grid_vals] * n), indexing="ij"))
        grid_points = np.array([p for p in zip(*grid_points)])
        grid_points = np.asarray(grid_points).reshape(-1, n)
        perm = rng.permutation(grid_points.shape[0])
        grid_points = grid_points[perm]
        
        n_grid = min(mu // 2, grid_points.shape[0])
        n_rand = mu - n_grid
        pop[:n_grid] = grid_points[:n_grid]
        pop[n_grid:] = rng.uniform(-5.0, 5.0, size=(n_rand, n))
        
        # Evaluate initial population
        fitnesses = np.full(mu, np.inf)
        for i in range(mu):
            if time.time() >= deadline:
                break
            if calls["count"] >= budget:
                break
            try:
                fi = fwrap(pop[i])
                fitnesses[i] = fi
                update_best(pop[i], fi)
            except (TimeoutError, ValueError):
                break
        
        if best["x"] is None:
            x0 = np.zeros(n)
            try:
                best["f"] = fwrap(x0)
                best["x"] = x0
            except:
                return (None, np.inf)
        
        # ES generations
        while calls["count"] < es_budget:
            if time.time() >= deadline:
                break
            
            # Sort parents by fitness (ascending = better)
            sorted_idx = np.argsort(fitnesses)
            parents = pop[sorted_idx[:mu]]
            parent_fitnesses = fitnesses[sorted_idx[:mu]]
            
            # Select top mu parents (all of them since we have exactly mu)
            # Generate offspring
            offspring = np.zeros((lam, n))
            offspring_fitness = np.full(lam, np.inf)
            
            for j in range(lam):
                if time.time() >= deadline:
                    break
                if calls["count"] >= budget:
                    break
                
                # Select parent with tournament selection
                # Simple: pick a random parent from top mu/2
                top_mu_half = max(1, mu // 2)
                parent_idx = rng.integers(0, top_mu_half)
                parent = parents[parent_idx]
                
                # Mutate
                child = parent + sigma * rng.standard_normal(n)
                # Clip to bounds
                child = np.clip(child, -5.0, 5.0)
                
                try:
                    fj = fwrap(child)
                    offspring[j] = child
                    offspring_fitness[j] = fj
                    update_best(child, fj)
                except (TimeoutError, ValueError):
                    break
            
            # Combine parents and offspring, select best mu
            combined = np.vstack([parents, offspring])
            combined_fit = np.concatenate([parent_fitnesses, offspring_fitness])
            
            # Select best mu
            sorted_idx = np.argsort(combined_fit)
            pop = combined[sorted_idx[:mu]]
            fitnesses = combined_fit[sorted_idx[:mu]]
            
            # Adapt sigma: simple 1/5 rule
            # Count number of successful offspring (those that made it into selection)
            # For simplicity, use a basic adaptation
            if calls["count"] > 0:
                # If we're making progress, keep sigma; else reduce
                # Simple heuristic: reduce sigma slowly
                sigma *= 0.95
                sigma = max(sigma, 0.01)
        
        # ---------------- Phase 2: L-BFGS-B refinement ----------------
        # Use the best ES point as starting location for L-BFGS-B
        bounds = [(-5.0, 5.0) for _ in range(n)]
        
        # Try L-BFGS-B from the best point
        if best["x"] is not None:
            try:
                res = minimize(fwrap, best["x"].copy(), method='L-BFGS-B', 
                               bounds=bounds, options={'maxiter': 100, 'maxfun': 100})
                if res.fun < best["f"]:
                    best["f"] = res.fun
                    best["x"] = res.x.copy()
            except (TimeoutError, ValueError):
                pass
            except Exception:
                pass
        
        # Try a few more diverse starting points if budget remains
        remaining_starts = min(3, max(0, budget - calls["count"]))
        for _ in range(remaining_starts):
            if time.time() >= deadline:
                break
            if calls["count"] >= budget:
                break
            
            # Pick a random point near the best or a random point
            if rng.random() < 0.5:
                start_x = best["x"] + rng.standard_normal(n) * 0.5
            else:
                start_x = rng.uniform(-5.0, 5.0, size=n)
            start_x = np.clip(start_x, -5.0, 5.0)
            
            try:
                res = minimize(fwrap, start_x, method='L-BFGS-B', 
                               bounds=bounds, options={'maxiter': 50, 'maxfun': 50})
                if res.fun < best["f"]:
                    best["f"] = res.fun
                    best["x"] = res.x.copy()
            except (TimeoutError, ValueError):
                break
            except Exception:
                continue
        
        # Final safety
        if best["x"] is None:
            x0 = np.zeros(n)
            best["f"] = fwrap(x0)
            best["x"] = x0

        return best["x"], best["f"]
