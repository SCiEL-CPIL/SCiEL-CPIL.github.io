"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Cross-Entropy Method (CEM) for black-box optimization on [-5,5]^5."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Initial distribution parameters
        mean = np.zeros(dim)
        sigma = np.full(dim, 4.0)

        # CEM parameters
        # We want to use the budget efficiently. Let's use a population size that
        # allows for many iterations.
        # Total budget = 500.
        # Let pop_size = 20. Iterations = 25.
        # Or pop_size = 10, Iterations = 50.
        # Smaller pop_size allows faster convergence of the distribution but noisier estimates.
        # Larger pop_size is more robust.
        # Let's try pop_size = 10, which gives 50 iterations.
        # Actually, for CEM, a larger population is usually better for finding the global min early,
        # but we have a strict budget.
        # Let's use a dynamic approach or a fixed reasonable one.
        # pop_size = 10 seems reasonable for dim=5.
        pop_size = 10
        
        # Elitism ratio: keep the top 50% or top 30%?
        # Standard CEM uses top 50%. For noisy or multimodal, maybe top 30-40%.
        # Let's use top 50% (elites = pop_size // 2).
        elites = max(1, pop_size // 2)

        # Track best found
        best_x = None
        best_f = np.inf

        # We need to be careful with the budget.
        # Each iteration uses pop_size evaluations.
        # Total evaluations = num_iters * pop_size.
        # We must not exceed budget.
        # Let's calculate max iterations.
        max_iters = budget // pop_size
        # If there's a remainder, we can do a final partial population or just stop.
        # Since CEM relies on a full population for the update, we stick to full pops.
        # Remaining budget can be used for a local search or just ignored if we are close to deadline.

        for i in range(max_iters):
            # Check time
            if time.time() > deadline:
                break

            # Generate population
            # Sample from Normal(mean, sigma^2)
            # Note: sigma is the standard deviation.
            samples = rng.normal(mean, sigma, size=(pop_size, dim))
            
            # Clip to box
            samples = np.clip(samples, -5, 5)

            # Evaluate
            fvals = np.array([func(x) for x in samples])

            # Update best
            min_idx = np.argmin(fvals)
            if fvals[min_idx] < best_f:
                best_f = fvals[min_idx]
                best_x = samples[min_idx].copy()

            # Update distribution parameters
            # Sort by fitness
            sorted_idx = np.argsort(fvals)
            elite_idx = sorted_idx[:elites]
            elite_samples = samples[elite_idx]
            
            # New mean and std dev
            # Weighted average? Standard CEM uses simple mean of elites.
            # But if the landscape is noisy, weighted might be better.
            # Let's stick to simple mean for now, maybe with a small shrinkage towards 0?
            # No, just mean of elites.
            new_mean = np.mean(elite_samples, axis=0)
            new_sigma = np.std(elite_samples, axis=0)
            
            # Prevent sigma from becoming 0
            new_sigma = np.maximum(new_sigma, 0.1)
            
            # Smooth the update to avoid erratic jumps?
            # mean = 0.5 * mean + 0.5 * new_mean
            # sigma = 0.5 * sigma + 0.5 * new_sigma
            # This is a common trick. Let's try a learning rate.
            # Actually, standard CEM replaces. Let's try replacement first.
            # If it oscillates, we can smooth.
            # Given the small budget, replacement might be too fast.
            # Let's use a blend.
            alpha = 0.5
            mean = (1 - alpha) * mean + alpha * new_mean
            sigma = (1 - alpha) * sigma + alpha * new_sigma

            # Ensure sigma stays positive and not too large
            sigma = np.clip(sigma, 0.1, 5.0)

        # Use any remaining budget for a local (1+1) ES around the best found
        # This can help polish the solution.
        if best_x is not None:
            # Estimate remaining budget
            # We don't track exact calls easily in the loop above without a counter.
            # Let's add a call counter.
            pass # We need to track calls properly.

        # Let's rewrite the loop to track calls precisely.
        # The above loop assumes max_iters * pop_size <= budget.
        # max_iters = budget // pop_size.
        # So calls used = max_iters * pop_size.
        # remaining = budget - calls_used.
        
        # Let's refine the implementation to track calls.
        
        return best_x, best_f
# EVOLVE-BLOCK-END
