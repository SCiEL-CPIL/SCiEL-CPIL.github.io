"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq

class Optimizer:
    """Multi-objective optimizer using Population-Based Evolutionary Strategy (ES)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: The evaluator kills at 330s. 
        # We aim to finish well before that to ensure validity.
        deadline = start_time + 280 
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Initialize population with Sobol or random points for diversity
        # We want a population size that allows multiple generations within budget.
        # Let's use pop_size = 40. 
        # Budget 200 / 40 = 5 generations. 
        # Actually, (mu+lambda) ES: 
        # Let's use mu=20, lambda=20. Total 40 evals per gen.
        # 200 / 40 = 5 generations.
        
        # To be safer and allow more exploration, let's use smaller population
        # but more generations? 
        # Hypervolume benefits from diversity. 
        # Let's try pop_size = 20.
        # 200 / 20 = 10 generations.
        
        pop_size = 20
        rng = np.random.default_rng(42)
        
        # Initial population: Random sampling
        # Use stratified sampling if possible, but random is fine for 5D
        population_x = np.zeros((pop_size, self.dim))
        for i in range(pop_size):
            population_x[i] = lo + (hi - lo) * rng.random(self.dim)
            
        # Evaluate initial population
        population_f = np.zeros((pop_size, self.n_obj))
        evals_used = 0
        
        for i in range(pop_size):
            if evals_used >= self.budget or time.time() > deadline:
                break
            f = problem(population_x[i])
            population_f[i] = f
            evals_used += 1
            
        # Store all evaluated points for final hypervolume calculation?
        # The task says: "Every returned objective vector is recorded".
        # So we don't need to return them. We just need to call problem() enough times
        # with good points.
        
        # Main Evolution Loop
        # We will use a (mu + lambda) selection strategy.
        # mu = pop_size (keep best mu from combined population)
        # lambda = pop_size (generate lambda offspring)
        
        # We need to track all non-dominated points found so far?
        # No, the evaluator records ALL calls to problem.
        # So even if a point is dominated in the final set, it was recorded.
        # However, to maximize hypervolume, we want the *recorded* set to be as good as possible.
        # The recorded set is the union of all points we ever evaluated.
        # So we should ensure that the points we evaluate are diverse and high quality.
        
        # Strategy:
        # 1. Maintain a population of the current best non-dominated points (or just best points).
        # 2. Generate offspring via mutation.
        # 3. Evaluate offspring.
        # 4. Combine parent and offspring.
        # 5. Select best mu points (non-dominated first, then crowding/hypervolume contribution).
        
        # For simplicity and robustness, let's use a simple non-dominated sorting.
        # If we run out of budget, we stop.
        
        while evals_used < self.budget and time.time() < deadline:
            # Generate offspring
            # Use Gaussian mutation with adaptive sigma
            # Initial sigma: 10% of range
            sigma = 0.1 * (hi - lo)
            
            offspring_x = np.zeros((pop_size, self.dim))
            for i in range(pop_size):
                # Pick a parent randomly
                parent_idx = rng.integers(0, pop_size)
                # Mutate
                offspring_x[i] = population_x[parent_idx] + rng.normal(0, sigma, self.dim)
                # Clip to bounds
                offspring_x[i] = np.clip(offspring_x[i], lo, hi)
                
            # Evaluate offspring
            offspring_f = np.zeros((pop_size, self.n_obj))
            for i in range(pop_size):
                if evals_used >= self.budget or time.time() > deadline:
                    break
                f = problem(offspring_x[i])
                offspring_f[i] = f
                evals_used += 1
                
            # Combine population and offspring
            combined_x = np.vstack([population_x, offspring_x])
            combined_f = np.vstack([population_f, offspring_f])
            
            # Selection: Keep best pop_size points
            # We need a fast way to select the best points.
            # 1. Non-dominated sorting.
            # 2. Among non-dominated, keep those with best crowding distance or hypervolume contribution.
            
            # Since pop_size is small (20), O(N^2) is fine.
            # Let's implement a simple non-dominated filter.
            
            # Identify non-dominated points in combined_f
            # Point i is dominated if there exists j such that f_j <= f_i and f_j != f_i
            n_total = len(combined_f)
            dominated_mask = np.zeros(n_total, dtype=bool)
            
            for i in range(n_total):
                if dominated_mask[i]:
                    continue
                for j in range(n_total):
                    if i == j or dominated_mask[j]:
                        continue
                    # Check if j dominates i
                    # f_j <= f_i and f_j != f_i
                    if np.all(combined_f[j] <= combined_f[i]) and np.any(combined_f[j] < combined_f[i]):
                        dominated_mask[i] = True
                        break
            
            # Get indices of non-dominated points
            non_dom_indices = np.where(~dominated_mask)[0]
            
            # If we have more non-dominated points than we can keep, we need to select
            # If we have fewer, we fill the rest with the best dominated points?
            # Actually, in MOEA, we usually fill fronts until we reach mu.
            # But here, we just want to keep the best 20.
            
            # Let's just keep the non-dominated ones if count <= pop_size.
            # If count > pop_size, we need to select the best ones.
            # A simple heuristic: Keep the ones with the smallest sum of objectives?
            # Or use crowding distance.
            
            # For simplicity and speed, let's just keep the first pop_size non-dominated points
            # and if there are fewer, add the best dominated ones based on scalarized value.
            
            if len(non_dom_indices) >= pop_size:
                # Select pop_size from non_dom_indices
                # Random selection might lose good points. 
                # Let's select based on a simple scalarization (sum) to prioritize "good" points
                # But this might hurt diversity.
                # Better: Use a simple grid or random.
                # Given small size, let's just take a random sample to maintain diversity?
                # Or sort by sum of objectives?
                # Let's sort by sum of objectives (normalized?)
                # Since we don't know scales, let's just take random non-dominated points.
                # This is risky.
                
                # Alternative: Keep ALL non-dominated if possible?
                # No, population size is fixed for the algorithm to work.
                
                # Let's use a simple heuristic: Keep the points that are farthest from each other?
                # Too complex.
                
                # Let's just keep the first pop_size non-dominated points.
                # To be fair, shuffle them first?
                # The order in combined_f is parents then offspring.
                # Let's shuffle the non_dom_indices to avoid bias towards parents.
                shuffled_indices = rng.permutation(non_dom_indices)
                selected_indices = shuffled_indices[:pop_size]
                
            else:
                # We have fewer than pop_size non-dominated points.
                # We need to add some dominated points to fill the population.
                # Which dominated points? The "least dominated" or "best" ones.
                # Let's select dominated points based on the minimum number of dominating points.
                
                dom_indices = np.where(dominated_mask)[0]
                
                if len(dom_indices) > 0:
                    # Calculate "domination count" for each dominated point
                    # Or just pick the ones with smallest sum of objectives?
                    # Let's pick the ones with smallest sum of objectives.
                    # This might bias towards one objective, but it's a fallback.
                    
                    # Better: Pick random dominated points to maintain diversity in the next gen?
                    # No, we want to keep the "best" dominated points to improve.
                    
                    # Let's sort dominated points by sum of objectives.
                    sums = np.sum(combined_f[dom_indices], axis=1)
                    sorted_dom = dom_indices[np.argsort(sums)]
                    
                    needed = pop_size - len(non_dom_indices)
                    selected_indices = np.concatenate([non_dom_indices, sorted_dom[:needed]])
                else:
                    # All non-dominated? (Shouldn't happen if len < pop_size and total >= pop_size)
                    selected_indices = non_dom_indices
            
            # Update population
            population_x = combined_x[selected_indices]
            population_f = combined_f[selected_indices]
            
            # Adapt sigma?
            # If we are stagnating, increase sigma.
            # For now, keep sigma constant.
            # Maybe decrease sigma slightly to refine?
            # sigma *= 0.95
            
        return []
# EVOLVE-BLOCK-END
