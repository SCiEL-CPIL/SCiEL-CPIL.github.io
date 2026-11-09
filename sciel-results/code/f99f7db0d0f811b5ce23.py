"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Differential Evolution with Non-dominated Sorting (NSGA-II style)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 300.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo

        # Population size
        # We have 200 evaluations. 
        # Initial population: 50
        # Generations: (200 - 50) / 50 = 3 generations? 
        # Let's try a smaller population for more generations or a standard one.
        # DE usually works well with pop size ~ 10 * dim. 10*12=120 is too big for 200 budget if we want generations.
        # Let's use pop_size = 40. 
        # 1 initial (40 evals) + 4 generations (40 evals each) = 200 evals.
        pop_size = 40
        
        # Parameters for DE
        CR = 0.9  # Crossover rate
        F = 0.8   # Scale factor

        # Initialize population randomly
        # Use np.random which is seeded by the evaluator
        X = np.random.rand(pop_size, self.dim)
        X = X * range_ + lo
        
        F_vals = np.array([problem(x) for x in X])
        evaluated_count = pop_size

        # Helper to compute non-dominated ranks and crowding distance
        def fast_non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]  # Dominated sets
            n_dom = np.zeros(n, dtype=int) # Number of dominators
            rank = np.zeros(n, dtype=int)
            
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dom[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_dom[i] += 1
            
            current_front = []
            for i in range(n):
                if n_dom[i] == 0:
                    rank[i] = 0
                    current_front.append(i)
            
            current_rank = 0
            while current_front:
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            rank[j] = current_rank + 1
                            next_front.append(j)
                current_rank += 1
                current_front = next_front
                
            return rank

        def crowding_distance(F, rank, front_indices):
            dist = np.zeros(len(F))
            if len(front_indices) <= 2:
                dist[front_indices] = float('inf')
                return dist
            
            m = self.n_obj
            for k in range(m):
                sorted_idx = np.argsort(F[front_indices, k])
                dist[sorted_idx[0]] = float('inf')
                dist[sorted_idx[-1]] = float('inf')
                
                if F[sorted_idx[-1], k] - F[sorted_idx[0], k] == 0:
                    continue
                    
                for i in range(1, len(sorted_idx) - 1):
                    idx = sorted_idx[i]
                    prev_idx = sorted_idx[i-1]
                    next_idx = sorted_idx[i+1]
                    dist[idx] += (F[next_idx, k] - F[prev_idx, k]) / (F[sorted_idx[-1], k] - F[sorted_idx[0], k])
            
            return dist

        # Main DE loop
        while evaluated_count < self.budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Create offspring
            offspring_X = np.zeros((pop_size, self.dim))
            offspring_F = np.zeros((pop_size, self.n_obj))
            
            for i in range(pop_size):
                if evaluated_count >= self.budget:
                    break
                    
                # Select 3 distinct random parents
                r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
                
                # Mutation
                v = X[r1] + F * (X[r2] - X[r3])
                
                # Crossover
                trial = np.random.rand(self.dim) < CR
                j_rand = np.random.randint(self.dim)
                trial[j_rand] = True
                
                trial_X = np.where(trial, v, X[i])
                
                # Bounds check
                trial_X = np.clip(trial_X, lo, hi)
                
                # Evaluate
                trial_F = problem(trial_X)
                evaluated_count += 1
                
                # Selection
                if np.all(trial_F <= F_vals[i]) and np.any(trial_F < F_vals[i]):
                    # Trial dominates parent
                    X[i] = trial_X
                    F_vals[i] = trial_F
                elif not (np.all(F_vals[i] <= trial_F) and np.any(F_vals[i] < trial_F)):
                    # Neither dominates, or parent dominates trial.
                    # In standard DE, we might just discard. 
                    # But to maintain diversity, we can keep the better one based on some metric or just keep parent.
                    # For simplicity and speed, let's just keep the parent if trial doesn't strictly dominate.
                    # Actually, a better strategy for MOO is to keep both in a combined population and sort.
                    # But that requires more bookkeeping. 
                    # Let's stick to simple DE replacement: replace if trial is better or non-dominated?
                    # If we only replace on dominance, we might stagnate.
                    # Let's use a simple rule: if trial is not dominated by parent, keep trial?
                    # No, that can lead to loss of good points.
                    # Standard DE for single objective replaces if better.
                    # For MOO, a common simple approach is to keep the one with better weighted sum or similar.
                    # However, since we have limited budget, let's just replace if trial is non-dominated by parent AND has better "spread" or just randomly?
                    # Let's try: Replace if trial dominates parent. Otherwise, keep parent.
                    # This is conservative.
                    # Alternative: Replace if trial is non-dominated by parent. This might increase diversity.
                    # Let's try replacing if trial is not dominated by parent.
                    pass 

            # Actually, the logic above inside the loop is inefficient for MOO.
            # Let's restructure: Generate all offspring, then combine and sort.
            # But we have a hard budget.
            
            # Revised Loop:
            # 1. Generate pop_size offspring.
            # 2. Evaluate them (budget check).
            # 3. Combine X and Offspring_X.
            # 4. Sort by non-domination.
            # 5. Select best pop_size for next generation.
            
            # We need to be careful with the budget.
            # Current logic:
            # Init: 40
            # Gen 1: 40 offspring -> Total 80. Select 40.
            # Gen 2: 40 offspring -> Total 120. Select 40.
            # Gen 3: 40 offspring -> Total 160. Select 40.
            # Gen 4: 40 offspring -> Total 200. Select 40.
            
            # Let's implement the "Combine and Sort" approach properly.
            
            # We already did init.
            # Let's restart the loop structure for clarity in code below.
            
            break # Break out of the dummy loop above to implement the real one below.
        
        # --- Real Implementation ---
        
        # Reset state for the real run if we used the dummy loop above? 
        # No, let's just write the clean code from scratch in the function.
        
        # Re-initialize to be safe and clean
        X = np.random.rand(pop_size, self.dim)
        X = X * range_ + lo
        F_vals = np.array([problem(x) for x in X])
        evaluated_count = pop_size

        while evaluated_count < self.budget:
            if time.time() > deadline:
                break
            
            # Generate offspring
            offspring_X = np.zeros((pop_size, self.dim))
            offspring_F = np.zeros((pop_size, self.n_obj))
            
            for i in range(pop_size):
                if evaluated_count >= self.budget:
                    break
                
                r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
                
                # Mutation
                v = X[r1] + F * (X[r2] - X[r3])
                
                # Crossover
                trial = np.random.rand(self.dim) < CR
                j_rand = np.random.randint(self.dim)
                trial[j_rand] = True
                
                trial_X = np.where(trial, v, X[i])
                trial_X = np.clip(trial_X, lo, hi)
                
                # Evaluate
                trial_F = problem(trial_X)
                evaluated_count += 1
                
                offspring_X[i] = trial_X
                offspring_F[i] = trial_F

            # Combine parent and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F_vals, offspring_F])
            
            # Non-dominated sort
            ranks = fast_non_dominated_sort(combined_F)
            
            # Select best pop_size
            # Sort indices by rank
            sorted_indices = np.argsort(ranks)
            
            # We need to handle crowding distance for the last front to ensure diversity
            # Find the max rank that fits in pop_size
            # This is a simplified selection. 
            # Better: Iterate through fronts.
            
            selected_indices = []
            current_front_size = 0
            
            # Get unique ranks
            unique_ranks = np.unique(ranks)
            
            for r in unique_ranks:
                front_idx = np.where(ranks == r)[0]
                
                if current_front_size + len(front_idx) <= pop_size:
                    # Take whole front
                    selected_indices.extend(front_idx)
                    current_front_size += len(front_idx)
                else:
                    # Need to select from this front
                    # Calculate crowding distance for this front
                    cd = crowding_distance(combined_F, ranks, front_idx)
                    # Sort by crowding distance descending
                    sorted_front = front_idx[np.argsort(cd[front_idx])[::-1]]
                    # Take remaining
                    remaining = pop_size - current_front_size
                    selected_indices.extend(sorted_front[:remaining])
                    break
            
            # Update population
            X = combined_X[selected_indices]
            F_vals = combined_F[selected_indices]

        return X

# EVOLVE-BLOCK-END
