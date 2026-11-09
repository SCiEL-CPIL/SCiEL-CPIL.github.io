"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer using NSGA-II.
    
    Strategy:
    1. Initialize a population of 50 individuals using Sobol sampling for good spread.
    2. Iterate through generations. Each generation involves:
       a. Evaluating new individuals.
       b. Merging parent and offspring populations.
       c. Non-dominated sorting to assign ranks.
       d. Calculating crowding distance within each rank.
       e. Selecting the best 'population_size' individuals for the next generation 
          based on rank and crowding distance.
    3. Use a binary tournament selection for parents and SBX (Simulated Binary Crossover) 
       and Polynomial Mutation for offspring generation.
    4. Continue until the budget (200 evaluations) is exhausted or time limit is reached.
    
    This approach is structurally different from scalarization-based methods as it 
    directly optimizes the set of non-dominated points, preserving diversity via 
    crowding distance rather than relying on weight vectors.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.pop_size = 50  # Population size for NSGA-II

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin: 300s limit, kill at 290s
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        # Evaluation counter
        eval_count = 0
        
        # Storage for all evaluated points and their fitness
        # We store (x, f) pairs
        all_X = []
        all_F = []
        
        # Helper function to evaluate a batch of points
        def evaluate(X_batch):
            nonlocal eval_count
            F_batch = []
            for x in X_batch:
                if eval_count >= self.budget or time.time() > deadline:
                    break
                f = problem(x)
                F_batch.append(f)
                eval_count += 1
            return F_batch

        # Initialize population using Sobol sequence for uniform coverage
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=42)
        X_init = sampler.random(self.pop_size)
        X_init = lo + range_ * X_init
        
        # Evaluate initial population
        F_init = evaluate(X_init)
        
        # If we ran out of budget just initializing, return what we have
        if eval_count >= self.budget or time.time() > deadline:
            return F_init

        all_X.extend(X_init)
        all_F.extend(F_init)
        
        # NSGA-II Main Loop
        # We maintain a population of size P. 
        # Each generation creates P offspring. 
        # Total evaluations per generation = P (for offspring, assuming parents are already evaluated).
        # We need to be careful: NSGA-II usually evaluates offspring. 
        # Parents are already in the archive.
        
        current_P = X_init
        current_F = F_init
        
        # Parameters for genetic operators
        crossover_rate = 0.9
        mutation_rate = 1.0 / self.dim
        eta_c = 15  # Distribution index for SBX
        eta_m = 20  # Distribution index for Polynomial Mutation
        
        while time.time() < deadline:
            # Check if we have enough budget for a full generation
            if eval_count + self.pop_size > self.budget:
                # If we don't have enough for a full generation, 
                # we might do a partial generation or just stop.
                # Given the scoring depends on hypervolume of ALL recorded points,
                # it's better to stop cleanly than to have an incomplete generation 
                # that might not contribute much. 
                # However, if we have *some* budget left, we can evaluate a few more 
                # random points or perturbations of the best points.
                # For simplicity and robustness, let's just stop if we can't do a full gen.
                # Actually, let's try to use remaining budget for local refinements 
                # or just random sampling if it's small.
                
                # Simple fallback: use remaining budget for random sampling
                remaining = self.budget - eval_count
                if remaining > 0:
                    X_rand = lo + range_ * np.random.rand(remaining, self.dim)
                    F_rand = evaluate(X_rand)
                    all_X.extend(X_rand[:len(F_rand)])
                    all_F.extend(F_rand)
                break

            # 1. Selection and Crossover/Mutation to create offspring
            offspring = []
            for _ in range(self.pop_size):
                # Binary Tournament Selection
                idx1, idx2 = np.random.choice(len(current_P), 2, replace=False)
                
                # Compare fitness
                # We need a comparison function that considers non-dominance and crowding distance
                # Since we don't have ranks/distances stored for current_P explicitly in a quick way 
                # without re-sorting, we can do a simple non-domination check on the fly 
                # or just pick randomly if we assume the population is already somewhat optimized.
                # Standard NSGA-II uses ranks. Let's implement a lightweight rank check 
                # or just use random selection to keep it fast, relying on the 
                # selection step after offspring generation to prune.
                # Actually, for a small dim=4, pop=50, we can afford to do a proper 
                # non-dominated sort on the current population to get ranks for selection.
                
                # To keep code concise and fast, we will use a simplified selection:
                # Pick two random parents. If one dominates the other, pick the dominator.
                # If they are non-dominated, pick one randomly.
                f1 = current_F[idx1]
                f2 = current_F[idx2]
                
                if self._dominates(f1, f2):
                    parent1_idx = idx1
                elif self._dominates(f2, f1):
                    parent1_idx = idx2
                else:
                    parent1_idx = idx1 if np.random.rand() < 0.5 else idx2
                    
                # Repeat for second parent
                idx3, idx4 = np.random.choice(len(current_P), 2, replace=False)
                f3 = current_F[idx3]
                f4 = current_F[idx4]
                
                if self._dominates(f3, f4):
                    parent2_idx = idx3
                elif self._dominates(f4, f3):
                    parent2_idx = idx4
                else:
                    parent2_idx = idx3 if np.random.rand() < 0.5 else idx4
                
                p1 = current_P[parent1_idx]
                p2 = current_P[parent2_idx]
                
                # Crossover (SBX)
                if np.random.rand() < crossover_rate:
                    c1, c2 = self._sbx(p1, p2, lo, hi, eta_c)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                
                # Mutation (Polynomial)
                c1 = self._polynomial_mutation(c1, lo, hi, mutation_rate, eta_m)
                c2 = self._polynomial_mutation(c2, lo, hi, mutation_rate, eta_m)
                
                offspring.append(c1)
                offspring.append(c2)
            
            # We generated 2 * pop_size offspring. We only need pop_size.
            # Take the first pop_size.
            offspring = np.array(offspring[:self.pop_size])
            
            # 2. Evaluate Offspring
            F_offspring = evaluate(offspring)
            
            # If evaluation stopped early due to budget/time, adjust offspring size
            if len(F_offspring) < self.pop_size:
                offspring = offspring[:len(F_offspring)]
            
            # 3. Merge Parent and Offspring
            P = np.vstack([current_P, offspring])
            F = np.vstack([current_F, F_offspring])
            
            # 4. Non-dominated Sorting
            ranks = self._non_dominated_sort(F)
            
            # 5. Crowding Distance
            # We calculate crowding distance for each front (rank)
            # and select the best 'pop_size' individuals.
            
            # Group indices by rank
            rank_groups = {}
            for i, r in enumerate(ranks):
                if r not in rank_groups:
                    rank_groups[r] = []
                rank_groups[r].append(i)
            
            # Sort ranks
            sorted_ranks = sorted(rank_groups.keys())
            
            new_P_indices = []
            for r in sorted_ranks:
                group_indices = rank_groups[r]
                if len(new_P_indices) + len(group_indices) <= self.pop_size:
                    # Take all
                    new_P_indices.extend(group_indices)
                else:
                    # Need to select some from this front
                    # Calculate crowding distance for this front
                    F_front = F[group_indices]
                    cd = self._crowding_distance(F_front)
                    
                    # Sort by crowding distance (descending)
                    sorted_idx = np.argsort(-cd)
                    needed = self.pop_size - len(new_P_indices)
                    for i in sorted_idx[:needed]:
                        new_P_indices.append(group_indices[i])
                    break
            
            # Update current population
            current_P = P[new_P_indices]
            current_F = F[new_P_indices]
            
            # Add to global archive (for final hypervolume calculation)
            all_X.extend(current_P)
            all_F.extend(current_F)
            
            # Note: We are adding the *new* population to the global list.
            # The global list will contain duplicates. This is fine for hypervolume 
            # calculation as dominated points or duplicates don't increase the volume, 
            # but it increases the list size. 
            # To be more efficient, we could maintain a non-dominated set, 
            # but the task says "Every returned objective vector is recorded".
            # The evaluator records the return of __call__? No, "Every returned objective vector is recorded"
            # implies the evaluator calls problem(x) and records the result?
            # "Every returned objective vector is recorded and the return value of __call__ is ignored."
            # This phrasing is slightly ambiguous. Usually, in these tasks, the evaluator 
            # wraps `problem` to record calls.
            # "More than 200 evaluations... makes the whole result invalid".
            # This confirms `problem(x)` calls are counted.
            # The scoring is based on "all recorded vectors". 
            # So we just need to ensure we don't exceed 200 calls to `problem`.
            # The list `all_F` is just for our internal bookkeeping if we wanted to return it, 
            # but we return `all_F`? 
            # "the return value of __call__ is ignored".
            # So we can return anything. I will return the last population's F or just an empty list.
            # But wait, if the evaluator records the *calls* to problem, then the 
            # "recorded vectors" are the F values from the 200 calls.
            # So my internal list `all_F` is not what is scored, the evaluator's log is.
            # So I just need to make good use of the 200 calls.

        return [] # Return value is ignored

    def _dominates(self, f1, f2):
        """Check if f1 dominates f2 (minimization)."""
        # f1 dominates f2 if f1 <= f2 for all objectives and f1 < f2 for at least one
        if not np.all(f1 <= f2):
            return False
        if not np.any(f1 < f2):
            return False
        return True

    def _non_dominated_sort(self, F):
        """
        Perform non-dominated sorting on a set of fitness vectors.
        Returns an array of ranks (0 is the best front).
        """
        n = len(F)
        ranks = np.zeros(n, dtype=int)
        S = [[] for _ in range(n)]  # S[i] = indices dominated by i
        n_dominating = np.zeros(n, dtype=int)
        
        for i in range(n):
            for j in range(i + 1, n):
                if self._dominates(F[i], F[j]):
                    S[i].append(j)
                    n_dominating[j] += 1
                elif self._dominates(F[j], F[i]):
                    S[j].append(i)
                    n_dominating[i] += 1
        
        # First front
        front = []
        for i in range(n):
            if n_dominating[i] == 0:
                ranks[i] = 0
                front.append(i)
        
        k = 0
        while front:
            next_front = []
            for i in front:
                for j in S[i]:
                    n_dominating[j] -= 1
                    if n_dominating[j] == 0:
                        ranks[j] = k + 1
                        next_front.append(j)
            front = next_front
            k += 1
        
        return ranks

    def _crowding_distance(self, F):
        """
        Calculate crowding distance for a front of points.
        F is an array of shape (N, n_obj).
        """
        N = len(F)
        if N <= 2:
            return np.full(N, np.inf)
        
        cd = np.zeros(N)
        for m in range(F.shape[1]):
            # Sort indices by objective m
            order = np.argsort(F[:, m])
            cd[order[0]] = np.inf
            cd[order[-1]] = np.inf
            
            f_min = F[order[0], m]
            f_max = F[order[-1], m]
            
            if f_max == f_min:
                continue
                
            for i in range(1, N - 1):
                cd[order[i]] += (F[order[i+1], m] - F[order[i-1], m]) / (f_max - f_min)
        
        return cd

    def _sbx(self, p1, p2, lo, hi, eta):
        """Simulated Binary Crossover."""
        c1 = p1.copy()
        c2 = p2.copy()
        
        for i in range(self.dim):
            if np.random.rand() < 0.5:
                if abs(p1[i] - p2[i]) > 1e-14:
                    if np.random.rand() <= 1.0 / (eta + 1):
                        beta = (2 * np.random.rand()) ** (1.0 / (eta + 1))
                    else:
                        beta = (1.0 / (2 * np.random.rand())) ** (1.0 / (eta + 1))
                    
                    c1[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
                    c2[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])
                    
                    # Clip to bounds
                    c1[i] = np.clip(c1[i], lo[i], hi[i])
                    c2[i] = np.clip(c2[i], lo[i], hi[i])
        
        return c1, c2

    def _polynomial_mutation(self, x, lo, hi, rate, eta):
        """Polynomial Mutation."""
        for i in range(self.dim):
            if np.random.rand() < rate:
                y = x[i]
                y_min = lo[i]
                y_max = hi[i]
                
                if y_min == y_max:
                    continue
                
                delta1 = (y - y_min) / (y_max - y_min)
                delta2 = (y_max - y) / (y_max - y_min)
                
                r = np.random.rand()
                if r < 0.5:
                    xy = 2 * r
                    c = (1 + delta1) ** (eta + 1)
                    beta = (1 + (eta + 1) * (xy - 1)) ** (-1.0 / (eta + 1)) if xy > 0 else 1
                    # Correction for polynomial mutation formula
                    # Standard formula:
                    # if r < 0.5:
                    #   delta = (2r)^(1/(eta+1))
                    #   x_new = x - delta * (x_max - x_min)
                    # else:
                    #   delta = (2(1-r))^(1/(eta+1))
                    #   x_new = x + delta * (x_max - x_min)
                    
                    delta = (2 * r) ** (1.0 / (eta + 1))
                    x[i] = y - delta * (y_max - y_min)
                else:
                    r = 1 - r
                    delta = (2 * r) ** (1.0 / (eta + 1))
                    x[i] = y + delta * (y_max - y_min)
                
                x[i] = np.clip(x[i], lo[i], hi[i])
        
        return x
# EVOLVE-BLOCK-END
