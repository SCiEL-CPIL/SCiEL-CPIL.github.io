"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """NSGA-II multi-objective optimizer for ZDT2."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        budget = self.budget
        
        # Parameters
        # With budget 200 and dim 30, we need a small population.
        # Let's try population size N=20.
        # Initial eval: 20.
        # Each generation: 20 offspring.
        # Generations: (200 - 20) / 20 = 9 generations.
        # Total 10 evaluations of 20.
        N = 20
        
        # SBX parameters
        eta_c = 15.0  # Distribution index for crossover
        prob_c = 0.9  # Probability of crossover
        
        # Polynomial Mutation parameters
        eta_m = 20.0  # Distribution index for mutation
        prob_m = 1.0 / dim  # Probability of mutation per gene

        # Initialize Population
        X = lo + (hi - lo) * np.random.random((N, dim))
        evaluations = 0
        
        # Evaluate initial population
        F = np.zeros((N, self.n_obj))
        for i in range(N):
            F[i] = problem(X[i])
            evaluations += 1
            
        def dominates(f1, f2):
            """Check if f1 dominates f2 (minimization)."""
            return np.all(f1 <= f2) and np.any(f1 < f2)

        def fast_non_dominated_sort(F):
            """Perform fast non-dominated sorting."""
            n = F.shape[0]
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            n_d = np.zeros(n, dtype=int)  # n_d[i] = number of solutions dominating i
            rank = np.zeros(n, dtype=int)
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if dominates(F[i], F[j]):
                        S[i].append(j)
                    elif dominates(F[j], F[i]):
                        n_d[i] += 1
                        
            # First front
            front_idx = 0
            current_front = []
            for i in range(n):
                if n_d[i] == 0:
                    rank[i] = front_idx
                    current_front.append(i)
                    
            while current_front:
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_d[j] -= 1
                        if n_d[j] == 0:
                            rank[j] = front_idx + 1
                            next_front.append(j)
                front_idx += 1
                current_front = next_front
                
            return rank

        def crowding_distance(F, rank):
            """Calculate crowding distance for solutions in the same front."""
            n = F.shape[0]
            cd = np.zeros(n)
            
            # Group by rank
            ranks = np.unique(rank)
            for r in ranks:
                indices = np.where(rank == r)[0]
                m = len(indices)
                if m <= 2:
                    cd[indices] = np.inf
                    continue
                    
                # Sort by each objective
                for obj in range(self.n_obj):
                    sorted_indices = indices[np.argsort(F[indices, obj])]
                    cd[sorted_indices[0]] = np.inf
                    cd[sorted_indices[-1]] = np.inf
                    f_min = F[sorted_indices[0], obj]
                    f_max = F[sorted_indices[-1], obj]
                    if f_max - f_min < 1e-15:
                        continue
                    for k in range(1, m - 1):
                        idx = sorted_indices[k]
                        cd[idx] += (F[sorted_indices[k+1], obj] - F[sorted_indices[k-1], obj]) / (f_max - f_min)
                        
            return cd

        def tournament_selection(rank, cd, k=2):
            """Select one individual using tournament selection."""
            idx1 = np.random.randint(N)
            idx2 = np.random.randint(N)
            
            if rank[idx1] < rank[idx2]:
                return idx1
            elif rank[idx2] < rank[idx1]:
                return idx2
            else:
                if cd[idx1] > cd[idx2]:
                    return idx1
                else:
                    return idx2

        def sbx_crossover(x1, x2):
            """Simulated Binary Crossover."""
            if np.random.random() > prob_c:
                return x1.copy(), x2.copy()
                
            x1_new = x1.copy()
            x2_new = x2.copy()
            
            for i in range(dim):
                if np.random.random() < 0.5:
                    if abs(x1[i] - x2[i]) < 1e-14:
                        continue
                    r = np.random.random()
                    if r <= 0.5:
                        beta = (2 * r) ** (1.0 / (eta_c + 1))
                    else:
                        beta = (1.0 / (2 * (1 - r))) ** (1.0 / (eta_c + 1))
                    x1_new[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                    x2_new[i] = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
                    
            x1_new = np.clip(x1_new, lo, hi)
            x2_new = np.clip(x2_new, lo, hi)
            
            return x1_new, x2_new

        def polynomial_mutation(x):
            """Polynomial Mutation."""
            x_new = x.copy()
            for i in range(dim):
                if np.random.random() < prob_m:
                    r = np.random.random()
                    if r < 0.5:
                        delta = (2 * r) ** (1.0 / (eta_m + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - r)) ** (1.0 / (eta_m + 1))
                    
                    x_new[i] = x[i] + delta * (hi[i] - lo[i])
                    x_new[i] = np.clip(x_new[i], lo[i], hi[i])
                    
            return x_new

        # Main Evolution Loop
        while evaluations < budget:
            # Generate offspring
            offspring_X = np.zeros((N, dim))
            offspring_F = np.zeros((N, self.n_obj))
            
            # Calculate rank and crowding distance for current population
            rank = fast_non_dominated_sort(F)
            cd = crowding_distance(F, rank)
            
            for i in range(N // 2):
                # Select parents
                idx1 = tournament_selection(rank, cd)
                idx2 = tournament_selection(rank, cd)
                
                # Crossover
                child1, child2 = sbx_crossover(X[idx1], X[idx2])
                
                # Mutation
                child1 = polynomial_mutation(child1)
                child2 = polynomial_mutation(child2)
                
                offspring_X[2*i] = child1
                offspring_X[2*i+1] = child2
                
                # Evaluate offspring
                if evaluations < budget:
                    offspring_F[2*i] = problem(child1)
                    evaluations += 1
                if evaluations < budget:
                    offspring_F[2*i+1] = problem(child2)
                    evaluations += 1
                else:
                    # If budget runs out mid-generation, we stop.
                    # The loop condition will break after this iteration.
                    # We need to handle the case where we don't have a full offspring set.
                    # But the contract says "More than 200 evaluations... makes the whole result invalid".
                    # So we must ensure we do not exceed budget.
                    # The while loop checks `evaluations < budget`.
                    # If we are at budget-1, we evaluate one, evaluations becomes budget.
                    # The next check `if evaluations < budget` will fail.
                    # But we are inside the for loop.
                    # We should break out of the for loop if we hit the budget.
                    pass
            
            # If we broke early due to budget, we might have a partial offspring set.
            # However, the standard NSGA-II combines parents and offspring.
            # If we don't have a full offspring set, we can't do the standard combine.
            # Given the budget is 200 and N=20, we have exactly 9 full generations after init.
            # 20 (init) + 9 * 20 (offspring) = 200.
            # So we should always have a full offspring set if we manage the loop correctly.
            # Let's verify:
            # Init: 20 evals.
            # Gen 1: 20 evals. Total 40.
            # ...
            # Gen 9: 20 evals. Total 200.
            # So we will always have a full offspring set.
            
            # Combine parents and offspring
            X_combined = np.vstack([X, offspring_X])
            F_combined = np.vstack([F, offspring_F])
            
            # Sort combined population
            rank_combined = fast_non_dominated_sort(F_combined)
            cd_combined = crowding_distance(F_combined, rank_combined)
            
            # Select best N individuals
            # Sort indices by rank, then by crowding distance (descending)
            # We need to pick the first N individuals.
            # Create a list of indices sorted by (rank, -cd)
            indices = np.arange(2 * N)
            # Stable sort by rank, then by -cd
            # np.lexsort uses the last key as the primary key.
            # So we want to sort by -cd first (secondary), then rank (primary)?
            # No, we want rank to be primary (smaller is better), and cd to be secondary (larger is better).
            # So sort by rank, then by -cd.
            # lexsort keys: (-cd_combined, rank_combined)
            # Wait, lexsort sorts by the last key first? No, "The last key is used as the primary sort key."
            # So keys = (-cd_combined, rank_combined) means rank is primary, -cd is secondary.
            # This is correct.
            sorted_indices = np.lexsort((-cd_combined, rank_combined))
            
            # Take the first N
            X = X_combined[sorted_indices[:N]]
            F = F_combined[sorted_indices[:N]]
            
        return None
# EVOLVE-BLOCK-END
