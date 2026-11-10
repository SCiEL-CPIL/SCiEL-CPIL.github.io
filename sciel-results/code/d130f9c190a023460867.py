import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        # Population size
        N = 20
        
        # Initialize population
        X = lo + (hi - lo) * rng.random((N, dim))
        F = np.array([problem(x) for x in X])
        used = N
        
        # Helper: Dominance check
        def dominates(f1, f2):
            return np.all(f1 <= f2) and np.any(f1 < f2)
        
        # Helper: Non-dominated sort
        def non_dominated_sort(F):
            n = len(F)
            S = [set() for _ in range(n)]
            n_doms = [0] * n
            ranks = [0] * n
            
            for i in range(n):
                for j in range(i + 1, n):
                    if dominates(F[i], F[j]):
                        S[i].add(j)
                        n_doms[i] += 1
                    elif dominates(F[j], F[i]):
                        S[j].add(i)
                        n_doms[j] += 1
                        
            front_idx = 0
            current_front = []
            for i in range(n):
                if n_doms[i] == 0:
                    current_front.append(i)
                    ranks[i] = front_idx
            front_idx += 1
            
            while current_front:
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_doms[j] -= 1
                        if n_doms[j] == 0:
                            next_front.append(j)
                            ranks[j] = front_idx
                current_front = next_front
                front_idx += 1
            return ranks

        # Helper: Crowding distance
        def crowding_distance(F, front_indices):
            dists = {i: 0.0 for i in front_indices}
            if len(front_indices) <= 2:
                for i in front_indices:
                    dists[i] = float('inf')
                return dists
            
            for m in range(n_obj):
                sorted_indices = sorted(front_indices, key=lambda i: F[i][m])
                dists[sorted_indices[0]] = float('inf')
                dists[sorted_indices[-1]] = float('inf')
                
                f_min = F[sorted_indices[0]][m]
                f_max = F[sorted_indices[-1]][m]
                range_f = f_max - f_min
                if range_f == 0:
                    continue
                    
                for k in range(1, len(sorted_indices) - 1):
                    i = sorted_indices[k]
                    dists[i] += (F[sorted_indices[k+1]][m] - F[sorted_indices[k-1]][m]) / range_f
            return dists

        # Helper: Tournament selection
        def tournament_select(X, F, ranks, dists, k=2):
            participants = rng.choice(len(X), size=k, replace=False)
            best = participants[0]
            for p in participants[1:]:
                if ranks[p] < ranks[best]:
                    best = p
                elif ranks[p] == ranks[best]:
                    if dists[p] > dists[best]:
                        best = p
            return best

        # Helper: SBX Crossover
        def sbx_crossover(x1, x2):
            if rng.random() > 0.9:
                return x1.copy(), x2.copy()
            
            u1 = x1.copy()
            u2 = x2.copy()
            
            for i in range(dim):
                if rng.random() < 0.5:
                    continue
                if x1[i] == x2[i]:
                    continue
                if x1[i] < x2[i]:
                    x_lo, x_hi = x1[i], x2[i]
                else:
                    x_lo, x_hi = x2[i], x1[i]
                
                x_lo = max(x_lo, lo[i])
                x_hi = min(x_hi, hi[i])
                if x_hi == x_lo:
                    continue
                    
                beta = 1.0 + 2.0 * (x_lo - lo[i]) / (x_hi - x_lo)
                rand = rng.random()
                if rand <= 1.0 / (beta + 1.0):
                    betaq = (0.5 * rand * (beta + 1.0)) ** (1.0 / (1.0 + 1.0/15.0))
                else:
                    betaq = (0.5 * (1.0 / (1.0 - rand + 1e-10))) ** (1.0 / (1.0 + 1.0/15.0))
                
                c1 = 0.5 * (x1[i] + x2[i] - betaq * (x2[i] - x1[i]))
                c2 = 0.5 * (x1[i] + x2[i] + betaq * (x2[i] - x1[i]))
                
                u1[i] = np.clip(c1, lo[i], hi[i])
                u2[i] = np.clip(c2, lo[i], hi[i])
                
            return u1, u2

        # Helper: Polynomial Mutation
        def polynomial_mutate(x):
            for i in range(dim):
                if rng.random() < 1.0 / dim:
                    rand = rng.random()
                    if rand < 0.5:
                        delta = (2.0 * rand) ** (1.0 / (1.0 + 1.0/20.0)) - 1.0
                    else:
                        delta = 1.0 - (2.0 * (1.0 - rand)) ** (1.0 / (1.0 + 1.0/20.0))
                    
                    x[i] = x[i] + delta * (hi[i] - lo[i])
                    x[i] = np.clip(x[i], lo[i], hi[i])
            return x

        # Main Loop
        while used < self.budget:
            # Prepare selection data for current population
            ranks = non_dominated_sort(F)
            dists = np.zeros(N)
            rank_groups = {}
            for i in range(N):
                r = ranks[i]
                if r not in rank_groups:
                    rank_groups[r] = []
                rank_groups[r].append(i)
            
            for r, indices in rank_groups.items():
                d = crowding_distance(F, indices)
                for i in indices:
                    dists[i] = d[i]

            offspring_X = []
            offspring_F = []
            
            # Generate offspring until we fill N or run out of budget
            while len(offspring_X) < N and used < self.budget:
                p1_idx = tournament_select(X, F, ranks, dists, k=2)
                p2_idx = tournament_select(X, F, ranks, dists, k=2)
                
                c1, c2 = sbx_crossover(X[p1_idx], X[p2_idx])
                c1 = polynomial_mutate(c1)
                c2 = polynomial_mutate(c2)
                
                # Evaluate c1
                f1 = problem(c1)
                used += 1
                offspring_X.append(c1)
                offspring_F.append(f1)
                
                # Check if we can evaluate c2
                if used < self.budget:
                    f2 = problem(c2)
                    used += 1
                    offspring_X.append(c2)
                    offspring_F.append(f2)
                else:
                    # If we can't evaluate c2, discard it to stay within budget
                    # Or we could just break and use the partial offspring?
                    # NSGA-II usually expects full generations. 
                    # If we have an odd number of offspring, the combined population size will be N + len(offspring).
                    # We will select the best N from the combined population.
                    break

            # Combine populations
            combined_X = np.vstack([X, np.array(offspring_X)])
            combined_F = np.vstack([F, np.array(offspring_F)])
            combined_N = len(combined_F)
            
            # Non-dominated sort on combined population
            combined_ranks = non_dominated_sort(combined_F)
            
            # Prepare crowding distances for combined population
            combined_dists = np.zeros(combined_N)
            combined_rank_groups = {}
            for i in range(combined_N):
                r = combined_ranks[i]
                if r not in combined_rank_groups:
                    combined_rank_groups[r] = []
                combined_rank_groups[r].append(i)
            
            for r, indices in combined_rank_groups.items():
                d = crowding_distance(combined_F, indices)
                for i in indices:
                    combined_dists[i] = d[i]
                    
            # Select next generation
            next_X = []
            next_F = []
            
            # Sort combined population by rank, then by crowding distance
            # We need to pick N individuals.
            # Group by rank
            selected_indices = []
            for r in sorted(combined_rank_groups.keys()):
                if len(selected_indices) + len(combined_rank_groups[r]) <= N:
                    selected_indices.extend(combined_rank_groups[r])
                else:
                    # Need to pick from this front
                    needed = N - len(selected_indices)
                    # Sort this front by crowding distance descending
                    front_indices = sorted(combined_rank_groups[r], key=lambda i: combined_dists[i], reverse=True)
                    selected_indices.extend(front_indices[:needed])
                    break
                    
            for i in selected_indices:
                next_X.append(combined_X[i])
                next_F.append(combined_F[i])
                
            X = np.array(next_X)
            F = np.array(next_F)
            
        return X
