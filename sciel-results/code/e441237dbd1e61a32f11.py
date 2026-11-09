import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        # Population size
        # We need to fit within budget.
        # Let's use pop_size = 20.
        # Initial evals: 20.
        # Remaining: 180.
        # Generations: 180 // 20 = 9.
        # Total: 20 + 9*20 = 200.
        
        pop_size = 20
        if pop_size > self.budget:
            pop_size = self.budget
            
        # Initialize
        X = lo + (hi - lo) * np.random.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        evals = pop_size
        
        # NSGA-II Loop
        while evals + 2 * pop_size <= self.budget or (evals < self.budget and pop_size == 1):
            # Check time
            if time.time() > deadline:
                break
                
            # Fast Non-Dominated Sorting
            ranks, fronts = self._fast_non_dominated_sort(F)
            
            # Crowding Distance
            cd = self._crowding_distance(F, fronts)
            
            # Create offspring
            offspring_X = []
            offspring_F = []
            
            # Generate pop_size offspring (2 per pair, so pop_size/2 pairs? No, NSGA-II generates N offspring from N parents)
            # Standard NSGA-II: Select N parents, create N children.
            # So we need to generate pop_size children.
            
            for _ in range(pop_size):
                # Selection: Binary Tournament
                p1 = self._tournament_select(ranks, cd, pop_size)
                p2 = self._tournament_select(ranks, cd, pop_size)
                
                # Crossover (SBX)
                c1, c2 = self._sbx_crossover(X[p1], X[p2], lo, hi)
                
                # Mutation (Polynomial)
                c1 = self._polynomial_mutation(c1, lo, hi)
                c2 = self._polynomial_mutation(c2, lo, hi)
                
                # We need to add exactly pop_size children to the offspring pool.
                # The loop above adds 2 children per iteration.
                # If pop_size is even, we do pop_size/2 iterations.
                # If pop_size is odd, we do (pop_size+1)//2 iterations and discard one?
                # Standard practice: Generate 2 children per pair. If we need N children, we do N/2 pairs.
                # Let's adjust the loop to generate exactly pop_size children.
                
                # Actually, simpler: 
                # For i in range(pop_size // 2):
                #   Generate 2 children
                # If pop_size is odd, we might have 1 left over.
                # Let's just generate pop_size children by doing pop_size//2 pairs and handling the remainder.
                # Or simpler: Just run the loop pop_size//2 times, and if pop_size is odd, add one more child from a single parent mutation?
                # Let's stick to even pop_size for simplicity. 20 is even.
                
                offspring_X.append(c1)
                offspring_X.append(c2)
                
                if evals + 1 <= self.budget:
                    f1 = problem(c1)
                    offspring_F.append(f1)
                    evals += 1
                else:
                    break
                    
                if evals + 1 <= self.budget:
                    f2 = problem(c2)
                    offspring_F.append(f2)
                    evals += 1
                else:
                    break
            
            if len(offspring_X) == 0:
                break

            # Combine Parent and Offspring
            # Ensure offspring_X is a list of arrays
            X_new = np.vstack([X, np.array(offspring_X)])
            F_new = np.vstack([F, np.array(offspring_F)])
            
            # Selection for next generation
            ranks_new, fronts_new = self._fast_non_dominated_sort(F_new)
            cd_new = self._crowding_distance(F_new, fronts_new)
            
            # Select best pop_size from 2*pop_size
            X_next = []
            F_next = []
            for front in fronts_new:
                if len(X_next) + len(front) <= pop_size:
                    X_next.extend([X_new[i] for i in front])
                    F_next.extend([F_new[i] for i in front])
                else:
                    # Need to pick some from this front based on crowding distance
                    front_cd = [cd_new[i] for i in front]
                    # Sort by cd descending
                    sorted_front = sorted(front, key=lambda i: front_cd[i], reverse=True)
                    needed = pop_size - len(X_next)
                    for i in sorted_front[:needed]:
                        X_next.append(X_new[i])
                        F_next.append(F_new[i])
                    break
            
            X = np.array(X_next)
            F = np.array(F_next)
            
        return X

    def _fast_non_dominated_sort(self, F):
        n = len(F)
        S = [[] for _ in range(n)]  # Dominated sets
        n_dominating = np.zeros(n, dtype=int)
        ranks = np.zeros(n, dtype=int)
        
        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_dominating[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    n_dominating[i] += 1
        
        fronts = []
        current_front = [i for i in range(n) if n_dominating[i] == 0]
        k = 0
        while current_front:
            fronts.append(current_front)
            next_front = []
            for i in current_front:
                for j in S[i]:
                    n_dominating[j] -= 1
                    if n_dominating[j] == 0:
                        ranks[j] = k
                        next_front.append(j)
            current_front = next_front
            k += 1
        
        # Assign ranks to remaining (if any)
        for i in range(n):
            if ranks[i] == 0 and i not in (fronts[0] if fronts else []):
                 # This case shouldn't happen if logic is correct, but just in case
                 pass
                 
        # Re-assign ranks properly based on fronts list
        for idx, front in enumerate(fronts):
            for i in front:
                ranks[i] = idx
                
        return ranks, fronts

    def _crowding_distance(self, F, fronts):
        n = len(F)
        cd = np.zeros(n)
        
        for front in fronts:
            if len(front) <= 2:
                for i in front:
                    cd[i] = float('inf')
                continue
            
            # Normalize objectives? Standard NSGA-II does.
            front_F = F[front]
            min_F = front_F.min(axis=0)
            max_F = front_F.max(axis=0)
            range_F = max_F - min_F
            range_F[range_F == 0] = 1  # Avoid division by zero
            
            for m in range(F.shape[1]):
                # Sort front by objective m
                sorted_indices = np.argsort(front_F[:, m])
                sorted_front = [front[i] for i in sorted_indices]
                
                # Assign inf to best and worst
                cd[sorted_front[0]] = float('inf')
                cd[sorted_front[-1]] = float('inf')
                
                # Assign distance to others
                for k in range(1, len(sorted_front) - 1):
                    i_prev = sorted_front[k - 1]
                    i_next = sorted_front[k + 1]
                    # Distance in objective m
                    dist = (front_F[i_next, m] - front_F[i_prev, m]) / range_F[m]
                    cd[sorted_front[k]] += dist
        
        return cd

    def _tournament_select(self, ranks, cd, pop_size):
        # Binary tournament
        i = np.random.randint(pop_size)
        j = np.random.randint(pop_size)
        
        if ranks[i] < ranks[j]:
            return i
        elif ranks[j] < ranks[i]:
            return j
        else:
            # Same rank, compare crowding distance (higher is better)
            if cd[i] > cd[j]:
                return i
            else:
                return j

    def _sbx_crossover(self, p1, p2, lo, hi):
        c1 = np.copy(p1)
        c2 = np.copy(p2)
        
        eta_c = 15
        prob_c = 0.9
        
        for i in range(len(p1)):
            if np.random.random() < prob_c:
                if p1[i] == p2[i]:
                    continue
                
                # Calculate beta
                if np.random.random() < 0.5:
                    beta = (2 * (min(p1[i], p2[i]) - lo[i]) / (p1[i] - p2[i])) ** (1 / (eta_c + 1))
                else:
                    beta = (2 * (hi[i] - max(p1[i], p2[i])) / (p2[i] - p1[i])) ** (1 / (eta_c + 1))
                
                c1[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
                c2[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])
                
                # Clip
                c1[i] = np.clip(c1[i], lo[i], hi[i])
                c2[i] = np.clip(c2[i], lo[i], hi[i])
        return c1, c2

    def _polynomial_mutation(self, x, lo, hi):
        eta_m = 20
        prob_m = 1.0 / self.dim
        
        for i in range(len(x)):
            if np.random.random() < prob_m:
                y = x[i]
                if y == lo[i] or y == hi[i]:
                    continue
                    
                delta1 = (y - lo[i]) / (hi[i] - lo[i])
                delta2 = (hi[i] - y) / (hi[i] - lo[i])
                
                r = np.random.random()
                if r < 0.5:
                    xy = (2 * r + (1 - 2 * r) * (delta1 ** (eta_m + 1))) ** (1 / (eta_m + 1))
                    delta_q = (xy - 1) * (hi[i] - lo[i])
                else:
                    xy = (2 * (1 - r) + (2 * (r - 0.5)) * (delta2 ** (eta_m + 1))) ** (1 / (eta_m + 1))
                    delta_q = (1 - xy) * (hi[i] - lo[i])
                    
                x[i] = y + delta_q
                x[i] = np.clip(x[i], lo[i], hi[i])
        return x
