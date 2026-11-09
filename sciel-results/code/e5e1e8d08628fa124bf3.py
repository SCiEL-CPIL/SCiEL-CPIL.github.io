import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Use global random state as per evaluator seeding
        # But for reproducibility within the run if needed, we rely on the seed set by evaluator.
        # However, using np.random directly is safer to align with "np.random.seed(seed)".
        
        all_F = []
        all_X = []
        
        def evaluate(x):
            x = np.clip(x, lo, hi)
            f = np.array(problem(x))
            all_F.append(f)
            all_X.append(x)
            return f
        
        # 1. Initial Population: Random + Grid
        # WFG9 is separable-ish but complex. A grid helps cover the front initially.
        # Budget 200. Let's use 20 random points and 180 for the main search.
        
        # Actually, let's try a different strategy:
        # Reference Point: [3.0, 5.0, 7.0]
        # WFG9 Pareto front is known. 
        # Strategy: Use a small population size (e.g., 10-20) and run many generations.
        # Or, use a specific multi-objective algorithm like NSGA-II with crowding distance.
        # The current program is a simple (1+1) EA or similar with NDS selection.
        # Let's try NSGA-II.
        
        pop_size = 20
        # Initialize population
        for _ in range(pop_size):
            x = lo + np.random.random(dim) * (hi - lo)
            evaluate(x)
        
        # Main loop
        # We have budget - pop_size evaluations left.
        # NSGA-II typically requires a mating pool.
        
        # Let's implement a simplified NSGA-II or SPEA2.
        # Given the small budget (200), overhead of sorting and non-dominated sorting should be minimal.
        
        def non_dominated_sort(F):
            n = len(F)
            if n == 0: return []
            S = [[] for _ in range(n)]  # S[i] = indices j dominated by i
            n_count = np.zeros(n, dtype=int)  # Number of points dominating i
            rank = np.zeros(n, dtype=int)
            
            for i in range(n):
                for j in range(i+1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_count[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_count[i] += 1
            
            front = [i for i in range(n) if n_count[i] == 0]
            if not front: return [[]] # Should not happen
            
            k = 0
            current_front = front
            fronts = []
            
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_count[j] -= 1
                        if n_count[j] == 0:
                            next_front.append(j)
                current_front = next_front
                k += 1
            return fronts
        
        def crowding_distance(F, indices):
            if len(indices) <= 2:
                return np.full(len(indices), np.inf)
            
            m = len(indices)
            dists = np.zeros(m)
            F_sub = F[indices]
            
            for k in range(n_obj):
                order = np.argsort(F_sub[:, k])
                dists[order[0]] = np.inf
                dists[order[-1]] = np.inf
                
                for j in range(1, m-1):
                    f_curr = F_sub[order[j], k]
                    f_next = F_sub[order[j+1], k]
                    f_prev = F_sub[order[j-1], k]
                    
                    f_min = F_sub[order[0], k]
                    f_max = F_sub[order[-1], k]
                    if f_max > f_min:
                        dists[order[j]] += (f_next - f_prev) / (f_max - f_min)
                    else:
                        dists[order[j]] += 0.0
            return dists

        # Main Loop
        while len(all_F) < budget:
            if time.time() > deadline:
                break
            
            # Combine current population and offspring? 
            # In (mu+lambda) or NSGA-II, we maintain a population.
            # Let's maintain a population of size pop_size.
            # Currently all_F contains history. We need to select parents from the current NDS or population.
            
            # For simplicity and speed, let's just select from the current Non-Dominated Set of all evaluated points.
            # This is a (1+1) EA with NDS archive.
            # To make it more robust, let's try a small population based approach.
            
            # Let's restart with a proper population-based EA.
            # We have already evaluated pop_size points.
            # Let's treat the first pop_size points as the initial population P.
            
            # If we are at the start, we have P.
            # Let's create a loop that generates offspring Q, combines P+Q, and selects P.
            
            # But the structure above appends to all_F. 
            # Let's restructure to a standard NSGA-II loop.
            
            # Since we already did initial eval, let's just proceed with (1+1) for the rest to keep it simple, 
            # OR implement a proper loop.
            
            # Let's implement a simple loop where we pick a parent from the current NDS, mutate, evaluate, and keep if non-dominated.
            # This is what the current program does. 
            # The prompt asks for a DIFFERENT strategy.
            
            # Strategy: Use a grid-based initialization followed by local search (Nelder-Mead or similar) on a few promising points?
            # Or use a differential evolution (DE) based multi-objective algorithm.
            
            # Let's try DE/rand/1/bin.
            # Population size 20.
            # Initialize P.
            # Loop:
            #   Create Q from P using DE.
            #   Evaluate Q.
            #   Combine P and Q.
            #   Select best P from combined set using NDS + Crowding.
            
            # We have 20 points in all_F. Let's extract them as P.
            P_X = np.array(all_X[:pop_size])
            P_F = np.array(all_F[:pop_size])
            
            # Clear all_F to only store final results? 
            # "Every returned objective vector is recorded". 
            # The evaluator records the return value of __call__? No, "Every returned objective vector is recorded".
            # Wait, "problem(x) ... returns 3 objective values". 
            # "Every returned objective vector is recorded" likely means every call to problem(x) is recorded.
            # So I must not discard evaluations.
            # But I can keep a separate population structure.
            
            # Let's continue with the (1+1) EA but improve the mutation and selection.
            # The current program uses Gaussian mutation.
            # Let's try Cauchy mutation or larger steps.
            # Or, let's try to exploit the structure of WFG9.
            
            # WFG9 is separable. 
            # Let's try a coordinate descent or axis-aligned search.
            
            # Actually, let's stick to a robust global optimizer.
            # Let's try using `scipy.optimize.differential_evolution`? 
            # It's single objective. We need multi-objective.
            
            # Let's implement a simple Multi-Objective DE.
            
            # Reset population to the first pop_size points.
            # We have 20 points.
            
            # Let's run generations.
            # Each generation: 20 offspring.
            # Total budget 200. 20 initial + 180 offspring = 9 generations.
            
            # Extract current P
            P_X = np.array(all_X[-pop_size:]) if len(all_X) >= pop_size else np.array(all_X)
            P_F = np.array(all_F[-pop_size:]) if len(all_F) >= pop_size else np.array(all_F)
            
            # If we haven't reached pop_size, fill with random.
            # We already did 20 random points. So P is ready.
            
            for gen in range(9): # 9 * 20 = 180
                if time.time() > deadline:
                    break
                
                Q_X = []
                Q_F = []
                
                for i in range(pop_size):
                    if time.time() > deadline:
                        break
                    
                    # DE/rand/1/bin
                    r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
                    CR = 0.9
                    F_factor = 0.8
                    
                    v = P_X[r1] + F_factor * (P_X[r2] - P_X[r3])
                    
                    # Crossover
                    j_rand = np.random.randint(dim)
                    trial = P_X[i].copy()
                    for j in range(dim):
                        if np.random.random() < CR or j == j_rand:
                            trial[j] = v[j]
                        
                    trial = np.clip(trial, lo, hi)
                    Q_X.append(trial)
                    f = evaluate(trial)
                    Q_F.append(f)
                
                if len(Q_F) == 0:
                    break
                    
                # Combine P and Q
                C_X = np.vstack([P_X, np.array(Q_X)])
                C_F = np.vstack([P_F, np.array(Q_F)])
                
                # Non-dominated sort
                fronts = non_dominated_sort(C_F)
                
                # Select new P
                new_P_idx = []
                for front in fronts:
                    if len(new_P_idx) + len(front) <= pop_size:
                        new_P_idx.extend(front)
                    else:
                        # Need to select from this front based on crowding
                        needed = pop_size - len(new_P_idx)
                        dists = crowding_distance(C_F, front)
                        # Sort by distance descending
                        order = np.argsort(dists)[::-1]
                        new_P_idx.extend(front[order[:needed]])
                        break
                
                P_X = C_X[new_P_idx]
                P_F = C_F[new_P_idx]
                
                # Update all_F/X is already done in evaluate()
            
            return None
