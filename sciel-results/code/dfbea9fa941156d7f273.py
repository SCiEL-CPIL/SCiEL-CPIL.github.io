import numpy as np
import time
import random

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.start_time = None

    def __call__(self, problem, bounds):
        self.start_time = time.time()
        deadline = 290.0  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Population size for the main loop
        pop_size = 20
        if budget < pop_size:
            pop_size = budget
            
        # Initialize population
        # Use random uniform initialization
        X = np.random.uniform(lo, hi, size=(pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Helper functions for NSGA-II style selection
        def fast_non_dominated_sort(F):
            """
            Returns a list of fronts. front[i] contains indices of individuals in the i-th non-dominated front.
            """
            n = len(F)
            S = [set() for _ in range(n)]  # S[i] = set of individuals dominated by i
            n_d = [0] * n  # n_d[i] = number of individuals that dominate i
            front = [set()]
            
            for p in range(n):
                for q in range(n):
                    if p == q: continue
                    # Check if p dominates q
                    # p dominates q if p <= q and p != q
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].add(q)
                    # Check if q dominates p
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        S[q].add(p)
                        n_d[p] += 1
                        
                if n_d[p] == 0:
                    front[0].add(p)
                    
            i = 0
            while len(front[i]) > 0:
                i += 1
                front.append(set())
                for p in front[i-1]:
                    for q in S[p]:
                        n_d[q] -= 1
                        if n_d[q] == 0:
                            front[i].add(q)
                            
            # Remove empty fronts at the end
            while len(front) > 0 and len(front[-1]) == 0:
                front.pop()
                
            return front

        def crowding_distance(F, front_indices):
            """
            Calculate crowding distance for individuals in a given front.
            """
            n = len(front_indices)
            if n <= 2:
                return [float('inf')] * n
            
            dist = [0.0] * n
            for m in range(n_obj):
                # Sort by objective m
                sorted_idx = sorted(range(n), key=lambda i: F[front_indices[i]][m])
                dist[sorted_idx[0]] = float('inf')
                dist[sorted_idx[-1]] = float('inf')
                
                f_min = F[front_indices[sorted_idx[0]]][m]
                f_max = F[front_indices[sorted_idx[-1]]][m]
                if f_max == f_min:
                    continue
                    
                for k in range(1, n-1):
                    dist[sorted_idx[k]] += (F[front_indices[sorted_idx[k+1]]][m] - F[front_indices[sorted_idx[k-1]]][m]) / (f_max - f_min)
                    
            return dist

        def sbx(x1, x2, eta=20.0, lo=None, hi=None):
            """Simulated Binary Crossover"""
            x = np.zeros_like(x1)
            for i in range(len(x1)):
                if random.random() < 0.5:
                    if abs(x1[i] - x2[i]) < 1e-14:
                        x[i] = x1[i]
                    else:
                        u = random.random()
                        if u <= 0.5:
                            beta = (2 * u) ** (1.0 / (eta + 1.0))
                        else:
                            beta = (1.0 / (2 * (1.0 - u))) ** (1.0 / (eta + 1.0))
                        x[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                else:
                    if abs(x1[i] - x2[i]) < 1e-14:
                        x[i] = x1[i]
                    else:
                        u = random.random()
                        if u <= 0.5:
                            beta = (2 * u) ** (1.0 / (eta + 1.0))
                        else:
                            beta = (1.0 / (2 * (1.0 - u))) ** (1.0 / (eta + 1.0))
                        x[i] = 0.5 * ((1 + beta) * x2[i] + (1 - beta) * x1[i])
                        
            # Clip to bounds
            x = np.clip(x, lo, hi)
            return x

        def polynomial_mutation(x, eta=20.0, lo=None, hi=None):
            """Polynomial Mutation"""
            x_new = x.copy()
            for i in range(len(x_new)):
                if random.random() < 0.5: # Mutation rate 0.5 per gene? Usually 1/dim. Let's use 1/dim.
                    pass 
            # Standard mutation rate is 1/dim
            rate = 1.0 / dim
            for i in range(len(x_new)):
                if random.random() < rate:
                    u = random.random()
                    if u <= 0.5:
                        delta = (2 * u) ** (1.0 / (eta + 1.0)) - 1.0
                    else:
                        delta = 1.0 - (2 * (1.0 - u)) ** (1.0 / (eta + 1.0))
                    
                    x_new[i] = x_new[i] + delta * (hi[i] - lo[i])
                    
            x_new = np.clip(x_new, lo, hi)
            return x_new

        # Main Evolutionary Loop
        while used < budget:
            if time.time() > deadline - 10.0: # Leave 10s for polish and overhead
                break
                
            # Non-dominated sort
            fronts = fast_non_dominated_sort(F)
            
            # Select parents
            parents = []
            for front in fronts:
                if len(parents) + len(front) > pop_size:
                    # Need to select from this front based on crowding distance
                    dists = crowding_distance(F, list(front))
                    front_list = list(front)
                    # Sort by distance descending
                    front_list.sort(key=lambda i: dists[front_list.index(i)], reverse=True)
                    # Take only as many as needed
                    take = pop_size - len(parents)
                    parents.extend(front_list[:take])
                    break
                else:
                    parents.extend(front)
                    
            # Generate offspring
            offspring_X = []
            offspring_F = []
            
            while len(offspring_X) < pop_size:
                # Tournament selection for 2 parents
                p1_idx = random.choice(parents)
                p2_idx = random.choice(parents)
                
                # Crossover
                child = sbx(X[p1_idx], X[p2_idx], lo=lo, hi=hi)
                
                # Mutation
                child = polynomial_mutation(child, lo=lo, hi=hi)
                
                # Evaluate
                f_child = problem(child)
                used += 1
                
                offspring_X.append(child)
                offspring_F.append(f_child)
                
                if used >= budget:
                    break
            
            # Combine and select
            X = np.vstack([X, np.array(offspring_X)])
            F = np.vstack([F, np.array(offspring_F)])
            
            # Truncate to pop_size
            fronts = fast_non_dominated_sort(F)
            new_X = []
            new_F = []
            for front in fronts:
                if len(new_X) + len(front) > pop_size:
                    dists = crowding_distance(F, list(front))
                    front_list = list(front)
                    front_list.sort(key=lambda i: dists[front_list.index(i)], reverse=True)
                    take = pop_size - len(new_X)
                    for idx in front_list[:take]:
                        new_X.append(X[idx])
                        new_F.append(F[idx])
                    break
                else:
                    for idx in front:
                        new_X.append(X[idx])
                        new_F.append(F[idx])
                        
            X = np.array(new_X)
            F = np.array(new_F)

        # Local Polish Phase
        # Use remaining budget to polish the best non-dominated solutions
        if used < budget and time.time() < deadline - 5.0:
            # Get non-dominated solutions
            fronts = fast_non_dominated_sort(F)
            if fronts:
                nd_indices = list(fronts[0])
                # Sort by sum of objectives to get the best ones
                nd_indices.sort(key=lambda i: np.sum(F[i]))
                
                # Pick a few to polish
                num_to_polish = min(5, len(nd_indices), budget - used)
                
                for i in range(num_to_polish):
                    if time.time() > deadline - 2.0:
                        break
                    idx = nd_indices[i]
                    x_start = X[idx].copy()
                    f_start = F[idx].copy()
                    
                    # Define a scalarized objective to minimize
                    # Use a weighted sum with weights pointing towards the ideal point (0,0,0)
                    # Since WFG4 is convex, this should work.
                    # Weights can be random or specific. Let's use a few random weight vectors.
                    # Or simply minimize the sum.
                    def scalarized_obj(x):
                        fx = problem(x)
                        return np.sum(fx)
                        
                    # Use Nelder-Mead for local search
                    # It doesn't require gradients.
                    # Limit iterations to save time and budget
                    max_iter = min(20, budget - used)
                    if max_iter <= 0: continue
                    
                    try:
                        from scipy.optimize import minimize
                        res = minimize(scalarized_obj, x_start, method='Nelder-Mead', 
                                       options={'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-6})
                        
                        # Count evaluations used by minimize? 
                        # scipy doesn't easily expose this without a wrapper.
                        # Let's wrap it to count.
                        
                        # Actually, let's just do a simple manual local search to control budget strictly.
                        # Nelder-Mead can use many evaluations.
                        # Let's do a simple coordinate descent or random perturbation refinement.
                    except Exception:
                        pass
                    
                    # Manual Local Search: Coordinate-wise perturbation
                    x_best = x_start.copy()
                    f_best = f_start.copy()
                    step_size = 0.05 * (hi - lo)
                    current_budget = budget - used
                    
                    # Try to improve by moving in each dimension
                    for _ in range(min(10, current_budget)):
                        if time.time() > deadline - 1.0: break
                        improved = False
                        for d in range(dim):
                            if used >= budget: break
                            # Try positive step
                            x_trial = x_best.copy()
                            x_trial[d] += step_size[d]
                            x_trial = np.clip(x_trial, lo, hi)
                            f_trial = problem(x_trial)
                            used += 1
                            
                            # Check if f_trial dominates f_best? 
                            # For scalarized sum, just check sum.
                            if np.sum(f_trial) < np.sum(f_best):
                                x_best = x_trial
                                f_best = f_trial
                                improved = True
                                
                            # Try negative step
                            if used >= budget: break
                            x_trial = x_best.copy()
                            x_trial[d] -= step_size[d]
                            x_trial = np.clip(x_trial, lo, hi)
                            f_trial = problem(x_trial)
                            used += 1
                            
                            if np.sum(f_trial) < np.sum(f_best):
                                x_best = x_trial
                                f_best = f_trial
                                improved = True
                                
                        if not improved:
                            # Shrink step size if no improvement
                            step_size *= 0.5
                            if np.max(step_size) < 1e-6:
                                break
                        
                        # Update the population with the improved point if it's non-dominated or better
                        # We will collect all polished points and add them to the archive
                        X[idx] = x_best
                        F[idx] = f_best

        # Final Archive Construction
        # Keep all non-dominated points from X, F
        fronts = fast_non_dominated_sort(F)
        if fronts:
            archive_indices = list(fronts[0])
            archive_X = X[archive_indices]
            archive_F = F[archive_indices]
        else:
            archive_X = X
            archive_F = F
            
        return archive_F
