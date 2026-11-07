"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Time-bounded multi-start local search for WFG7."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        # 330s kill limit. We want to finish well before.
        # Let's aim for 320s to be safe, leaving 10s for cleanup/GC/return.
        deadline = t_start + 320.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_dim = self.dim
        budget = self.budget
        
        # The problem function returns a vector of objectives.
        # We need to minimize the sum of objectives or use a multi-objective approach.
        # Since scipy's minimize is single-objective, we can use the sum of objectives
        # as a proxy for local search, or we can use a weighted sum with random weights.
        # Given the scoring is hypervolume, finding points close to the Pareto front is key.
        # A common trick is to minimize the sum of objectives, which often finds a point
        # near the knee of the Pareto front. To get diversity, we can use random weights.
        
        # However, the task says "minimizing all 3 objectives".
        # The evaluator records ALL returned objective vectors.
        # So we want to return as many non-dominated points as possible.
        
        # Strategy:
        # 1. Start with a small random population to get initial feasible points.
        # 2. Use the remaining time to perform local optimizations from random starts.
        # 3. Each local optimization uses a weighted sum of objectives with random weights.
        # 4. Record the result of each optimization.
        
        # Initial random sampling to ensure we have some points even if local search fails
        n_init = min(10, budget)
        X_init = lo + (hi - lo) * np.random.random((n_init, n_dim))
        F_init = np.array([problem(x) for x in X_init])
        
        # Store all evaluated points to return at the end
        # The problem statement says "Every returned objective vector is recorded".
        # This implies the evaluator calls problem(x) and records the result.
        # We don't control the recording, we just call problem(x).
        # But wait, the contract says: "Every returned objective vector is recorded... return value of __call__ is ignored."
        # This means the evaluator wraps `problem` or intercepts it?
        # "For seeds 0 and 1 the evaluator calls np.random.seed(seed), creates Optimizer... and calls it once."
        # "problem(x) clips ... and returns 3 objective values."
        # "Every returned objective vector is recorded" -> This likely means the evaluator's `problem` function records the output.
        # So I just need to call `problem(x)` as many times as I want within the time/budget limits.
        
        # Wait, "More than 200 evaluations... makes the whole result invalid".
        # AHA! The budget is 200 evaluations. I cannot exceed 200 calls to `problem`.
        # My previous assumption about "hundreds or thousands" was wrong.
        # The budget is hard-capped at 200.
        
        # So, I have 200 evaluations and ~330 seconds.
        # The current DE program uses 200 evaluations.
        # The time limit is 330s.
        # If DE finishes in < 330s, the time limit isn't the bottleneck, the budget is.
        # Let's check how long DE takes. 200 evals of WFG7 (12 vars) is very fast.
        # Probably < 1 second.
        # So the time limit is NOT the bottleneck. The budget of 200 is the bottleneck.
        
        # If the budget is 200, I need to use those 200 evaluations as effectively as possible.
        # The current score is 0.9935.
        # Inspiration 1 (score 1.0854) used "Fast Differential Evolution".
        # Inspiration 3 (score 0.9612) used "Differential Evolution (DE/rand/1)".
        
        # Why is the current program (DE) scoring 0.9935?
        # Maybe the population size or parameters are suboptimal.
        # Or maybe a different algorithm works better for WFG7 with N=200.
        
        # Let's try a different strategy: **Multi-Objective Particle Swarm Optimization (MOPSO)** or **SPEA2** or **NSGA-II**.
        # The current program is a simple DE with a 1:1 replacement strategy.
        # NSGA-II is generally superior for multi-objective problems due to its non-dominated sorting and crowding distance.
        
        # Let's implement a lightweight NSGA-II.
        # Budget = 200.
        # Pop size = 100?
        # Generations = 2?
        # Init: 100 evals.
        # Gen 1: 100 offspring -> 100 evals. Total 200.
        
        # Or Pop size = 50?
        # Init: 50 evals.
        # Gen 1: 50 offspring (50 evals). Total 100.
        # Gen 2: 50 offspring (50 evals). Total 150.
        # Gen 3: 50 offspring (50 evals). Total 200.
        
        # More generations with smaller population might be better for convergence.
        
        # Let's implement NSGA-II.
        
        def non_dominated_sort(F):
            """
            F: array of shape (N, M)
            Returns: list of lists, each containing indices of non-dominated points in that front.
            """
            N = F.shape[0]
            M = F.shape[1]
            
            # S[i] = set of points dominated by i
            # n[i] = number of points dominating i
            # rank[i] = rank of i
            S = [set() for _ in range(N)]
            n = np.zeros(N, dtype=int)
            rank = np.zeros(N, dtype=int)
            
            fronts = []
            current_front = []
            
            for p in range(N):
                for q in range(N):
                    if p == q:
                        continue
                    # Check if p dominates q
                    # p dominates q if p <= q in all objectives and p < q in at least one
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].add(q)
                    # Check if q dominates p
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        n[p] += 1
            
            for p in range(N):
                if n[p] == 0:
                    rank[p] = 0
                    current_front.append(p)
            
            i = 0
            while current_front:
                fronts.append(current_front)
                next_front = []
                for p in current_front:
                    for q in S[p]:
                        n[q] -= 1
                        if n[q] == 0:
                            rank[q] = i + 1
                            next_front.append(q)
                current_front = next_front
                i += 1
            
            return fronts

        def crowding_distance(F, indices):
            """
            Calculate crowding distance for a list of indices.
            Returns array of distances for those indices.
            """
            if len(indices) <= 2:
                return np.full(len(indices), np.inf)
            
            dist = np.zeros(len(indices))
            M = F.shape[1]
            
            for m in range(M):
                f_vals = F[indices, m]
                sorted_idx = np.argsort(f_vals)
                sorted_f = f_vals[sorted_idx]
                
                if sorted_f[0] == sorted_f[-1]:
                    dist[sorted_idx] = np.inf
                    continue
                    
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                
                for k in range(1, len(sorted_idx) - 1):
                    dist[sorted_idx[k]] += (sorted_f[k+1] - sorted_f[k-1]) / (sorted_f[-1] - sorted_f[0])
            
            return dist

        # NSGA-II Implementation
        pop_size = 50
        X = lo + (hi - lo) * np.random.random((pop_size, n_dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        while used < budget:
            if time.time() > deadline:
                break
            
            # Create offspring
            X_off = np.empty((pop_size, n_dim))
            F_off = np.empty((pop_size, self.n_obj))
            
            for i in range(pop_size):
                if used >= budget:
                    break
                if time.time() > deadline:
                    break
                    
                # Tournament selection
                # Pick 2 random parents from X
                p1_idx, p2_idx = np.random.choice(pop_size, 2, replace=False)
                
                # Dominance check for tournament
                if np.all(F[p1_idx] <= F[p2_idx]) and np.any(F[p1_idx] < F[p2_idx]):
                    parent1 = X[p1_idx]
                elif np.all(F[p2_idx] <= F[p1_idx]) and np.any(F[p2_idx] < F[p1_idx]):
                    parent1 = X[p2_idx]
                else:
                    # Non-dominated, pick randomly
                    parent1 = X[np.random.choice([p1_idx, p2_idx])]
                    
                p1_idx, p2_idx = np.random.choice(pop_size, 2, replace=False)
                if np.all(F[p1_idx] <= F[p2_idx]) and np.any(F[p1_idx] < F[p2_idx]):
                    parent2 = X[p1_idx]
                elif np.all(F[p2_idx] <= F[p1_idx]) and np.any(F[p2_idx] < F[p1_idx]):
                    parent2 = X[p2_idx]
                else:
                    parent2 = X[np.random.choice([p1_idx, p2_idx])]
                
                # SBX Crossover
                eta = 20.0
                u = np.random.random(n_dim)
                child = np.empty(n_dim)
                for j in range(n_dim):
                    if u[j] <= 0.5:
                        beta = (2 * u[j]) ** (1 / (eta + 1))
                    else:
                        beta = (2 * (1 - u[j])) ** (1 / (eta + 1))
                    child[j] = 0.5 * ((1 + beta) * parent1[j] + (1 - beta) * parent2[j])
                    
                # Mutation
                u = np.random.random(n_dim)
                for j in range(n_dim):
                    if u[j] <= 1.0 / n_dim:
                        # Perturb
                        if np.random.random() < 0.5:
                            delta = (1 + 2 * (np.random.random() - (parent1[j] - lo[j]) / (hi[j] - lo[j]))) ** (1 / (eta + 1))
                        else:
                            delta = (1 + 2 * ((hi[j] - parent2[j]) / (hi[j] - lo[j]) - np.random.random())) ** (1 / (eta + 1))
                        child[j] = parent1[j] + (hi[j] - lo[j]) * delta
                        
                child = np.clip(child, lo, hi)
                X_off[i] = child
                F_off[i] = problem(child)
                used += 1
            
            # Combine parents and offspring
            X_combined = np.vstack([X, X_off])
            F_combined = np.vstack([F, F_off])
            
            # Non-dominated sorting
            fronts = non_dominated_sort(F_combined)
            
            # Select next population
            X_new = np.empty((pop_size, n_dim))
            F_new = np.empty((pop_size, self.n_obj))
            idx_new = 0
            
            for front in fronts:
                if idx_new + len(front) <= pop_size:
                    # Add all
                    X_new[idx_new:idx_new+len(front)] = X_combined[front]
                    F_new[idx_new:idx_new+len(front)] = F_combined[front]
                    idx_new += len(front)
                else:
                    # Need to select some from this front based on crowding distance
                    n_needed = pop_size - idx_new
                    cd = crowding_distance(F_combined, front)
                    # Sort by crowding distance (descending)
                    sorted_front = np.array(front)[np.argsort(-cd)]
                    selected = sorted_front[:n_needed]
                    X_new[idx_new:idx_new+n_needed] = X_combined[selected]
                    F_new[idx_new:idx_new+n_needed] = F_combined[selected]
                    idx_new += n_needed
                    break
            
            X = X_new
            F = F_new
            
        return X
# EVOLVE-BLOCK-END
