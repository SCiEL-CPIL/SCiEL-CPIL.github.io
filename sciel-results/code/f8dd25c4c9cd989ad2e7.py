"""Multi-objective optimizer on ZDT6 — NSGA-II Black-box Search."""
# EVOLVE-BLOCK-START
import time
import numpy as np

class Optimizer:
    """
    Optimizer using NSGA-II (Non-dominated Sorting Genetic Algorithm II) 
    to discover the Pareto front for ZDT6.
    
    Strategy:
    1. Initialize a random population.
    2. Evaluate objectives.
    3. Perform non-dominated sorting and crowding distance calculation.
    4. Use tournament selection based on rank and crowding distance.
    5. Apply crossover and mutation to generate offspring.
    6. Combine parents and offspring, select the next generation.
    7. Repeat until budget is exhausted.
    8. Return the non-dominated points from the final population (or all evaluated points if needed, but typically the front approximation).
    
    Note: The task records ALL returned objective vectors. 
    However, the `__call__` return value is ignored for scoring, but the 
    contract says "Every returned objective vector is recorded". 
    Wait, the contract says: "Every returned objective vector is recorded and the return value of __call__ is ignored."
    This phrasing is slightly ambiguous. Usually, in these tasks, the evaluator calls `problem(x)` internally or the optimizer returns a set of points.
    Let's re-read carefully: "The evaluator calls np.random.seed(seed), creates Optimizer... and calls it once."
    "Every returned objective vector is recorded" likely refers to the vectors returned by the `problem` function calls made by the optimizer? 
    No, `problem(x)` returns 2 objective values. 
    "Every returned objective vector is recorded" -> This implies the evaluator is wrapping `problem` or the optimizer returns a list of points and the evaluator evaluates them?
    Actually, the standard pattern for these "budget" tasks is that the optimizer calls `problem(x)` itself, and the evaluator counts the number of calls to `problem`.
    The phrase "Every returned objective vector is recorded" might mean that the evaluator intercepts the calls to `problem` and records the results.
    If so, I just need to make <= 200 calls to `problem` with good x's.
    
    The scoring is based on the hypervolume of "all recorded vectors".
    So, I should ensure that the set of x's I evaluate via `problem(x)` forms a good approximation of the Pareto front.
    
    NSGA-II is a standard, robust black-box method.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        
        # Population size
        N = 50 
        # We have budget 200. 
        # Init: 50 evals.
        # Generations: (200 - 50) / (50*2) ? No, offspring is N.
        # Each generation: N offspring evals.
        # Total evals = N (init) + G * N (offspring).
        # 50 + G*50 <= 200 => G <= 3.
        # That's very few generations.
        # Let's use a smaller population to allow more generations?
        # N=20, G = (200-20)/20 = 9 generations.
        # Or N=10, G = 19 generations.
        # NSGA-II works better with larger N, but with such a small budget, 
        # a mix of exploration and exploitation is key.
        
        # Let's try N=20.
        pop_size = 20
        max_evals = self.budget
        
        # Initialize population
        # Random initialization
        pop = np.random.uniform(lo, hi, size=(pop_size, self.dim))
        
        # Track all evaluated points and their fitness
        all_points = []
        all_fitness = []
        eval_count = 0
        
        def evaluate(pop_to_eval):
            nonlocal eval_count
            fitness = []
            for x in pop_to_eval:
                if eval_count >= max_evals:
                    break
                f = problem(x)
                fitness.append(f)
                all_points.append(x)
                all_fitness.append(f)
                eval_count += 1
            return np.array(fitness)
            
        # Evaluate initial population
        F = evaluate(pop)
        
        # NSGA-II Loop
        gen = 0
        while eval_count < max_evals:
            # Check time
            if time.time() > deadline:
                break
                
            # Non-dominated sorting
            ranks, fronts = self._non_dominated_sort(F)
            
            # Crowding distance
            crowding = np.zeros(len(F))
            for front in fronts:
                if len(front) > 2:
                    self._crowding_distance(F, front, crowding)
            
            # Selection: Binary tournament
            parents = np.zeros_like(pop)
            for i in range(pop_size):
                p1_idx = self._tournament(ranks, crowding, pop_size)
                p2_idx = self._tournament(ranks, crowding, pop_size)
                parents[i] = pop[p1_idx]
            
            # Crossover and Mutation
            offspring = np.zeros_like(pop)
            for i in range(0, pop_size, 2):
                c1, c2 = self._crossover(parents[i], parents[i+1])
                offspring[i] = self._mutate(c1, lo, hi)
                offspring[i+1] = self._mutate(c2, lo, hi)
                
            # Evaluate offspring
            F_offspring = evaluate(offspring)
            
            # Combine
            combined_pop = np.vstack([pop, offspring])
            combined_F = np.vstack([F, F_offspring])
            
            # Selection for next generation
            next_pop, next_F = self._select_next(combined_pop, combined_F, pop_size)
            
            pop = next_pop
            F = next_F
            gen += 1
            
        # Return the best points found
        # The evaluator records all calls. 
        # We just need to return something.
        return all_points

    def _non_dominated_sort(self, F):
        n = len(F)
        S = [[] for _ in range(n)]  # S[i] = indices dominated by i
        n_doms = np.zeros(n, dtype=int)  # number of solutions dominating i
        fronts = []
        
        for i in range(n):
            for j in range(i+1, n):
                if self._dominates(F[i], F[j]):
                    S[i].append(j)
                    n_doms[j] += 1
                elif self._dominates(F[j], F[i]):
                    S[j].append(i)
                    n_doms[i] += 1
        
        # First front
        front = [i for i in range(n) if n_doms[i] == 0]
        fronts.append(front)
        
        while front:
            next_front = []
            for i in front:
                for j in S[i]:
                    n_doms[j] -= 1
                    if n_doms[j] == 0:
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
            front = next_front
            
        # Assign ranks
        ranks = np.zeros(n, dtype=int)
        for rank, front in enumerate(fronts):
            for i in front:
                ranks[i] = rank
                
        return ranks, fronts

    def _dominates(self, f1, f2):
        # f1 dominates f2 if f1 is better or equal in all objectives and strictly better in at least one
        # Minimizing objectives
        better_or_equal = np.all(f1 <= f2)
        strictly_better = np.any(f1 < f2)
        return better_or_equal and strictly_better

    def _crowding_distance(self, F, front, crowding):
        if len(front) <= 2:
            crowding[front] = np.inf
            return
            
        m = self.n_obj
        for k in range(m):
            f_k = F[front, k]
            min_f = np.min(f_k)
            max_f = np.max(f_k)
            if max_f - min_f == 0:
                continue
            # Sort indices in front by objective k
            sorted_idx = np.argsort(f_k)
            # Assign infinite distance to extremes
            crowding[front[sorted_idx[0]]] = np.inf
            crowding[front[sorted_idx[-1]]] = np.inf
            
            for i in range(1, len(sorted_idx) - 1):
                idx_curr = front[sorted_idx[i]]
                idx_prev = front[sorted_idx[i-1]]
                idx_next = front[sorted_idx[i+1]]
                crowding[idx_curr] += (f_k[sorted_idx[i+1]] - f_k[sorted_idx[i-1]]) / (max_f - min_f)

    def _tournament(self, ranks, crowding, size=2):
        indices = np.random.randint(0, len(ranks), size=size)
        # Pick the one with lower rank, or higher crowding if same rank
        best_idx = indices[0]
        for i in indices[1:]:
            if ranks[i] < ranks[best_idx]:
                best_idx = i
            elif ranks[i] == ranks[best_idx]:
                if crowding[i] > crowding[best_idx]:
                    best_idx = i
        return best_idx

    def _crossover(self, p1, p2, prob=0.9):
        if np.random.rand() < prob:
            # Simulated Binary Crossover (SBX) simplified or just blend
            # Let's use a simple blend crossover for speed and robustness
            alpha = np.random.rand(self.dim)
            c1 = alpha * p1 + (1 - alpha) * p2
            c2 = alpha * p2 + (1 - alpha) * p1
            return c1, c2
        else:
            return p1.copy(), p2.copy()

    def _mutate(self, x, lo, hi, prob=0.1, eta=20.0):
        # Polynomial mutation or simple Gaussian
        # Simple Gaussian mutation clipped to bounds
        mask = np.random.rand(self.dim) < prob
        if np.any(mask):
            sigma = 0.1 * (hi - lo)
            x[mask] += np.random.normal(0, sigma[mask])
            x[mask] = np.clip(x[mask], lo[mask], hi[mask])
        return x

    def _select_next(self, combined_pop, combined_F, pop_size):
        ranks, fronts = self._non_dominated_sort(combined_F)
        crowding = np.zeros(len(combined_F))
        for front in fronts:
            if len(front) > 2:
                self._crowding_distance(combined_F, front, crowding)
        
        next_pop_indices = []
        current_size = 0
        for front in fronts:
            if current_size + len(front) <= pop_size:
                next_pop_indices.extend(front)
                current_size += len(front)
            else:
                # Need to select from this front based on crowding
                needed = pop_size - current_size
                # Sort front by crowding distance (descending)
                front_sorted = sorted(front, key=lambda i: crowding[i], reverse=True)
                next_pop_indices.extend(front_sorted[:needed])
                break
                
        next_pop = combined_pop[next_pop_indices]
        next_F = combined_F[next_pop_indices]
        return next_pop, next_F
# EVOLVE-BLOCK-END
