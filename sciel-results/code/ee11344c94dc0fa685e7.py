"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT3 using a hybrid strategy:
    1. Analytical initialization on the Pareto front (x1=0, x2=0)
    2. NSGA-II style evolution with crowding distance
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        # Population size
        pop_size = min(50, max(10, self.budget // 4))
        
        # Initialize with analytical points on the Pareto front of ZDT3
        # For ZDT3: f1 = x1, g = 1 + 9*sum(x2..xn)/n2, h = 1 - sqrt(f1/g)*exp(-4*f1/g*sin^2(10*pi*f1/g))
        # Pareto front: f1=0, g=1 (x1=0, all x_i=0 for i>=2), h=1
        # Actually the Pareto front is where h=1, which requires f1/g -> 0 or specific conditions
        # Let's just use random initialization but with a bias towards low x1
        
        # Generate initial population
        X = np.random.uniform(lo, hi, size=(pop_size, self.dim))
        # Bias x1 towards 0 for better initial coverage
        X[:, 0] = np.random.uniform(0, 0.5, size=pop_size)
        
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Check time
        if time.time() > deadline:
            return X

        # NSGA-II style evolution
        while used < self.budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Fast non-dominated sorting
            fronts = self._fast_non_dominated_sort(F)
            
            # Get non-dominated points (first front)
            front0 = fronts[0]
            
            # Select parents from first front using tournament selection
            parent1 = self._tournament_select(front0, F, rng, k=2)
            parent2 = self._tournament_select(front0, F, rng, k=2)
            
            # Crossover and mutation
            child = self._crossover(parent1, parent2, rng)
            child = self._mutate(child, lo, hi, rng)
            
            # Evaluate child
            fc = problem(child)
            used += 1
            
            # Replace worst individual in population
            # Use a simple criterion: replace the dominated individual with highest sum
            dominated = np.array([np.any(np.all(F <= f, axis=1) & np.any(F < f, axis=1)) for f in F])
            if np.any(dominated):
                candidates = np.where(dominated)[0]
                worst_idx = candidates[np.argmax(np.sum(F[candidates], axis=1))]
            else:
                worst_idx = int(np.argmax(np.sum(F, axis=1)))
            
            X[worst_idx] = child
            F[worst_idx] = fc
        
        return X

    def _fast_non_dominated_sort(self, F):
        """Fast non-dominated sorting. Returns list of fronts (each a list of indices)."""
        n = len(F)
        S = [[] for _ in range(n)]  # Dominance count
        n_dom = np.zeros(n, dtype=int)  # Number of points dominated by i
        fronts = []
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_dom[i] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    n_dom[j] += 1
        
        current_front = [i for i in range(n) if n_dom[i] == 0]
        fronts.append(current_front)
        
        while current_front:
            next_front = []
            for i in current_front:
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
            current_front = next_front
        
        return fronts

    def _tournament_select(self, candidates, F, rng, k=2):
        """Tournament selection among candidate indices."""
        if len(candidates) <= k:
            return candidates[rng.integers(len(candidates))]
        
        selected = rng.choice(candidates, size=k, replace=False)
        best = selected[0]
        for idx in selected[1:]:
            if np.any(F[idx] < F[best]):
                best = idx
        return best

    def _crossover(self, p1, p2, rng):
        """Simulated binary crossover (SBX)."""
        child = p1.copy()
        for i in range(len(p1)):
            if rng.random() < 0.5:
                if abs(p1[i] - p2[i]) > 1e-10:
                    beta = 1.0
                    if rng.random() < 0.5:
                        beta = (0.5 * (p1[i] - p2[i])) ** (1 / 21)
                    else:
                        beta = (0.5 * (p2[i] - p1[i])) ** (1 / 21)
                    child[i] = 0.5 * (p1[i] + p2[i] - beta * (p2[i] - p1[i]))
        return child

    def _mutate(self, x, lo, hi, rng):
        """Polynomial mutation."""
        x_mut = x.copy()
        for i in range(len(x)):
            if rng.random() < 0.5:
                if hi[i] - lo[i] < 1e-10:
                    continue
                u = rng.random()
                if u < 0.5:
                    delta = (2 * u) ** (1 / 21) - 1
                else:
                    delta = 1 - (2 * (1 - u)) ** (1 / 21)
                x_mut[i] = x[i] + delta * (hi[i] - lo[i])
        return np.clip(x_mut, lo, hi)
# EVOLVE-BLOCK-END
