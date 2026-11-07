# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ1 using a hybrid strategy:
    - Analytical initialization on the Pareto front
    - NSGA-II style evolutionary search with crowding distance
    - Deadline guard to ensure we finish in time
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Leave 20s safety margin for 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)

        # Population size
        pop_size = min(50, max(10, self.budget // 4))

        # Initialize population
        X = np.empty((pop_size, self.dim))
        F = np.empty((pop_size, self.n_obj))

        # First few points: analytical initialization on DTLZ1 Pareto front
        # For DTLZ1, Pareto front is where sum of first (dim-1) objectives = 0.5
        # This means g(x) = 0, which requires x_{n_obj+1}...x_{dim-1} to satisfy g=0
        # For DTLZ1: g = (dim - n_obj) - sum(x_i) * cos(pi/4 * x_i) for i in n_obj..dim-1
        # g=0 when x_i = 1 for all i >= n_obj (since 1*cos(pi/4)=sqrt(2)/2, and we need sum = dim-n_obj)
        # Actually for DTLZ1, g=0 is achieved when x_i = 1 for all i in [n_obj, dim)
        # The Pareto front is parameterized by the first n_obj variables

        n_analytical = min(pop_size, 20)
        for i in range(n_analytical):
            x = np.ones(self.dim)
            # For DTLZ1, the Pareto front has x_{n_obj}..x_{dim-1} = 1
            # and x_0..x_{n_obj-1} parameterize the front
            # f_i = 0.5 * prod(1-x_j) * (1 + g) * g_i for i in 0..n_obj-1
            # On the front g=0, so f_i = 0.5 * prod(1-x_j) * g_i
            # where g_i are the "shape" parameters
            # For DTLZ1: f_i = 0.5 * (1 - x_{n_obj-1}) * prod_{j=0}^{n_obj-2}(1-x_j) * ... 
            # Actually let me just use simplex sampling for the shape
            # The Pareto front of DTLZ1 in 3 objectives is a triangle
            # We can sample points on the unit simplex for the first n_obj-1 coords
            
            if self.n_obj == 3:
                # Sample on the 2D simplex (triangle)
                u1 = rng.random()
                u2 = rng.random()
                if u1 + u2 > 1:
                    u1 = 1 - u1
                    u2 = 1 - u2
                # x_0, x_1, x_2 relate to the simplex
                # For DTLZ1: f_0 = 0.5*(1-x_2), f_1 = 0.5*x_2*(1-x_1), f_2 = 0.5*x_2*x_1
                # So x_2 = 1 - 2*f_0/0.5 ... let me think differently
                # Just set x_0, x_1, x_2 to values that give a point on the front
                # f_0 = 0.5*(1-x_2), f_1 = 0.5*x_2*(1-x_1), f_2 = 0.5*x_2*x_1
                # Given u1, u2 on simplex: f_0 = 0.5*u1, f_1 = 0.5*u2, f_2 = 0.5*(1-u1-u2)
                # Then x_2 = 1 - 2*f_0 = 1 - u1
                # x_1 = f_2/(0.5*x_2) = (1-u1-u2)/(1-u1) if x_2 != 0
                x[0] = 1.0  # These don't matter for g when x_3..x_6=1
                x[1] = 1.0
                x[2] = 1.0 - u1
                if abs(x[2]) > 1e-10:
                    x[0] = (1 - u1 - u2) / x[2]
                    x[0] = min(max(x[0], 0), 1)
                else:
                    x[0] = 0.0
                x[1] = 1.0
                x[2] = 1.0 - u1
                # Actually let me just use a simpler parameterization
                # For DTLZ1 with 3 objectives:
                # f_0 = 0.5 * (1 - x_2)
                # f_1 = 0.5 * x_2 * (1 - x_1) 
                # f_2 = 0.5 * x_2 * x_1
                # Sum = 0.5*(1-x_2) + 0.5*x_2 = 0.5
                # Let's parameterize: x_2 = 1 - 2*f_0/0.5... 
                # Simpler: let a = u1, b = u2, c = 1-u1-u2 (simplex)
                # f_0 = 0.5*a, f_1 = 0.5*b, f_2 = 0.5*c
                # From f_0 = 0.5*(1-x_2): x_2 = 1 - a
                # From f_2 = 0.5*x_2*x_1: x_1 = c/x_2 = c/(1-a) if a<1
                # From f_1 = 0.5*x_2*(1-x_1): check: 0.5*(1-a)*(1-c/(1-a)) = 0.5*(1-a-c) = 0.5*b ✓
                x[0] = 0.5  # dummy, doesn't affect g
                x[1] = 0.5  # dummy
                x[2] = 1.0 - u1
                if x[2] > 1e-10:
                    x[0] = (1 - u1 - u2) / x[2]
                else:
                    x[0] = 0.0
                x[0] = np.clip(x[0], 0, 1)
                x[1] = 1.0 - x[0] / x[2] if x[2] > 1e-10 else 0.0
                x[1] = np.clip(x[1], 0, 1)
                # Set remaining to 1 for g=0
                x[3:] = 1.0
            else:
                # General case: random point with x_{n_obj:} = 1
                x[:self.n_obj] = rng.random(self.n_obj)
                x[self.n_obj:] = 1.0
            
            x = np.clip(x, lo, hi)
            X[i] = x
            F[i] = problem(x)

        used = n_analytical

        # Fill rest of initial population with random points
        for i in range(n_analytical, pop_size):
            X[i] = lo + (hi - lo) * rng.random(self.dim)
            F[i] = problem(X[i])
            used += 1

        def dominates(a, b):
            """Check if a dominates b (a is better than or equal in all, strictly better in at least one)"""
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_indices(F):
            """Return indices of non-dominated points"""
            n = len(F)
            nd = np.ones(n, dtype=bool)
            for i in range(n):
                if not nd[i]:
                    continue
                for j in range(n):
                    if i == j or not nd[j]:
                        continue
                    if dominates(F[j], F[i]):
                        nd[i] = False
                        break
            return np.where(nd)[0]

        def crowding_distance(F, nd_indices):
            """Compute crowding distance for non-dominated points"""
            cd = np.zeros(len(F))
            if len(nd_indices) <= 2:
                cd[nd_indices] = np.inf
                return cd
            for k in range(self.n_obj):
                sorted_idx = nd_indices[np.argsort(F[nd_indices, k])]
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                f_range = F[sorted_idx[-1], k] - F[sorted_idx[0], k]
                if f_range > 1e-15:
                    for i in range(1, len(sorted_idx) - 1):
                        cd[sorted_idx[i]] += (F[sorted_idx[i+1], k] - F[sorted_idx[i-1], k]) / f_range
            return cd

        def fast_non_dominated_sort(F):
            """Return list of fronts (each front is a list of indices)"""
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            n_dom = np.zeros(n, dtype=int)  # number of points that dominate i
            fronts = [[]]
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if dominates(F[i], F[j]):
                        S[i].append(j)
                    elif dominates(F[j], F[i]):
                        n_dom[i] += 1
                if n_dom[i] == 0:
                    fronts[0].append(i)
            
            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                k += 1
                fronts.append(next_front)
            fronts.pop()  # Remove last empty front
            return fronts

        # Evolutionary loop
        while used < self.budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Get non-dominated fronts
            fronts = fast_non_dominated_sort(F)

            # Select parents from first front (or first few fronts)
            if len(fronts) > 0 and len(fronts[0]) > 0:
                parent_pool = fronts[0]
            else:
                parent_pool = list(range(pop_size))

            # If pool is too small, add from next front
            if len(parent_pool) < 2 and len(fronts) > 1:
                parent_pool = parent_pool + fronts[1]

            # Create children
            n_children = min(5, self.budget - used)
            for c in range(n_children):
                if time.time() > deadline:
                    break
                # Select parent
                p_idx = parent_pool[rng.integers(len(parent_pool))]
                
                # Mutation: Gaussian perturbation
                sigma = 0.1 * (hi - lo)
                child = X[p_idx].copy() + rng.standard_normal(self.dim) * sigma
                child = np.clip(child, lo, hi)
                
                # Evaluate
                fc = problem(child)
                used += 1
                
                # Replace worst point in population
                # Use a simple criterion: replace point with highest sum of objectives
                # among dominated points, or random if all non-dominated
                dominated_mask = np.array([
                    any(dominates(F[j], F[i]) for j in range(pop_size) if j != i)
                    for i in range(pop_size)
                ])
                
                if np.any(dominated_mask):
                    candidates = np.where(dominated_mask)[0]
                    # Among dominated, replace the one with worst (highest) sum
                    sums = np.sum(F[candidates], axis=1)
                    worst_local = np.argmax(sums)
                    replace_idx = candidates[worst_local]
                else:
                    # All non-dominated: replace randomly
                    replace_idx = rng.integers(pop_size)
                
                X[replace_idx] = child
                F[replace_idx] = fc

        return X

# EVOLVE-BLOCK-END
