"""Multi-objective optimizer on WFG4 using MOEA/D with systematic weights and local polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """MOEA/D multi-objective optimizer for WFG4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 315.0  # Safety margin for 330s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Use a fixed seed for reproducibility within the optimizer instance
        rng = np.random.default_rng(42)

        # Population size N
        # With budget 200, N=100 allows 100 initial evals and 100 iterative evals.
        N = 100

        # Generate systematic weight vectors for 3 objectives (2D simplex)
        # We want N points on the triangle w1+w2+w3=1, wi >= 0
        # A simple way is to use a triangular grid.
        # For N=100, we can approximate a grid.
        # Alternatively, use random points but ensure good spread via rejection or sorting.
        # Let's use a quasi-random sequence (Halton) or just random normalized for simplicity, 
        # but systematic is better for MOEA/D.
        
        # Generate points on the simplex
        # We can generate random points and normalize, but to ensure coverage, 
        # let's use a grid-like distribution.
        # For 3 objectives, we can define w1, w2 such that w1+w2 <= 1, w3 = 1-w1-w2.
        # Let's create a grid of (w1, w2) and filter/normalize.
        
        # Simpler approach: Use random points but sort and select to ensure spread?
        # Actually, for MOEA/D, the specific weights matter less than the neighborhood structure.
        # Random normalized weights are often sufficient if N is large enough.
        # Let's stick to random normalized weights for robustness against specific grid artifacts.
        
        weights = rng.random((N, n_obj))
        weights /= np.sum(weights, axis=1, keepdims=True)
        
        # Calculate neighborhood based on distance in weight space
        T = 20  # Neighborhood size
        dists = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                dists[i, j] = np.linalg.norm(weights[i] - weights[j])
        
        neighbors = [None] * N
        for i in range(N):
            sorted_idx = np.argsort(dists[i])
            # Exclude self
            sorted_idx = sorted_idx[sorted_idx != i]
            neighbors[i] = sorted_idx[:T]
            
        # Initialize population
        X = lo + (hi - lo) * rng.random((N, dim))
        F = np.array([problem(x) for x in X], dtype=float)
        used = N
        
        # Ideal point (component-wise minimum)
        f_ideal = np.min(F, axis=0)
        
        # MOEA/D parameters
        eta_c = 20.0
        eta_m = 20.0
        p_c = 0.9
        p_m = 1.0 / dim
        
        # Helper: SBX Crossover
        def sbx_crossover(rng, p1, p2, lo, hi, eta):
            u = rng.random(dim)
            beta = np.where(
                u <= 0.5,
                (2 * u) ** (1.0 / (eta + 1)),
                (1 / (2 * (1 - u))) ** (1.0 / (eta + 1))
            )
            c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
            c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
            return np.clip(c1, lo, hi), np.clip(c2, lo, hi)
        
        # Helper: Polynomial Mutation
        def poly_mutation(rng, x, lo, hi, eta, p_m):
            mask = rng.random(dim) < p_m
            if not np.any(mask):
                return x
            u = rng.random(np.sum(mask))
            delta = np.where(
                u <= 0.5,
                (2 * u) ** (1.0 / (eta + 1)) - 1,
                1 - (2 * (1 - u)) ** (1.0 / (eta + 1))
            )
            x_new = x.copy()
            x_new[mask] += delta * (hi[mask] - lo[mask])
            return np.clip(x_new, lo, hi)
        
        # Helper: Tchebycheff Distance
        def tchebycheff(f, w, f_ideal):
            # d = max_k ( w_k * |f_k - f_ideal_k| )
            # Note: Some implementations use (f_k - f_ideal_k) * w_k
            # We use the absolute difference from ideal, scaled by weight.
            return np.max(w * np.abs(f - f_ideal))
        
        # Main Loop
        remaining = budget - used
        if remaining <= 0:
            # No budget left for search, return best found so far
            # We should return the non-dominated points from F
            return self._extract_nondominated(X, F, lo, hi, problem, rng, budget - used, deadline)
            
        # Shuffle order to avoid bias
        order = rng.permutation(N)
        
        for k in range(remaining):
            if time.time() > deadline - 10.0:
                break
                
            i = order[k % N]
            
            # Select parent from neighborhood of i
            if len(neighbors[i]) > 0:
                j_idx = neighbors[i][rng.integers(len(neighbors[i]))]
            else:
                j_idx = rng.integers(N)
                
            p1 = X[j_idx]
            
            # Select a second parent
            if len(neighbors[i]) > 1:
                j2_idx = neighbors[i][rng.integers(len(neighbors[i]))]
                if j2_idx == j_idx:
                    j2_idx = (j2_idx + 1) % len(neighbors[i])
                    j2_idx = neighbors[i][j2_idx]
            else:
                j2_idx = rng.integers(N)
                
            p2 = X[j2_idx]
            
            # Crossover
            if rng.random() < p_c:
                c1, _ = sbx_crossover(rng, p1, p2, lo, hi, eta_c)
            else:
                c1 = p1.copy()
                
            # Mutation
            c1 = poly_mutation(rng, c1, lo, hi, eta_m, p_m)
            
            f_new = np.array(problem(c1), dtype=float)
            used += 1
            
            # Update ideal point if new solution is better
            f_ideal = np.minimum(f_ideal, f_new)
            
            # Update neighborhood of i
            # Replace X[j] with c1 if c1 is better for subproblem j
            for j in neighbors[i]:
                # Calculate Tchebycheff for current X[j] and new c1
                # Using current f_ideal
                d_current = tchebycheff(F[j], weights[j], f_ideal)
                d_new = tchebycheff(f_new, weights[j], f_ideal)
                
                if d_new < d_current:
                    X[j] = c1
                    F[j] = f_new
                    # Note: f_ideal was already updated above if f_new was globally better.
                    # If f_new is not globally better, f_ideal remains same.
                    # This is a standard online update.
        
        # Final Local Polish
        if used < budget and time.time() < deadline - 20.0:
            X, F = self._local_polish(X, F, lo, hi, problem, rng, budget - used, deadline)
            
        return self._extract_nondominated(X, F, lo, hi, problem, rng, budget - used, deadline)

    def _extract_nondominated(self, X, F, lo, hi, problem, rng, remaining_budget, deadline):
        """Extract non-dominated points from the final population."""
        if len(F) == 0:
            return []
        
        # Simple non-dominated sorting
        n = len(F)
        nd = np.zeros(n, dtype=int)
        S = [[] for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    nd[j] += 1
        
        fronts = [[]]
        for i in range(n):
            if nd[i] == 0:
                fronts[0].append(i)
        
        i = 0
        while len(fronts[i]) > 0:
            Q = []
            for p in fronts[i]:
                for q in S[p]:
                    nd[q] -= 1
                    if nd[q] == 0:
                        Q.append(q)
            i += 1
            if len(Q) > 0:
                fronts.append(Q)
            else:
                break
        
        # Return indices of the first front (non-dominated)
        if fronts:
            return [X[i] for i in fronts[0]]
        return []

    def _local_polish(self, X, F, lo, hi, problem, rng, remaining_budget, deadline):
        """Refine non-dominated points with small perturbations."""
        if remaining_budget <= 0:
            return X, F
            
        # Identify non-dominated points
        n = len(F)
        nd = np.zeros(n, dtype=int)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    nd[j] += 1
        
        nd_indices = [i for i in range(n) if nd[i] == 0]
        
        if not nd_indices:
            return X, F
            
        # Try to improve each non-dominated point
        for idx in nd_indices:
            if remaining_budget <= 0 or time.time() > deadline - 15.0:
                break
                
            base_x = X[idx].copy()
            base_f = F[idx].copy()
            
            # Try a few small perturbations
            for _ in range(3):
                if remaining_budget <= 0 or time.time() > deadline - 15.0:
                    break
                    
                perturbation = 0.01 * (hi - lo) * rng.standard_normal(self.dim)
                candidate_x = np.clip(base_x + perturbation, lo, hi)
                candidate_f = np.array(problem(candidate_x), dtype=float)
                remaining_budget -= 1
                
                # Check if candidate dominates base
                if np.all(candidate_f <= base_f) and np.any(candidate_f < base_f):
                    X[idx] = candidate_x
                    F[idx] = candidate_f
                    base_x = candidate_x
                    base_f = candidate_f
        return X, F
# EVOLVE-BLOCK-END
