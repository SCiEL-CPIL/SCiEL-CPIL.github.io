"""MOEA/D with reference points, differential mutation, and neighborhood-based reproduction."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        range_ = hi - lo
        
        N = 40
        if self.budget < N:
            N = max(2, self.budget)
        
        G = (self.budget - N) // N
        if G < 1:
            X = lo + range_ * np.random.random((N, self.dim))
            F = np.array([problem(x) for x in X])
            return X
        
        # Generate weight vectors on simplex
        W = self._generate_weights(self.n_obj, N)
        
        # Initialize population
        X = lo + range_ * np.random.random((N, self.dim))
        F = np.array([problem(x) for x in X])
        used = N
        
        # Ideal and nadir points
        ideal = np.min(F, axis=0)
        nadir = np.max(F, axis=0)
        
        # Compute Tchebycheff fitness for each solution with each weight
        # fitness[i, j] = Tchebycheff(F[i], W[j], ideal)
        def tchebycheff(F, W, ideal):
            """Compute Tchebycheff distance for all solutions vs all weights."""
            n = F.shape[0]
            m = W.shape[0]
            # Normalize
            F_norm = (F - ideal) / (nadir - ideal + 1e-10)
            # Tchebycheff: max_k (w_k * |f_k - z_k*| / (z_k - z_k*))
            # For minimization: tchebycheff = max_k (w_k * f_k_norm_k)
            # Reshape: F_norm (n, n_obj), W (m, n_obj)
            # We want: for each i, j: max_k W[j,k] * F_norm[i,k]
            # = max over k of (F_norm[:,k,None] * W[None,:,k])
            prod = F_norm[:, :, None] * W[None, :, :]  # (n, n_obj, m)
            return np.max(prod, axis=1)  # (n, m)
        
        # Build neighborhood: for each weight, find K nearest weights
        K = 20
        neighbors = self._build_neighbors(W, K)
        
        for g in range(G):
            if time.time() > deadline:
                break
            
            # Update ideal point
            ideal = np.min(F, axis=0)
            nadir = np.max(F, axis=0)
            
            # Compute fitness matrix
            fit = tchebycheff(F, W, ideal)  # (N, N)
            
            # For each solution i, generate one offspring
            Q = np.empty((N, self.dim))
            Fq = np.empty((N, self.n_obj))
            
            for i in range(N):
                if time.time() > deadline or used >= self.budget:
                    break
                
                # Select 2 parents from neighborhood of i
                neigh = neighbors[i]
                # Tournament within neighborhood
                p1_idx = self._tournament_neighborhood(neigh, fit[i], 2)
                p2_idx = self._tournament_neighborhood(neigh, fit[i], 2)
                
                # Differential mutation
                if np.random.random() < 0.9:
                    # Select third parent from neighborhood
                    p3_idx = self._tournament_neighborhood(neigh, fit[i], 1)[0]
                    F_ = 0.8  # differential weight
                    child = X[p1_idx] + F_ * (X[p2_idx] - X[p3_idx])
                else:
                    child = X[p1_idx].copy()
                
                # Random mutation
                if np.random.random() < 0.5:
                    mut_idx = np.random.randint(self.dim)
                    child[mut_idx] = lo[mut_idx] + (hi[mut_idx] - lo[mut_idx]) * np.random.random()
                
                child = np.clip(child, lo, hi)
                
                Fq[i] = problem(child)
                Q[i] = child
                used += 1
            
            # Update population: replace solution i if child is better for weight i
            # Compare Tchebycheff fitness of child vs current solution i
            for i in range(N):
                if i >= used - N:  # only for evaluated children
                    break
                # Compute Tchebycheff for child with weight i
                Fc = (Fq[i] - ideal) / (nadir - ideal + 1e-10)
                child_fit = np.max(W[i] * Fc)
                if child_fit < fit[i, i]:
                    X[i] = Q[i]
                    F[i] = Fq[i]
                    fit[i, :] = tchebycheff(F[i:i+1], W, ideal)[0]
            
            # Update ideal/nadir after population update
            ideal = np.min(F, axis=0)
            nadir = np.max(F, axis=0)
        
        return X

    def _generate_weights(self, n_obj, N):
        """Generate N weight vectors on the (n_obj-1)-dimensional simplex."""
        # Use systematic generation for 3 objectives
        if n_obj == 3:
            # Generate points on 2D simplex
            # Use grid-like distribution
            n = int(np.ceil(np.sqrt(N)))
            weights = []
            for i in range(n):
                for j in range(n - i):
                    w1 = i / (n - 1) if n > 1 else 0
                    w2 = j / (n - 1 - i) if (n - 1 - i) > 0 else 0
                    w3 = 1 - w1 - w2
                    if w3 >= -1e-10:
                        weights.append([max(0, w1), max(0, w2), max(0, w3)])
            
            W = np.array(weights)
            # Normalize
            W = W / W.sum(axis=1, keepdims=True)
            
            # If we have more than N, sample; if less, add random
            if len(W) > N:
                idx = np.random.choice(len(W), N, replace=False)
                W = W[idx]
            elif len(W) < N:
                extra = np.random.dirichlet(np.ones(n_obj), size=N - len(W))
                W = np.vstack([W, extra])
            
            return W[:N]
        else:
            # General case: random Dirichlet
            return np.random.dirichlet(np.ones(n_obj), size=N)

    def _build_neighbors(self, W, K):
        """Build K-nearest neighbors for each weight vector."""
        N = len(W)
        # Compute pairwise distances
        diff = W[:, None, :] - W[None, :, :]
        dist = np.sum(diff ** 2, axis=2)
        
        neighbors = []
        for i in range(N):
            # Sort by distance to i
            order = np.argsort(dist[i])
            # Exclude self
            neigh = list(order[1:K+1])
            neighbors.append(neigh)
        
        return neighbors

    def _tournament_neighborhood(self, neigh, fit_row, k):
        """Tournament selection within a neighborhood list."""
        if len(neigh) <= k:
            return list(neigh)
        
        idxs = np.random.choice(len(neigh), size=k, replace=False)
        best = idxs[0]
        best_fit = fit_row[neigh[best]]
        
        for idx in idxs[1:]:
            f = fit_row[neigh[idx]]
            if f < best_fit:
                best = idx
                best_fit = f
        
        return [neigh[best]]
# EVOLVE-BLOCK-END
