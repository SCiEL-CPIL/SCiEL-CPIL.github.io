"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-estimation-based island model for WFG4 multi-objective optimization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290  # safety margin before 300s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        d = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Reference point for hypervolume calculation
        ref_point = np.array([3.0, 5.0, 7.0])

        # Number of islands and population per island
        n_islands = 5
        pop_per_island = 12

        # Initialize islands with different weight vectors for hypervolume estimation
        # Each island focuses on a different region of the Pareto front
        islands = []
        used = 0

        # Generate diverse initial points using weight-based sampling
        for i in range(n_islands):
            # Create weight vector emphasizing different objectives
            weights = np.random.dirichlet(np.ones(n_obj))
            # Initialize population around the weighted centroid
            X_island = lo + span * np.random.rand(pop_per_island, d)
            F_island = np.array([problem(x) for x in X_island])
            used += pop_per_island
            islands.append({
                'X': X_island,
                'F': F_island,
                'weights': weights
            })

        # Main optimization loop
        while used < budget:
            if time.time() - t0 > deadline:
                break

            # Process each island
            for island_idx in range(n_islands):
                if used >= budget or time.time() - t0 > deadline:
                    break

                island = islands[island_idx]
                X = island['X']
                F = island['F']
                weights = island['weights']

                # Generate one offspring per island per generation
                # Use weighted hypervolume contribution to guide search
                # Select parent based on hypervolume contribution
                parent_idx = self._select_parent_by_hv(F, weights, ref_point)
                base = X[parent_idx]

                # DE mutation with island-specific intensity
                # Higher weight variance means more exploration needed
                weight_spread = np.std(weights)
                F_factor = 0.3 + 0.4 * weight_spread + 0.3 * np.random.rand()
                
                # Select two other parents
                other_indices = np.random.choice(len(X), 2, replace=False)
                a = X[other_indices[0]]
                b = X[other_indices[1]]
                
                v = base + F_factor * (a - b)
                
                # Crossover
                CR = 0.6 + 0.2 * np.random.rand()
                u = np.random.rand(d) < CR
                if not np.any(u):
                    u[np.random.randint(d)] = True

                child = np.where(u, v, base)
                child = np.clip(child, lo, hi)

                fc = problem(child)
                used += 1

                # Update island population using hypervolume-based selection
                X_comb = np.vstack([X, child.reshape(1, -1)])
                F_comb = np.vstack([F, fc.reshape(1, -1)])

                # Select top pop_per_island based on hypervolume contribution
                selected_idx = self._select_by_hv(F_comb, weights, ref_point, pop_per_island)
                island['X'] = X_comb[selected_idx]
                island['F'] = F_comb[selected_idx]

            # Migration: exchange best individuals between islands
            if used < budget and time.time() - t0 < deadline:
                for i in range(n_islands):
                    if used >= budget:
                        break
                    # Find best individual in island i by hypervolume
                    best_idx = self._select_parent_by_hv(
                        islands[i]['F'], islands[i]['weights'], ref_point
                    )
                    donor_x = islands[i]['X'][best_idx].copy()
                    
                    # Migrate to a random other island
                    target = np.random.choice([j for j in range(n_islands) if j != i])
                    # Replace worst individual in target island
                    worst_idx = self._find_worst_by_hv(
                        islands[target]['F'], islands[target]['weights'], ref_point
                    )
                    islands[target]['X'][worst_idx] = donor_x
                    # Re-evaluate if needed (but we already have the value)
                    # For simplicity, we just replace the position; the F value 
                    # needs to be consistent. Since we're migrating the same point,
                    # we need to re-evaluate or track it. Let's re-evaluate to be safe.
                    fc = problem(islands[target]['X'][worst_idx])
                    islands[target]['F'][worst_idx] = fc
                    used += 1

        return None

    def _hypervolume_contribution(self, F, weights, ref_point):
        """
        Estimate hypervolume contribution of each point in F.
        Uses a simplified weighted approach for speed.
        """
        n = len(F)
        contributions = np.zeros(n)
        
        # Filter out points dominated by reference point
        valid_mask = np.all(F < ref_point, axis=1)
        if not np.any(valid_mask):
            return contributions
        
        valid_indices = np.where(valid_mask)[0]
        
        # Simple approximation: weight by distance to reference point
        # and inverse of local density
        for i in valid_indices:
            # Distance to reference point (normalized)
            dist = np.sum((ref_point - F[i]) ** 2)
            # Weight by objective weights
            weighted_score = np.sum(weights * F[i])
            # Contribution is higher if point is closer to reference (smaller dist)
            # and has better weighted score
            contributions[i] = dist * (1.0 / (1.0 + weighted_score))
        
        # Invert so that lower contribution means better (closer to ref, better score)
        # Actually, we want to select points that maximize hypervolume.
        # A point contributes more if it's "extreme" in some direction.
        # Let's use a different metric: the volume of the box from point to ref,
        # weighted by how unique it is.
        
        # Simpler: use the product of (ref - F[i]) as a proxy for volume
        volumes = np.prod(ref_point - F[valid_indices], axis=1)
        # But this doesn't account for non-domination.
        
        # Let's use a proper but fast HV approximation:
        # For each point, compute the volume it uniquely covers.
        # This is expensive. Instead, use a scalarized approach guided by weights.
        
        # Final simple approach: score = weighted sum of (ref - F)
        # Points with higher score contribute more to HV in the weighted direction.
        scores = np.sum(weights * (ref_point - F), axis=1)
        scores[~valid_mask] = -np.inf
        
        return scores

    def _select_parent_by_hv(self, F, weights, ref_point):
        """Select the best parent based on hypervolume contribution score."""
        scores = self._hypervolume_contribution(F, weights, ref_point)
        if np.all(scores == -np.inf):
            return np.random.randint(len(F))
        return np.argmax(scores)

    def _select_by_hv(self, F, weights, ref_point, k):
        """Select top k indices based on hypervolume contribution score."""
        scores = self._hypervolume_contribution(F, weights, ref_point)
        # Sort by score descending, take top k
        order = np.argsort(scores)[::-1]
        return order[:k]

    def _find_worst_by_hv(self, F, weights, ref_point):
        """Find the worst individual based on hypervolume contribution score."""
        scores = self._hypervolume_contribution(F, weights, ref_point)
        if np.all(scores == -np.inf):
            return np.random.randint(len(F))
        return np.argmin(scores)
# EVOLVE-BLOCK-END
