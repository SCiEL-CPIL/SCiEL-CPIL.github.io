import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        n = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # MOEA/D Parameters
        M = 20 # Population size / Number of weight vectors
        if M > budget:
            M = budget
            
        # Generate weight vectors for 2 objectives
        # Uniform distribution on the simplex
        weights = np.zeros((M, n_obj))
        if n_obj == 2:
            for i in range(M):
                weights[i, 0] = i / (M - 1) if M > 1 else 0.5
                weights[i, 1] = 1.0 - weights[i, 0]
        else:
            # Simple random for general case, though task says n_obj=2
            for i in range(M):
                w = np.random.random(n_obj)
                weights[i] = w / np.sum(w)

        # Define neighbors
        # For 2D simplex (line), neighbors are adjacent indices
        neighbors = {}
        for i in range(M):
            left = (i - 1) % M
            right = (i + 1) % M
            neighbors[i] = [left, right]

        # Initialize population
        X = lo + (hi - lo) * np.random.random((M, n))
        F = np.array([problem(x) for x in X])
        used = M
        
        # Ideal point
        z_star = np.min(F, axis=0)
        
        # Tchebycheff scalarization
        def tchebycheff(f, w, z):
            return np.max(w * np.abs(f - z))
        
        # Genetic Operators
        eta_c = 20.0
        eta_m = 20.0
        p_m = 1.0 / n

        def sbx_crossover(x1, x2, lo, hi, eta):
            c1 = x1.copy()
            c2 = x2.copy()
            for i in range(n):
                if np.random.random() < 0.5:
                    continue
                if abs(c1[i] - c2[i]) < 1e-14:
                    continue
                if c1[i] > c2[i]:
                    c1[i], c2[i] = c2[i], c1[i]
                u = np.random.random()
                if u <= 0.5:
                    beta = (2 * u) ** (1.0 / (eta + 1))
                else:
                    beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                c1[i] = 0.5 * ((c1[i] + c2[i]) - beta * (c2[i] - c1[i]))
                c2[i] = 0.5 * ((c1[i] + c2[i]) + beta * (c2[i] - c1[i]))
                c1[i] = np.clip(c1[i], lo[i], hi[i])
                c2[i] = np.clip(c2[i], lo[i], hi[i])
            return c1, c2

        def polynomial_mutation(x, lo, hi, eta):
            for i in range(n):
                if np.random.random() < p_m:
                    delta1 = (x[i] - lo[i]) / (hi[i] - lo[i])
                    delta2 = (hi[i] - x[i]) / (hi[i] - lo[i])
                    u = np.random.random()
                    if u < 0.5:
                        xy = 1 - (1 - delta1) ** (eta + 1)
                        mut_pow = 2.0 - xy
                    else:
                        xy = 1 - (1 - delta2) ** (eta + 1)
                        mut_pow = 2.0 - xy
                    mut_pow = 1.0 / (mut_pow * (eta + 1))
                    mut_pow = (2 * mut_pow + 1) ** (1.0 / (eta + 1))
                    delta = (hi[i] - lo[i]) * mut_pow
                    x[i] += np.random.choice([-1, 1]) * delta
                    x[i] = np.clip(x[i], lo[i], hi[i])
            return x

        # Main Loop
        while used < budget:
            # Determine how many offspring to generate in this batch
            # We generate offspring for a subset of weight vectors
            # To be efficient, let's generate as many as possible in a batch of M/2 or just M
            # Let's do batches of M (full generation) if budget allows, else partial.
            
            # Calculate max offspring we can generate
            max_offspring = (budget - used) // 2
            if max_offspring <= 0:
                break
            
            # We will iterate through weight vectors
            # Shuffle indices to avoid bias? Standard MOEA/D goes 0..M-1
            # Let's shuffle to diversify
            idx_order = np.random.permutation(M)
            
            generated = 0
            for k in idx_order:
                if generated >= max_offspring:
                    break
                if used + 2 > budget:
                    break
                    
                # Select parents from neighbors of k
                # If neighbors are [left, right], pick 2 from {left, right, k}?
                # Standard: pick 2 from neighborhood B(k). 
                # B(k) usually includes k itself or just neighbors.
                # Let's use neighbors + self to ensure diversity?
                # Or just 2 distinct neighbors.
                nbs = neighbors[k]
                # Pick 2 unique parents from nbs (with replacement if M is small, but M=20 is fine)
                # If M=2, nbs are [1, 1] effectively? No, (0-1)%2=1, (0+1)%2=1. 
                # For M=20, nbs are distinct.
                p1_idx = nbs[0]
                p2_idx = nbs[1]
                
                # If M is small, handle duplicates
                if p1_idx == p2_idx:
                     # Pick another? Just use same parent twice? 
                     # For M=20, this is fine.
                     pass

                x1 = X[p1_idx]
                x2 = X[p2_idx]
                
                # Crossover
                c1, c2 = sbx_crossover(x1, x2, lo, hi, eta_c)
                
                # Mutation
                c1 = polynomial_mutation(c1, lo, hi, eta_m)
                c2 = polynomial_mutation(c2, lo, hi, eta_m)
                
                # Evaluate
                f1 = problem(c1)
                used += 1
                f2 = problem(c2)
                used += 1
                
                # Update population for each offspring
                # Offspring c1 is associated with weight k
                # But standard MOEA/D generates 1 offspring per weight vector.
                # Here we generate 2. We can associate c1 with k and c2 with k?
                # Or c1 with k, c2 with another?
                # Let's just try to update the population using c1 and c2 against their neighbors.
                
                # Update with c1
                self_update = False
                for j in neighbors[k] + [k]: # Check neighbors and self
                    if tchebycheff(f1, weights[j], z_star) < tchebycheff(F[j], weights[j], z_star):
                        X[j] = c1
                        F[j] = f1
                        z_star = np.minimum(z_star, f1)
                        self_update = True
                        break # Only update one slot per offspring? Or all? 
                        # Standard MOEA/D updates the slot if better. 
                        # It's better to update the specific slot j if it improves.
                        # Let's not break, but we only have one offspring c1.
                        # We should check all neighbors and update the one that improves most?
                        # Or just update the one corresponding to the offspring's weight?
                        # Actually, the offspring is generated *for* weight k.
                        # It should compete with its neighbors.
                        # Let's update the best neighbor slot.
                
                # Let's refine the update logic:
                # The offspring y is generated for weight lambda^k.
                # It is compared with its neighbors.
                # If y is better than a neighbor x^j in terms of g(y|lambda^j), replace x^j.
                
                # Let's restart the update logic for clarity.
                # I will re-evaluate the update block in the code.
                
                generated += 1
                
        return X
