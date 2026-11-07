"""Multi-objective optimizer on WFG4 combining structured initialization with NSGA-II."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer using a hybrid strategy:
    1. Structured initialization to quickly locate the Pareto front region.
    2. NSGA-II style evolutionary search to refine and spread the population.
    3. Global archive tracking all non-dominated points for hypervolume calculation.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: Evaluator kills at 300s. We stop at 280s.
        deadline = start_time + 280.0

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Use global numpy random state (seeded by evaluator)
        rng = np.random

        all_F = []
        used = 0

        def evaluate(x):
            nonlocal used
            if used >= budget or time.time() > deadline:
                return None
            fc = problem(x)
            used += 1
            all_F.append(fc)
            return fc

        # --- Phase 1: Structured Initialization ---
        # WFG4 often has the trade-off controlled by the first variable.
        # We sample x[0] across its range and set x[1:] to midpoints.
        n_structured = min(20, budget // 5)
        if n_structured > 0:
            x0_vals = np.linspace(lo[0], hi[0], n_structured)
            x_mid = (lo + hi) / 2.0
            for x0 in x0_vals:
                x = x_mid.copy()
                x[0] = x0
                x = np.clip(x, lo, hi)
                evaluate(x)

        # Add some random points to ensure coverage
        n_random = min(10, budget - used)
        if n_random > 0:
            X_rand = lo + span * rng.random((n_random, dim))
            for x in X_rand:
                evaluate(x)

        # --- Phase 2: Evolutionary Search (NSGA-II) ---
        # Current population is the best non-dominated points from Phase 1
        # For simplicity, we start the population with the last evaluated points 
        # or a random subset if we have too many. 
        # Let's select a diverse subset of the current all_F for the initial population.
        
        # Helper: Check if point f is dominated by any point in list F_list
        def is_dominated(f, F_list):
            for af in F_list:
                if np.all(af <= f) and np.any(af < f):
                    return True
            return False

        # Helper: Fast Non-Dominated Sort
        def fast_non_dominated_sort(F_mat):
            n = len(F_mat)
            if n == 0:
                return []
            
            S = [[] for _ in range(n)]
            nd = [0] * n
            fronts = [[]]

            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F_mat[i] <= F_mat[j]) and np.any(F_mat[i] < F_mat[j]):
                        S[i].append(j)
                    elif np.all(F_mat[j] <= F_mat[i]) and np.any(F_mat[j] < F_mat[i]):
                        nd[i] += 1
                if nd[i] == 0:
                    fronts[0].append(i)

            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        nd[j] -= 1
                        if nd[j] == 0:
                            next_front.append(j)
                k += 1
                if next_front:
                    fronts.append(next_front)
            
            # Remove empty fronts at the end if any
            while fronts and not fronts[-1]:
                fronts.pop()
                
            return fronts

        # Helper: Crowding Distance
        def crowding_distance(F_mat, front_indices):
            cd = np.zeros(len(front_indices))
            if len(front_indices) <= 2:
                cd[:] = np.inf
                return cd
            
            for m in range(n_obj):
                vals = F_mat[front_indices, m]
                idx_sorted = np.argsort(vals)
                cd_sorted = np.zeros(len(front_indices))
                cd_sorted[0] = np.inf
                cd_sorted[-1] = np.inf
                rng_val = vals[idx_sorted[-1]] - vals[idx_sorted[0]]
                if rng_val > 1e-12:
                    for i in range(1, len(front_indices) - 1):
                        cd_sorted[i] += (vals[idx_sorted[i + 1]] - vals[idx_sorted[i - 1]]) / rng_val
                
                # Map back to original indices in front_indices
                # front_indices is a list of indices into F_mat
                # We need to assign cd_sorted[i] to the point front_indices[idx_sorted[i]]
                # But cd is an array aligned with front_indices.
                # Let's reconstruct carefully.
                # Actually, simpler:
                # Create a temp array of same length as front_indices
                # The sorted order tells us which point is which.
                # We just need to add the contribution to the correct slot in cd.
                
                # Re-doing this part to be safe:
                # cd is initialized to 0.
                # For each objective m:
                # 1. Sort indices in front_indices by F_mat[idx, m]
                # 2. Assign inf to min and max
                # 3. Add normalized difference to others
                
                # Let's implement a cleaner version of CD assignment
                pass # The pass above was just for thought, implementing below

            # Correct implementation of Crowding Distance
            cd = np.zeros(len(front_indices))
            if len(front_indices) <= 2:
                cd[:] = np.inf
                return cd

            for m in range(n_obj):
                # Get values for this front
                vals = F_mat[front_indices, m]
                # Sort indices within the front based on this objective
                sorted_local_indices = np.argsort(vals)
                
                cd_sorted = np.zeros(len(front_indices))
                cd_sorted[0] = np.inf
                cd_sorted[-1] = np.inf
                
                rng_val = vals[sorted_local_indices[-1]] - vals[sorted_local_indices[0]]
                if rng_val > 1e-12:
                    for i in range(1, len(front_indices) - 1):
                        cd_sorted[i] += (vals[sorted_local_indices[i + 1]] - vals[sorted_local_indices[i - 1]]) / rng_val
                
                # Add to main cd array
                # The i-th element in cd_sorted corresponds to the point front_indices[sorted_local_indices[i]]
                # So we need to add cd_sorted[i] to cd[index_of_front_indices[sorted_local_indices[i]]]
                # This mapping is tricky if we just use local indices.
                
                # Alternative: Create a dictionary or map
                # Let's just iterate through the sorted list and update cd
                # front_indices is a list. 
                # sorted_local_indices contains positions in the sorted array of vals.
                # The point corresponding to sorted_local_indices[i] is front_indices[sorted_local_indices[i]]?
                # No. 
                # vals = F_mat[front_indices, m]. 
                # So vals[k] corresponds to point front_indices[k].
                # sorted_local_indices is the permutation of [0..len-1] that sorts vals.
                # So the point at sorted position i is front_indices[sorted_local_indices[i]].
                # We want to add cd_sorted[i] to the cd entry for that point.
                # The cd entry for point front_indices[k] is cd[k].
                # So we add cd_sorted[i] to cd[sorted_local_indices[i]].
                
                for i in range(len(front_indices)):
                    cd[sorted_local_indices[i]] += cd_sorted[i]
            
            return cd

        # Initialize Population for Evolution
        # We take the non-dominated points from the initial phase.
        if all_F:
            F_init = np.array(all_F)
            fronts = fast_non_dominated_sort(F_init)
            
            # Select up to 20 diverse points from the first front
            pop_size = 20
            X_pop = []
            F_pop = []
            
            # We need the X vectors corresponding to F_init. 
            # We didn't store X for all_F. 
            # To fix this, we should store X in parallel with F during evaluation.
            
            # Let's restart the evaluation logic to store X.
            
            # --- RE-IMPLEMENTATION WITH X STORAGE ---
            
            all_F = []
            all_X = []
            used = 0

            def evaluate(x):
                nonlocal used
                if used >= budget or time.time() > deadline:
                    return None
                fc = problem(x)
                used += 1
                all_F.append(fc)
                all_X.append(x.copy())
                return fc

            # Phase 1: Structured Initialization
            n_structured = min(20, budget // 5)
            if n_structured > 0:
                x0_vals = np.linspace(lo[0], hi[0], n_structured)
                x_mid = (lo + hi) / 2.0
                for x0 in x0_vals:
                    x = x_mid.copy()
                    x[0] = x0
                    x = np.clip(x, lo, hi)
                    evaluate(x)

            # Add some random points
            n_random = min(10, budget - used)
            if n_random > 0:
                X_rand = lo + span * rng.random((n_random, dim))
                for x in X_rand:
                    evaluate(x)

            # Now we have all_X and all_F.
            # Select initial population
            if all_F:
                F_init = np.array(all_F)
                X_init = np.array(all_X)
                fronts = fast_non_dominated_sort(F_init)
                
                X_pop = []
                F_pop = []
                
                # Take from first front
                if fronts:
                    front0 = fronts[0]
                    # Sort by crowding distance
                    cd = crowding_distance(F_init, front0)
                    sorted_indices = np.argsort(cd)[::-1] # Descending CD
                    
                    for idx in sorted_indices:
                        if len(X_pop) >= pop_size:
                            break
                        X_pop.append(X_init[front0[idx]])
                        F_pop.append(F_init[front0[idx]])
                    
                    # If we don't have enough, fill with random from rest
                    if len(X_pop) < pop_size:
                        remaining_indices = [i for i in range(len(F_init)) if i not in front0]
                        rng.shuffle(remaining_indices)
                        for idx in remaining_indices:
                            if len(X_pop) >= pop_size:
                                break
                            X_pop.append(X_init[idx])
                            F_pop.append(F_init[idx])
                
                X_pop = np.array(X_pop)
                F_pop = np.array(F_pop)
            else:
                # Fallback: Random initialization
                X_pop = lo + span * rng.random((pop_size, dim))
                F_pop = np.array([problem(x) for x in X_pop])
                all_F.extend(F_pop)
                all_X.extend([x.copy() for x in X_pop])
                used += pop_size

        # Evolutionary Loop
        while used < budget and time.time() < deadline - 5.0:
            # Generate Offspring
            offspring_X = []
            offspring_F = []
            
            n_offspring = pop_size
            sigma = 0.1 # Mutation rate factor
            
            for _ in range(n_offspring):
                # Tournament Selection
                i = rng.randint(0, len(X_pop))
                j = rng.randint(0, len(X_pop))
                if i == j:
                    j = (j + 1) % len(X_pop)
                
                # SBX Crossover
                eta = 20.0
                if rng.random() < 0.5:
                    c1 = 0.5 * (1 + rng.random())
                    c2 = 1 - c1
                else:
                    c1 = 0.5 * (1 - rng.random())
                    c2 = 1 - c1
                    
                beta1 = (1 / (1 + 2 * c1)) ** (1 / (eta + 1)) if c1 > 0 else 1.0
                beta2 = (1 / (1 + 2 * c2)) ** (1 / (eta + 1)) if c2 > 0 else 1.0
                
                child1 = 0.5 * ((1 + beta1) * X_pop[i] + (1 - beta1) * X_pop[j])
                child2 = 0.5 * ((1 + beta2) * X_pop[j] + (1 - beta2) * X_pop[i])
                
                # Polynomial Mutation
                def mutate(x):
                    y = x.copy()
                    for k in range(dim):
                        if rng.random() < 0.9:
                            u = rng.random()
                            if u < 0.5:
                                delta = (2 * u) ** (1 / (eta + 1)) - 1
                            else:
                                delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
                            y[k] += delta * (hi[k] - lo[k]) * sigma
                    return np.clip(y, lo, hi)
                
                child1 = mutate(child1)
                child2 = mutate(child2)
                
                offspring_X.append(child1)
                offspring_X.append(child2)
                
                # Evaluate
                f1 = evaluate(child1)
                if f1 is not None:
                    offspring_F.append(f1)
                
                if used >= budget or time.time() > deadline:
                    break
                    
                f2 = evaluate(child2)
                if f2 is not None:
                    offspring_F.append(f2)
            
            if not offspring_X:
                break
                
            # Trim offspring to match evaluated F (in case of budget cutoff mid-pair)
            # offspring_X has 2 per iteration, offspring_F has 1 or 2.
            # We need to align them.
            # Since we append to offspring_X before evaluating, we might have extra Xs.
            # Let's just use the evaluated ones.
            # Re-evaluating is expensive, so we assume the order matches.
            # Actually, if f1 is None, we don't append to F, but we did append to X.
            # This desyncs X and F.
            
            # Better approach: Evaluate first, then append.
            # Let's restart the loop logic with safer appending.
            
            # (Self-Correction: The code above is getting messy with the desync. 
            # I will rewrite the loop cleanly in the final output.)
            
            break # Break out of this messy draft logic to write clean code below

        # --- CLEAN IMPLEMENTATION FOLLOWS ---
        
        # Reset state for clean implementation
        all_F = []
        all_X = []
        used = 0

        def evaluate(x):
            nonlocal used
            if used >= budget or time.time() > deadline:
                return None
            fc = problem(x)
            used += 1
            all_F.append(fc)
            all_X.append(x.copy())
            return fc

        # Phase 1: Structured Initialization
        n_structured = min(20, budget // 5)
        if n_structured > 0:
            x0_vals = np.linspace(lo[0], hi[0], n_structured)
            x_mid = (lo + hi) / 2.0
            for x0 in x0_vals:
                x = x_mid.copy()
                x[0] = x0
                x = np.clip(x, lo, hi)
                evaluate(x)

        # Add random points
        n_random = min(10, budget - used)
        if n_random > 0:
            X_rand = lo + span * rng.random((n_random, dim))
            for x in X_rand:
                evaluate(x)

        # Initialize Population
        pop_size = 20
        if all_F:
            F_init = np.array(all_F)
            X_init = np.array(all_X)
            fronts = fast_non_dominated_sort(F_init)
            
            X_pop = []
            F_pop = []
            
            if fronts:
                front0 = fronts[0]
                cd = crowding_distance(F_init, front0)
                sorted_indices = np.argsort(cd)[::-1]
                
                for idx in sorted_indices:
                    if len(X_pop) >= pop_size:
                        break
                    X_pop.append(X_init[front0[idx]])
                    F_pop.append(F_init[front0[idx]])
                
                if len(X_pop) < pop_size:
                    remaining_indices = [i for i in range(len(F_init)) if i not in front0]
                    rng.shuffle(remaining_indices)
                    for idx in remaining_indices:
                        if len(X_pop) >= pop_size:
                            break
                        X_pop.append(X_init[idx])
                        F_pop.append(F_init[idx])
            
            if len(X_pop) == 0:
                X_pop = [X_init[0]]
                F_pop = [F_init[0]]
            
            X_pop = np.array(X_pop)
            F_pop = np.array(F_pop)
        else:
            X_pop = lo + span * rng.random((pop_size, dim))
            F_pop = np.array([problem(x) for x in X_pop])
            all_F.extend(F_pop)
            all_X.extend([x.copy() for x in X_pop])
            used += pop_size

        # Evolutionary Loop
        eta = 20.0
        sigma = 0.1
        
        def mutate(x):
            y = x.copy()
            for k in range(dim):
                if rng.random() < 0.9:
                    u = rng.random()
                    if u < 0.5:
                        delta = (2 * u) ** (1 / (eta + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
                    y[k] += delta * (hi[k] - lo[k]) * sigma
            return np.clip(y, lo, hi)

        while used < budget and time.time() < deadline - 10.0:
            offspring_X = []
            offspring_F = []
            
            for _ in range(pop_size):
                if used >= budget or time.time() > deadline:
                    break
                
                i = rng.randint(0, len(X_pop))
                j = rng.randint(0, len(X_pop))
                if i == j:
                    j = (j + 1) % len(X_pop)
                
                # SBX Crossover
                if rng.random() < 0.5:
                    c1 = 0.5 * (1 + rng.random())
                    c2 = 1 - c1
                else:
                    c1 = 0.5 * (1 - rng.random())
                    c2 = 1 - c1
                    
                beta1 = (1 / (1 + 2 * c1)) ** (1 / (eta + 1)) if c1 > 0 else 1.0
                beta2 = (1 / (1 + 2 * c2)) ** (1 / (eta + 1)) if c2 > 0 else 1.0
                
                child1 = 0.5 * ((1 + beta1) * X_pop[i] + (1 - beta1) * X_pop[j])
                child2 = 0.5 * ((1 + beta2) * X_pop[j] + (1 - beta2) * X_pop[i])
                
                child1 = mutate(child1)
                child2 = mutate(child2)
                
                # Evaluate child 1
                f1 = evaluate(child1)
                if f1 is not None:
                    offspring_X.append(child1)
                    offspring_F.append(f1)
                    
                if used >= budget or time.time() > deadline:
                    break
                
                # Evaluate child 2
                f2 = evaluate(child2)
                if f2 is not None:
                    offspring_X.append(child2)
                    offspring_F.append(f2)
            
            if not offspring_X:
                break
            
            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)
            
            # Combine Parent and Offspring
            combined_X = np.vstack([X_pop, offspring_X])
            combined_F = np.vstack([F_pop, offspring_F])
            
            # Non-dominated Sort
            fronts = fast_non_dominated_sort(combined_F)
            
            # Select next population
            new_X = []
            new_F = []
            
            for front in fronts:
                if len(new_X) + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                else:
                    # Need to truncate this front
                    remaining_slots = pop_size - len(new_X)
                    cd = crowding_distance(combined_F, front)
                    sorted_indices = np.argsort(cd)[::-1]
                    for idx in sorted_indices[:remaining_slots]:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                    break
            
            if not new_X:
                break
                
            X_pop = np.array(new_X)
            F_pop = np.array(new_F)

        return all_F

# EVOLVE-BLOCK-END
