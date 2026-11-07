"""Multi-objective optimizer on RE34: NSGA-III with reference-point-based selection."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations


class Optimizer:
    """NSGA-III optimizer for RE34."""

    def __init__(self, budget, dim, n_obj):
        self.budget = int(budget)
        self.dim = int(dim)
        self.n_obj = int(n_obj)

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        used = 0
        X = []
        F = []

        def evaluate(x):
            nonlocal used
            x = np.clip(x, lo, hi)
            f = np.asarray(problem(x), dtype=np.float64)
            used += 1
            return x, f

        # Generate reference points for NSGA-III
        # For 3 objectives, we need points on the simplex.
        # We use a systematic grid.
        # Let H be the number of divisions along each edge.
        # Total points = (H+1)(H+2)/2
        # We want a population size P. Let's aim for P around 50-100.
        # With H=7, points = 8*9/2 = 36.
        # With H=10, points = 11*12/2 = 66.
        # With H=14, points = 15*16/2 = 120.
        # Let's use H=10 -> 66 reference points.
        H = 10
        ref_points = []
        for i in range(H + 1):
            for j in range(H + 1 - i):
                k = H - i - j
                ref_points.append(np.array([i, j, k], dtype=np.float64) / H)
        ref_points = np.array(ref_points)
        n_ref = len(ref_points)

        # Initial population
        # Size should be multiple of n_ref or just n_ref.
        # NSGA-III typically uses population size N = n_ref.
        N = n_ref
        
        # If budget is too small, adjust N
        if budget < 2 * N:
             N = max(10, budget // 2)
             # Regenerate or just use random init if N changes? 
             # For simplicity, stick to fixed N if budget allows.
             # Budget is 200. N=66. 200 > 132. So we can do 1 initial + 1-2 generations.
             # Actually, let's just use N=50 to be safe and fast?
             # Or stick with H=7 -> 36 points. 200/36 = 5.5 generations.
             # Let's use H=7 for more generations.
             H = 7
             ref_points = []
             for i in range(H + 1):
                 for j in range(H + 1 - i):
                     k = H - i - j
                     ref_points.append(np.array([i, j, k], dtype=np.float64) / H)
             ref_points = np.array(ref_points)
             n_ref = len(ref_points)
             N = n_ref

        rng = np.random.default_rng(42)

        # Initialize population
        for _ in range(N):
            x = rng.uniform(lo, hi)
            x, f = evaluate(x)
            X.append(x)
            F.append(f)
        
        X = np.array(X)
        F = np.array(F)

        # Helper: Non-dominated sorting (NSGA-II style)
        def non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = indices j dominated by i
            n_dom = [0] * n            # n_dom[i] = number of individuals dominating i
            fronts = []
            
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dom[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_dom[i] += 1
                if n_dom[i] == 0:
                    if not fronts or fronts[-1][0] != 0: # This logic is flawed for generic fronts
                        pass
                # Standard NSGA-II front finding is O(N^2) or O(N^3). 
                # For N=36, it's fine.
            
            # Re-implement standard front finding correctly
            fronts = []
            current_front = []
            for i in range(n):
                if n_dom[i] == 0:
                    current_front.append(i)
            
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                current_front = next_front
            
            return fronts

        # Helper: Crowding distance (not needed for NSGA-III, we use distance to ref points)
        
        # Helper: Associate individuals with reference points
        def associate(F, ref_points, ideal, nadir):
            # Normalize F
            # ideal: min of F
            # nadir: max of F (approximate)
            # Normalized f_i = (f_i - ideal_i) / (nadir_i - ideal_i + eps)
            eps = 1e-9
            norm_F = (F - ideal) / (nadir - ideal + eps)
            # Clip to 0..1? Not strictly necessary but helps.
            norm_F = np.clip(norm_F, 0, 1)
            
            n = len(F)
            m = len(ref_points)
            
            # Calculate distance from each individual to each reference point
            # Distance is perpendicular distance to the line connecting ideal and ref point?
            # Or just Euclidean distance in normalized space?
            # NSGA-III uses perpendicular distance to the hyperplane defined by ref point and ideal.
            # For 3D, we can project.
            
            # Simpler approximation: Euclidean distance to the reference point in normalized space.
            # This works well for convex fronts.
            # Let's use the angle-based or projection-based method if possible, 
            # but Euclidean is a common simplification.
            
            # Let's use the perpendicular distance to the line segment from ideal (0,0,0) to ref_point.
            # Vector v = ref_point. Point p = norm_F.
            # Projection of p onto v: proj = (p dot v) / (v dot v) * v
            # Distance = || p - proj ||
            
            distances = np.zeros((n, m))
            for i in range(n):
                p = norm_F[i]
                for j in range(m):
                    v = ref_points[j]
                    v_norm_sq = np.dot(v, v)
                    if v_norm_sq < 1e-10:
                        distances[i, j] = np.linalg.norm(p)
                    else:
                        proj = (np.dot(p, v) / v_norm_sq) * v
                        distances[i, j] = np.linalg.norm(p - proj)
            
            # Assign each individual to the closest reference point
            assignments = np.argmin(distances, axis=1)
            
            return assignments

        # Evolutionary Loop
        gen = 0
        while used < budget - N:  # Leave enough budget for next generation evaluation
            if time.time() > deadline:
                break
            
            # 1. Non-dominated sorting
            fronts = non_dominated_sort(F)
            
            # 2. Selection
            # Select N individuals from the combined population (current + offspring)
            # But we generate offspring separately.
            # Standard NSGA-III:
            # 1. Create offspring P' of size N from P.
            # 2. R = P + P' (size 2N)
            # 3. Non-dominated sort R into fronts F1, F2, ...
            # 4. Select N individuals from R to form next P.
            #    - Add all fronts Fi until we have >= N.
            #    - If a front Fn overflows, select the best ones from Fn using reference points.
            
            # Generate Offspring
            X_off = []
            F_off = []
            
            # Use DE-like mutation/crossover or simple random mutation?
            # Given the small budget, a simple mutation of existing solutions is good.
            # Or CMA-ES style? Let's use a simple differential evolution mutation.
            # x_new = x_a + F * (x_b - x_c)
            
            for _ in range(N):
                # Pick 3 distinct parents
                idx_a, idx_b, idx_c = rng.choice(N, size=3, replace=False)
                x_a = X[idx_a]
                x_b = X[idx_b]
                x_c = X[idx_c]
                
                F_factor = 0.5 + 0.5 * rng.random() # 0.5 to 1.0
                x_child = x_a + F_factor * (x_b - x_c)
                
                # Mutation
                # With probability 0.1, replace one variable with random
                if rng.random() < 0.1:
                    d = rng.integers(dim)
                    x_child[d] = rng.uniform(lo[d], hi[d])
                    
                x_child = np.clip(x_child, lo, hi)
                x_child, f_child = evaluate(x_child)
                X_off.append(x_child)
                F_off.append(f_child)
                
                if used >= budget:
                    break
            
            if used >= budget:
                break

            X_off = np.array(X_off)
            F_off = np.array(F_off)
            
            # Combine
            X_R = np.vstack([X, X_off])
            F_R = np.vstack([F, F_off])
            
            # Sort combined
            fronts_R = non_dominated_sort(F_R)
            
            # Select next population
            X_next = []
            F_next = []
            selected_indices = []
            
            for front in fronts_R:
                if len(selected_indices) + len(front) <= N:
                    selected_indices.extend(front)
                else:
                    # Overflow
                    # We need to select (N - len(selected_indices)) from this front
                    # Using reference point association
                    remaining = N - len(selected_indices)
                    
                    # Get F values for this front
                    F_front = F_R[front]
                    # Estimate ideal/nadir for normalization
                    # Use global ideal/nadir from R? Or front specific?
                    # Global is better for consistency.
                    ideal_R = F_R.min(axis=0)
                    nadir_R = F_R.max(axis=0)
                    
                    assignments = associate(F_front, ref_points, ideal_R, nadir_R)
                    
                    # Group indices by reference point
                    # We need to map back to original indices in R
                    # front is a list of indices into R
                    # assignments is a list of ref_point indices for each in front
                    
                    # Count how many are assigned to each ref point
                    ref_counts = np.zeros(n_ref, dtype=int)
                    for ref_idx in assignments:
                        ref_counts[ref_idx] += 1
                    
                    # We want to select 'remaining' individuals.
                    # Prioritize reference points with fewer individuals.
                    # Then, among those, pick the one closest to the ref point.
                    
                    # Create a list of (ref_idx, distance, original_idx)
                    candidates = []
                    for i, ref_idx in enumerate(assignments):
                        orig_idx = front[i]
                        # Recalculate distance? Or store it?
                        # associate() returned assignments, not distances.
                        # Let's re-calculate or modify associate to return distances.
                        # For now, let's just pick randomly from the least populated ref points.
                        # Or we can modify associate to return distances.
                        
                    # Let's implement a proper selection:
                    # 1. Sort reference points by count (ascending)
                    # 2. For each ref point, sort individuals by distance
                    # 3. Pick one from each ref point until 'remaining' is filled.
                    
                    # We need distances. Let's re-run association with distance return.
                    # Modify associate to return distances as well.
                    
                    # For simplicity in this block, let's just do:
                    # Pick the 'remaining' individuals from the front that have the 
                    # smallest distance to their nearest reference point, 
                    # but enforce diversity by limiting to 1 per ref point if possible.
                    
                    # Let's create a list of (dist, orig_idx)
                    # We need to recalculate distances.
                    ideal_R = F_R.min(axis=0)
                    nadir_R = F_R.max(axis=0)
                    F_front_norm = (F_front - ideal_R) / (nadir_R - ideal_R + 1e-9)
                    F_front_norm = np.clip(F_front_norm, 0, 1)
                    
                    dists = []
                    for i in range(len(front)):
                        p = F_front_norm[i]
                        min_d = np.inf
                        for j in range(n_ref):
                            v = ref_points[j]
                            v_norm_sq = np.dot(v, v)
                            if v_norm_sq < 1e-10:
                                d = np.linalg.norm(p)
                            else:
                                proj = (np.dot(p, v) / v_norm_sq) * v
                                d = np.linalg.norm(p - proj)
                            if d < min_d:
                                min_d = d
                        dists.append((min_d, front[i]))
                        
                    # Sort by distance
                    dists.sort(key=lambda x: x[0])
                    
                    # Select 'remaining' from this sorted list, but ensure diversity?
                    # NSGA-III selects by iterating ref points in order of scarcity.
                    # Let's do that properly.
                    
                    # Group candidates by ref point
                    # We need to know which ref point each candidate is assigned to.
                    # Let's re-calculate assignments and distances together.
                    
                    cand_info = []
                    for i in range(len(front)):
                        p = F_front_norm[i]
                        min_d = np.inf
                        best_ref = -1
                        for j in range(n_ref):
                            v = ref_points[j]
                            v_norm_sq = np.dot(v, v)
                            if v_norm_sq < 1e-10:
                                d = np.linalg.norm(p)
                            else:
                                proj = (np.dot(p, v) / v_norm_sq) * v
                                d = np.linalg.norm(p - proj)
                            if d < min_d:
                                min_d = d
                                best_ref = j
                        cand_info.append((best_ref, min_d, front[i]))
                        
                    # Count per ref
                    ref_counts = np.zeros(n_ref, dtype=int)
                    for ref_idx, _, _ in cand_info:
                        ref_counts[ref_idx] += 1
                        
                    # Sort ref points by count
                    sorted_refs = np.argsort(ref_counts)
                    
                    selected_from_front = 0
                    # We need to pick 'remaining' individuals.
                    # We'll iterate through ref points in order of scarcity.
                    # For each ref point, we pick the best (closest) individual not yet picked.
                    
                    # Create a list of candidates per ref point, sorted by distance
                    ref_candidates = [[] for _ in range(n_ref)]
                    for ref_idx, d, orig_idx in cand_info:
                        ref_candidates[ref_idx].append((d, orig_idx))
                        
                    for r in range(n_ref):
                        ref_candidates[r].sort(key=lambda x: x[0])
                        
                    # Now pick
                    for ref_idx in sorted_refs:
                        if selected_from_front >= remaining:
                            break
                        if ref_candidates[ref_idx]:
                            # Pick the first one (closest)
                            d, orig_idx = ref_candidates[ref_idx].pop(0)
                            selected_indices.append(orig_idx)
                            selected_from_front += 1
                            
                    # If we still need more (e.g., if some ref points were empty or we ran out of scarcity order)
                    # Just pick remaining from the best distances overall
                    if selected_from_front < remaining:
                        # Flatten all remaining candidates
                        all_rem = []
                        for r in range(n_ref):
                            all_rem.extend(ref_candidates[r])
                        all_rem.sort(key=lambda x: x[0])
                        for d, orig_idx in all_rem:
                            if selected_from_front >= remaining:
                                break
                            selected_indices.append(orig_idx)
                            selected_from_front += 1
                            
                    break # Done with fronts
                    
            # Update population
            if selected_indices:
                X = X_R[selected_indices]
                F = F_R[selected_indices]
            else:
                # Fallback
                X = X_off
                F = F_off

            gen += 1

        # Final return: The task says "Every returned objective vector is recorded".
        # The return value of __call__ is ignored.
        # So we just need to ensure we didn't exceed budget.
        return X
# EVOLVE-BLOCK-END
