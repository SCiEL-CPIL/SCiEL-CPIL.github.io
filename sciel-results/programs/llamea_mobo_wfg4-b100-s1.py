"""Multi-objective optimizer for WFG4 using NSGA-II with deterministic RNG and local search polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II based multi-objective optimizer for WFG4 with deterministic RNG, archive guidance, and local search polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 290.0  # Safety margin from 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        span = hi - lo

        # Parameters
        pop_size = 20
        cx_prob = 0.9
        mut_prob = 1.0 / dim
        eta_c = 15.0
        eta_m = 20.0
        archive_prob = 0.2
        polish_budget = 20  # Reserve evaluations for local search polish

        # Use a deterministic RNG to ensure consistent, high-quality search behavior
        rng = np.random.default_rng(42)
        used = 0

        def time_left():
            return deadline - (time.time() - start_time)

        def evaluate(x):
            nonlocal used
            f = problem(x)
            used += 1
            return f

        # Global Archive: list of (x, f) tuples for all non-dominated points found
        archive_X = []
        archive_F = []

        def is_dominated(f_new, archive_F_list):
            """Check if f_new is dominated by any point in archive."""
            for f_old in archive_F_list:
                if np.all(f_old <= f_new) and np.any(f_old < f_new):
                    return True
            return False

        def add_to_archive(x, f):
            """Add point to archive if non-dominated; remove dominated points."""
            nonlocal archive_X, archive_F
            if is_dominated(f, archive_F):
                return False
            
            # Check if it's a duplicate or strictly worse in all objectives (handled by is_dominated)
            # If not dominated, remove any points it dominates
            new_archive_X = []
            new_archive_F = []
            for i in range(len(archive_X)):
                f_old = archive_F[i]
                if not (np.all(f <= f_old) and np.any(f < f_old)):
                    new_archive_X.append(archive_X[i])
                    new_archive_F.append(f_old)
            
            new_archive_X.append(x.copy())
            new_archive_F.append(np.array(f))
            archive_X = new_archive_X
            archive_F = new_archive_F
            return True

        def non_dominated_sort(F):
            """
            Fast non-dominated sorting.
            Returns a list of lists, where each inner list contains indices of solutions in that front.
            """
            n = len(F)
            if n == 0:
                return [], np.zeros(0, dtype=int)
            S = [[] for _ in range(n)]  # S[i] = set of indices dominated by i
            n_d = np.zeros(n, dtype=int)  # n_d[i] = number of solutions dominating i
            rank = np.zeros(n, dtype=int)
            
            fronts = []
            current_front = []
            
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_d[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_d[i] += 1
                
                if n_d[i] == 0:
                    rank[i] = 0
                    current_front.append(i)
            
            if current_front:
                fronts.append(current_front)
            k = 1
            
            while current_front:
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_d[j] -= 1
                        if n_d[j] == 0:
                            rank[j] = k
                            next_front.append(j)
                if not next_front:
                    break
                fronts.append(next_front)
                current_front = next_front
                k += 1
                
            return fronts, rank

        def crowding_distance(F, front_indices):
            """
            Calculate crowding distance for solutions in a front.
            Returns a dictionary mapping index to distance.
            """
            if not front_indices:
                return {}
            
            m = len(front_indices)
            if m <= 2:
                return {i: float('inf') for i in front_indices}
            
            dist = {i: 0.0 for i in front_indices}
            F_sub = F[front_indices]
            
            for k in range(n_obj):
                # Sort by objective k
                sorted_indices = np.argsort(F_sub[:, k])
                sorted_F_k = F_sub[sorted_indices, k]
                
                # First and last have infinite distance
                dist[front_indices[sorted_indices[0]]] = float('inf')
                dist[front_indices[sorted_indices[-1]]] = float('inf')
                
                range_k = sorted_F_k[-1] - sorted_F_k[0]
                if range_k == 0:
                    continue
                    
                for i in range(1, m - 1):
                    idx = front_indices[sorted_indices[i]]
                    dist[idx] += (sorted_F_k[i + 1] - sorted_F_k[i - 1]) / range_k
                    
            return dist

        def select_tournament(pop_X, pop_F, pop_rank, pop_dist):
            """
            Binary tournament selection from the population.
            Returns indices of two parents.
            """
            n = len(pop_X)
            if n == 0:
                return 0, 0
            if n == 1:
                return 0, 0
            
            i1 = rng.integers(0, n)
            i2 = rng.integers(0, n)
            
            # Compare rank, then crowding distance
            if pop_rank[i1] < pop_rank[i2]:
                winner1 = i1
            elif pop_rank[i2] < pop_rank[i1]:
                winner1 = i2
            else:
                if pop_dist[i1] > pop_dist[i2]:
                    winner1 = i1
                else:
                    winner1 = i2
                    
            # Select second parent
            i3 = rng.integers(0, n)
            i4 = rng.integers(0, n)
            
            if pop_rank[i3] < pop_rank[i4]:
                winner2 = i3
            elif pop_rank[i4] < pop_rank[i3]:
                winner2 = i4
            else:
                if pop_dist[i3] > pop_dist[i4]:
                    winner2 = i3
                else:
                    winner2 = i4
                    
            return winner1, winner2

        def select_from_archive():
            """
            Select two random parents from the global archive.
            Returns the two x-vectors.
            """
            if len(archive_X) == 0:
                return None, None
            if len(archive_X) == 1:
                return archive_X[0].copy(), archive_X[0].copy()
            
            i1 = rng.integers(0, len(archive_X))
            i2 = rng.integers(0, len(archive_X))
            return archive_X[i1].copy(), archive_X[i2].copy()

        def sbx_crossover(p1, p2):
            """
            Simulated Binary Crossover.
            """
            if rng.random() > cx_prob:
                return p1.copy(), p2.copy()
            
            c1 = p1.copy()
            c2 = p2.copy()
            
            for i in range(dim):
                if rng.random() < 0.5:
                    if p1[i] != p2[i]:
                        if p1[i] < p2[i]:
                            x1, x2 = p1[i], p2[i]
                        else:
                            x1, x2 = p2[i], p1[i]
                        
                        u = rng.random()
                        if u <= 0.5:
                            beta = (2 * u) ** (1 / (eta_c + 1))
                        else:
                            beta = (1 / (2 * (1 - u))) ** (1 / (eta_c + 1))
                        
                        c1[i] = 0.5 * ((1 + beta) * x1 + (1 - beta) * x2)
                        c2[i] = 0.5 * ((1 - beta) * x1 + (1 + beta) * x2)
                        
                        # Clip to bounds
                        c1[i] = np.clip(c1[i], lo[i], hi[i])
                        c2[i] = np.clip(c2[i], lo[i], hi[i])
                        
            return c1, c2

        def polynomial_mutation(x):
            """
            Polynomial Mutation.
            """
            x_mut = x.copy()
            for i in range(dim):
                if rng.random() < mut_prob:
                    u = rng.random()
                    if u < 0.5:
                        delta_1 = (2 * u) ** (1 / (eta_m + 1)) - 1
                    else:
                        delta_1 = 1 - (2 * (1 - u)) ** (1 / (eta_m + 1))
                    
                    x_mut[i] = x[i] + delta_1 * (hi[i] - lo[i])
                    x_mut[i] = np.clip(x_mut[i], lo[i], hi[i])
            return x_mut

        # Phase 1: Initialization with LHS
        init_X = np.zeros((pop_size, dim))
        for d in range(dim):
            perm = rng.permutation(pop_size)
            init_X[:, d] = (perm + rng.random(pop_size)) / pop_size
        init_X = lo + init_X * span
        
        init_F = []
        for i in range(pop_size):
            if used >= budget or time_left() < 5.0:
                break
            f = evaluate(init_X[i])
            init_F.append(f)
            add_to_archive(init_X[i].copy(), np.array(f))
            
        # If we ran out of budget during init, return what we have
        if used >= budget or time_left() < 5.0:
            if archive_X:
                return np.array(archive_X)
            return np.array(init_X[:used]) if used > 0 else np.zeros((1, dim))

        pop_X = init_X[:used]
        pop_F = np.array(init_F)

        # Main Loop: NSGA-II
        # We stop the main loop when we have used (budget - polish_budget) evaluations
        main_budget = budget - polish_budget
        
        while used < main_budget and time_left() > 10.0:
            # Compute ranks and crowding distances for current population
            fronts, pop_rank = non_dominated_sort(pop_F)
            pop_dist = np.zeros(len(pop_X))
            for front in fronts:
                dist_dict = crowding_distance(pop_F, front)
                for idx in front:
                    pop_dist[idx] = dist_dict[idx]
            
            # Generate Offspring
            off_X = []
            off_F = []
            
            while len(off_X) < pop_size and used < main_budget and time_left() > 5.0:
                # Decide whether to use archive or population for parents
                use_archive = rng.random() < archive_prob
                
                if use_archive and len(archive_X) >= 2:
                    p1, p2 = select_from_archive()
                    if p1 is None:
                        use_archive = False
                        
                if not use_archive:
                    p1_idx, p2_idx = select_tournament(pop_X, pop_F, pop_rank, pop_dist)
                    p1 = pop_X[p1_idx]
                    p2 = pop_X[p2_idx]
                
                c1, c2 = sbx_crossover(p1, p2)
                c1 = polynomial_mutation(c1)
                c2 = polynomial_mutation(c2)
                
                f1 = evaluate(c1)
                off_X.append(c1)
                off_F.append(f1)
                add_to_archive(c1.copy(), np.array(f1))
                
                if used < main_budget and time_left() > 5.0:
                    f2 = evaluate(c2)
                    off_X.append(c2)
                    off_F.append(f2)
                    add_to_archive(c2.copy(), np.array(f2))
                    
            if not off_X:
                break
                
            # Combine Parents and Offspring
            comb_X = np.vstack([pop_X, np.array(off_X)])
            comb_F = np.vstack([pop_F, np.array(off_F)])
            
            # Sort and Select Next Generation
            fronts, comb_rank = non_dominated_sort(comb_F)
            comb_dist = np.zeros(len(comb_X))
            for front in fronts:
                dist_dict = crowding_distance(comb_F, front)
                for idx in front:
                    comb_dist[idx] = dist_dict[idx]
            
            next_X = []
            next_F = []
            
            for front in fronts:
                if len(next_X) + len(front) <= pop_size:
                    next_X.extend([comb_X[i] for i in front])
                    next_F.extend([comb_F[i] for i in front])
                else:
                    # Need to select from this front based on crowding distance
                    remaining = pop_size - len(next_X)
                    # Sort front indices by crowding distance (descending)
                    front_sorted = sorted(front, key=lambda i: comb_dist[i], reverse=True)
                    selected = front_sorted[:remaining]
                    next_X.extend([comb_X[i] for i in selected])
                    next_F.extend([comb_F[i] for i in selected])
                    break
            
            if not next_X:
                break
                
            pop_X = np.array(next_X)
            pop_F = np.array(next_F)

        # Phase 2: Local Search Polish
        # Use remaining budget to polish the best points in the archive
        # We pick the top K points in the archive (based on non-domination, though all are non-dominated)
        # and try to improve them with small random perturbations.
        
        if archive_X and used < budget and time_left() > 5.0:
            # Sort archive by a simple metric (e.g., sum of objectives) to pick "best" candidates
            # Since all are non-dominated, we just pick a random subset or all of them if small
            # Let's pick up to 5 best candidates (lowest sum of objectives) for polishing
            archive_indices = np.argsort(np.sum(archive_F, axis=1))
            polish_candidates = archive_indices[:min(5, len(archive_X))]
            
            polish_evals = budget - used
            
            for idx in polish_candidates:
                if used >= budget or time_left() < 2.0:
                    break
                
                x_base = archive_X[idx].copy()
                f_base = archive_F[idx].copy()
                
                # Try a few local perturbations
                local_steps = min(4, polish_evals)
                for _ in range(local_steps):
                    if used >= budget or time_left() < 2.0:
                        break
                    
                    # Small random walk
                    x_trial = x_base + rng.normal(0, 0.05, size=dim)
                    x_trial = np.clip(x_trial, lo, hi)
                    
                    f_trial = evaluate(x_trial)
                    
                    # If better than base (in at least one obj and not worse in others), update base
                    # Actually, for hypervolume, we just want to add to archive if non-dominated
                    # But if it dominates f_base, we should replace f_base in the archive logic
                    # add_to_archive handles this: if f_trial dominates f_base, f_base is removed.
                    # If f_trial is non-dominated, it is added.
                    # We want to push the front, so we check if it's an improvement.
                    # Simple check: is f_trial better than f_base in at least one objective?
                    if np.any(f_trial < f_base):
                        # It might be better, let's add it to archive
                        # If it dominates f_base, the archive update will remove f_base
                        # If it doesn't dominate, it adds a new point
                        add_to_archive(x_trial, f_trial)
                        # Update base for next local step? 
                        # If we found a better point, use it as new base
                        if np.all(f_trial <= f_base) and np.any(f_trial < f_base):
                            x_base = x_trial.copy()
                            f_base = f_trial.copy()

        # Return the global archive - all non-dominated points found during the run
        if archive_X:
            return np.array(archive_X)
        return np.array(pop_X)

# EVOLVE-BLOCK-END
