"""Multi-objective optimizer for RE21 with hypervolume-guided DE approach."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-driven multi-objective optimizer for RE21 (2 objectives, 4 vars)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        ref = np.array([3000.0, 0.06])

        # Track all evaluated points
        all_X = []
        all_F = []
        used = 0

        def evaluate(x):
            nonlocal used
            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=float)
            all_X.append(x.copy())
            all_F.append(f)
            used += 1
            return f

        def get_non_dominated(F_list):
            """Return indices of non-dominated points in F_list."""
            if len(F_list) == 0:
                return []
            F_arr = np.array(F_list, dtype=float)
            n = len(F_arr)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        dominated[i] = True
                        break
            return [i for i in range(n) if not dominated[i]]

        def is_dominated_by_front(f_new, front_indices):
            """Check if f_new is dominated by any point in front_indices."""
            for idx in front_indices:
                if np.all(all_F[idx] <= f_new) and np.any(all_F[idx] < f_new):
                    return True
            return False

        def compute_hv_contribution(front_pts, ref_pt):
            """Compute 2D hypervolume of front_pts w.r.t. ref_pt."""
            if len(front_pts) == 0:
                return 0.0
            pts = np.array(front_pts, dtype=float)
            mask = np.all(pts < ref_pt, axis=1)
            pts = pts[mask]
            if len(pts) == 0:
                return 0.0
            order = np.argsort(pts[:, 0])
            pts = pts[order]
            hv = 0.0
            prev_y = ref_pt[1]
            for i in range(len(pts)):
                x = pts[i, 0]
                y = pts[i, 1]
                if i + 1 < len(pts):
                    w0 = pts[i + 1, 0] - x
                else:
                    w0 = ref_pt[0] - x
                h1 = prev_y - y
                if h1 > 0 and w0 > 0:
                    hv += w0 * h1
                if y < prev_y:
                    prev_y = y
            return hv

        # Phase 1: Stratified initialization
        n_init = 30
        for i in range(n_init):
            if used >= self.budget or time.time() > deadline:
                break
            x = np.zeros(self.dim)
            for d in range(self.dim):
                segment = int(i * self.dim / n_init) % self.dim
                if d == segment:
                    frac = (i % (n_init // self.dim + 1)) / (n_init // self.dim + 1)
                    x[d] = lo[d] + frac * (hi[d] - lo[d])
                else:
                    x[d] = (lo[d] + hi[d]) / 2.0
            x = np.clip(x, lo, hi)
            evaluate(x)

        for i in range(10):
            if used >= self.budget or time.time() > deadline:
                break
            x = lo + (hi - lo) * np.random.random(self.dim)
            evaluate(x)

        # Phase 2: DE with cached front
        pop_size = min(18, self.budget - used)
        if pop_size < 4:
            while used < self.budget:
                if time.time() > deadline:
                    break
                x = lo + (hi - lo) * np.random.random(self.dim)
                evaluate(x)
            return all_X

        front_idx = get_non_dominated(all_F)
        pop = []
        pop_F = []

        for idx in front_idx[:pop_size // 2]:
            pop.append(all_X[idx].copy())
            pop_F.append(all_F[idx].copy())

        while len(pop) < pop_size and used < self.budget:
            if time.time() > deadline:
                break
            x = lo + (hi - lo) * np.random.random(self.dim)
            f = evaluate(x)
            pop.append(x)
            pop_F.append(f)

        if len(pop) == 0:
            while used < self.budget:
                if time.time() > deadline:
                    break
                x = lo + (hi - lo) * np.random.random(self.dim)
                evaluate(x)
            return all_X

        # DE loop
        max_generations = max(4, (self.budget - used) // (pop_size + 2))

        for gen in range(max_generations):
            if used >= self.budget or time.time() > deadline:
                break

            F_mut = 0.9 * (1.0 - gen / max(max_generations, 1)) + 0.1
            CR = 0.5

            # Cache the front once per generation
            current_front = get_non_dominated(all_F)

            for i in range(len(pop)):
                if used >= self.budget or time.time() > deadline:
                    break

                if len(pop) >= 3:
                    candidates = [j for j in range(len(pop)) if j != i]
                    a, b, c = np.random.choice(candidates, size=3, replace=False)
                    mutant = pop[a] + F_mut * (pop[b] - pop[c])
                else:
                    mutant = pop[i] + (hi - lo) * 0.1 * np.random.randn(self.dim)

                trial = pop[i].copy()
                for d in range(self.dim):
                    if np.random.random() < CR or d == np.random.randint(self.dim):
                        trial[d] = mutant[d]

                trial = np.clip(trial, lo, hi)
                f_trial = evaluate(trial)

                # Quick dominance check against cached front
                if not is_dominated_by_front(f_trial, current_front):
                    pop[i] = trial
                    pop_F[i] = f_trial
                    # Update front cache if this point might be non-dominated
                    # (defer full recomputation to next generation)

            # Refresh: replace worst 2 population members with random
            if gen % 4 == 3 and used < self.budget:
                for _ in range(2):
                    if used >= self.budget or time.time() > deadline:
                        break
                    x = lo + (hi - lo) * np.random.random(self.dim)
                    f = evaluate(x)
                    idx_refresh = np.random.randint(len(pop))
                    pop[idx_refresh] = x
                    pop_F[idx_refresh] = f

        # Phase 3: Hypervolume-guided local refinement
        front_idx = get_non_dominated(all_F)
        if len(front_idx) > 0:
            front_F = [all_F[i] for i in front_idx]

            # Compute HV contribution for each front point
            hv_contribs = []
            for i in range(len(front_F)):
                hv_with = compute_hv_contribution(front_F, ref)
                without = [f for j, f in enumerate(front_F) if j != i]
                hv_without = compute_hv_contribution(without, ref)
                hv_contribs.append(hv_with - hv_without)

            order = np.argsort(hv_contribs)[::-1]

            # Refine top HV-contributing points
            n_refine = min(10, len(order))
            for idx_order in range(n_refine):
                if used >= self.budget or time.time() > deadline:
                    break
                base_idx = front_idx[order[idx_order]]
                base_x = all_X[base_idx].copy()
                base_f = all_F[base_idx].copy()

                # Per-dimension coordinate descent with adaptive step
                for dim_i in range(self.dim):
                    if used >= self.budget or time.time() > deadline:
                        break
                    step_frac = 0.04
                    for _ in range(5):
                        if used >= self.budget or time.time() > deadline:
                            break
                        step_size = (hi[dim_i] - lo[dim_i]) * step_frac
                        improved = False
                        for direction in [-1, 1]:
                            if used >= self.budget or time.time() > deadline:
                                break
                            x_new = base_x.copy()
                            x_new[dim_i] += direction * step_size
                            x_new = np.clip(x_new, lo, hi)
                            f_new = evaluate(x_new)
                            if (f_new[0] < base_f[0] and f_new[1] <= base_f[1]) or \
                               (f_new[1] < base_f[1] and f_new[0] <= base_f[1]):
                                base_x = x_new.copy()
                                base_f = f_new.copy()
                                improved = True
                                break
                        if not improved:
                            step_frac *= 0.5
                        if step_frac < 0.002:
                            break

        # Phase 4: Targeted exploration near front
        remaining = self.budget - used
        if remaining > 0:
            front_idx = get_non_dominated(all_F)
            for _ in range(remaining):
                if used >= self.budget or time.time() > deadline:
                    break
                if len(front_idx) > 0 and np.random.random() < 0.6:
                    base_idx = front_idx[np.random.randint(len(front_idx))]
                    base_x = all_X[base_idx]
                    scale = 0.05 + 0.1 * np.random.random()
                    x = base_x + scale * (hi - lo) * np.random.randn(self.dim)
                else:
                    x = lo + (hi - lo) * np.random.random(self.dim)
                x = np.clip(x, lo, hi)
                evaluate(x)

        return all_X
# EVOLVE-BLOCK-END
