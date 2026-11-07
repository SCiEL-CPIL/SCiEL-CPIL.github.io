import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        def elapsed():
            return time.time() - t0

        def in_time():
            return time.time() < deadline

        def dominates(a, b):
            """Return True if a dominates b (a <= b in all objectives, a < b in at least one)."""
            return np.all(a <= b) and np.any(a < b)

        def is_pareto(F):
            """Return boolean mask of non-dominated points in F (n x n_obj)."""
            n = len(F)
            mask = np.ones(n, dtype=bool)
            for i in range(n):
                if not mask[i]:
                    continue
                for j in range(n):
                    if i == j or not mask[j]:
                        continue
                    if dominates(F[j], F[i]):
                        mask[i] = False
                        break
            return mask

        # Phase 1: Grid-based decomposition
        # Partition 4D space into a 2x2x2x2 grid = 16 cells
        # Use remaining budget for local simplex in each cell
        n_per_dim = 2
        n_cells = n_per_dim ** dim  # 16 cells

        # Reserve budget: 1 initial eval per cell + simplex iterations
        # Total budget = 200. Use 16 for initial, rest for simplex
        initial_budget_per_cell = 1
        remaining = budget - n_cells * initial_budget_per_cell
        # Each simplex step uses (dim+1) evaluations to build simplex, then 1 per iteration
        # Be conservative: limit simplex iterations
        max_simplex_iter_per_cell = max(2, remaining // (n_cells * (dim + 2)))

        all_X = []
        all_F = []
        used = 0

        # Generate cell centers and corners
        cell_centers = []
        for i in range(n_cells):
            coords = []
            idx = i
            for d in range(dim):
                n_per_d = n_per_dim
                c = idx % n_per_d
                idx //= n_per_d
                lo_d = lo[d] + c * (hi[d] - lo[d]) / n_per_d
                hi_d = lo[d] + (c + 1) * (hi[d] - lo[d]) / n_per_d
                coords.append((lo_d + hi_d) / 2.0)
            cell_centers.append(np.array(coords))

        # Phase 1: Evaluate cell centers
        for c in range(n_cells):
            if used >= budget or not in_time():
                break
            x = cell_centers[c]
            f = np.array(problem(x), dtype=np.float64)
            all_X.append(x.copy())
            all_F.append(f)
            used += 1

        # Phase 2: Local Nelder-Mead simplex in each cell
        for c in range(n_cells):
            if used >= budget or not in_time():
                break

            # Get current best in this cell region (just use the center as starting point)
            start_x = cell_centers[c].copy()
            start_f = np.array(problem(start_x), dtype=np.float64)
            all_X.append(start_x.copy())
            all_F.append(start_f)
            used += 1

            if used >= budget or not in_time():
                break

            # Build simplex: dim+1 points
            # Start with start_x, then perturb each dimension
            simplex_X = [start_x.copy()]
            simplex_F = [start_f.copy()]

            # Scale perturbations based on cell size
            cell_size = (hi - lo) / n_per_dim
            for d in range(dim):
                if used >= budget or not in_time():
                    break
                x_new = start_x.copy()
                # Perturb in dimension d
                step = 0.1 * cell_size[d]
                x_new[d] += step
                x_new = np.clip(x_new, lo, hi)
                f_new = np.array(problem(x_new), dtype=np.float64)
                simplex_X.append(x_new)
                simplex_F.append(f_new)
                used += 1

            # Simplex iterations
            for it in range(max_simplex_iter_per_cell):
                if used >= budget or not in_time():
                    break

                # Sort simplex by first objective (use a weighted sum for ordering)
                # Use lexicographic: sort by obj0, then obj1
                order = np.lexsort((simplex_F[:, 1], simplex_F[:, 0]))
                simplex_X = [simplex_X[i] for i in order]
                simplex_F = [simplex_F[i] for i in order]

                # Find worst point (last in sorted order)
                worst_idx = len(simplex_X) - 1
                worst_x = simplex_X[worst_idx]
                worst_f = simplex_F[worst_idx]

                # Centroid of all but worst
                centroid = np.mean(simplex_X[:-1], axis=0)

                # Reflection
                alpha = 1.0
                x_ref = centroid + alpha * (centroid - worst_x)
                x_ref = np.clip(x_ref, lo, hi)
                f_ref = np.array(problem(x_ref), dtype=np.float64)
                used += 1

                if used >= budget or not in_time():
                    break

                # Check if reflection is good
                # Compare f_ref against best and second-worst
                best_f = simplex_F[0]
                second_worst_f = simplex_F[-2]

                if np.all(f_ref <= best_f):
                    # Expansion
                    gamma = 2.0
                    x_exp = centroid + gamma * (centroid - worst_x)
                    x_exp = np.clip(x_exp, lo, hi)
                    f_exp = np.array(problem(x_exp), dtype=np.float64)
                    used += 1
                    if used >= budget or not in_time():
                        break
                    if np.all(f_exp <= f_ref):
                        simplex_X[worst_idx] = x_exp
                        simplex_F[worst_idx] = f_exp
                        all_X.append(x_exp.copy())
                        all_F.append(f_exp.copy())
                    else:
                        simplex_X[worst_idx] = x_ref
                        simplex_F[worst_idx] = f_ref
                        all_X.append(x_ref.copy())
                        all_F.append(f_ref.copy())
                elif np.all(f_ref < second_worst_f):
                    # Accept reflection
                    simplex_X[worst_idx] = x_ref
                    simplex_F[worst_idx] = f_ref
                    all_X.append(x_ref.copy())
                    all_F.append(f_ref.copy())
                else:
                    # Contraction
                    rho = 0.5
                    x_con = centroid + rho * (worst_x - centroid)
                    x_con = np.clip(x_con, lo, hi)
                    f_con = np.array(problem(x_con), dtype=np.float64)
                    used += 1
                    if used >= budget or not in_time():
                        break
                    if np.all(f_con < worst_f):
                        simplex_X[worst_idx] = x_con
                        simplex_F[worst_idx] = f_con
                        all_X.append(x_con.copy())
                        all_F.append(f_con.copy())
                    else:
                        # Shrink
                        for i in range(1, len(simplex_X)):
                            simplex_X[i] = simplex_X[0] + 0.5 * (simplex_X[i] - simplex_X[0])
                            simplex_X[i] = np.clip(simplex_X[i], lo, hi)
                            f_shrink = np.array(problem(simplex_X[i]), dtype=np.float64)
                            simplex_F[i] = f_shrink
                            used += 1
                            all_X.append(simplex_X[i].copy())
                            all_F.append(f_shrink.copy())
                            if used >= budget or not in_time():
                                break

        # Phase 3: Use remaining budget for targeted refinement
        # Sort all points by first objective, keep diverse set
        if used < budget and in_time() and len(all_F) > 0:
            F_arr = np.array(all_F)
            X_arr = np.array(all_X)

            # Keep non-dominated points
            mask = is_pareto(F_arr)
            if np.any(mask):
                X_nd = X_arr[mask]
                F_nd = F_arr[mask]
            else:
                X_nd = X_arr
                F_nd = F_arr

            # Use remaining budget to perturb best points
            n_nd = len(X_nd)
            remaining_budget = budget - used
            if n_nd > 0 and remaining_budget > 0:
                # Perturb each non-dominated point
                for i in range(n_nd):
                    if used >= budget or not in_time():
                        break
                    x_base = X_nd[i]
                    # Try several small perturbations
                    n_perturb = min(3, remaining_budget)
                    for p in range(n_perturb):
                        if used >= budget or not in_time():
                            break
                        direction = np.random.randn(dim)
                        direction /= (np.linalg.norm(direction) + 1e-12)
                        for scale in [0.05, 0.02]:
                            if used >= budget or not in_time():
                                break
                            x_new = x_base + scale * (hi - lo) * direction
                            x_new = np.clip(x_new, lo, hi)
                            f_new = np.array(problem(x_new), dtype=np.float64)
                            all_X.append(x_new.copy())
                            all_F.append(f_new)
                            used += 1

        # Final: return all evaluated points
        if len(all_X) == 0:
            # Fallback: random point
            x = (lo + hi) / 2.0
            f = np.array(problem(x), dtype=np.float64)
            all_X.append(x)
            all_F.append(f)

        return np.array(all_X)
