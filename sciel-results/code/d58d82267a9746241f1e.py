# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution based multi-objective optimizer for RE34."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _hv_3d(self, pts, ref):
        """Exact hypervolume for 3D points."""
        n = len(pts)
        if n == 0:
            return 0.0
        mask = np.all(pts < ref, axis=1)
        pts = pts[mask]
        n = len(pts)
        if n == 0:
            return 0.0
        if n == 1:
            return float(np.prod(ref - pts[0]))

        # Sort by x
        idx = np.argsort(pts[:, 0])
        pts = pts[idx]

        hv = 0.0
        for i in range(n):
            x_cur = pts[i, 0]
            x_next = pts[i + 1, 0] if i + 1 < n else ref[0]
            yz = pts[:i + 1, 1:3]
            hv += (x_next - x_cur) * self._hv_2d(yz, ref[1:3])
        return hv

    def _hv_2d(self, pts, ref):
        """Exact hypervolume for 2D points."""
        n = len(pts)
        if n == 0:
            return 0.0
        if n == 1:
            return float((ref[0] - pts[0, 0]) * (ref[1] - pts[0, 1]))

        idx = np.argsort(pts[:, 0])
        pts = pts[idx]

        hv = 0.0
        y_min = ref[1]
        for i in range(n - 1, -1, -1):
            if pts[i, 1] < y_min:
                y_min = pts[i, 1]
                x_cur = pts[i, 0]
                x_next = pts[i + 1, 0] if i + 1 < n else ref[0]
                hv += (x_next - x_cur) * (ref[1] - y_min)
        return hv

    def _is_pareto(self, F):
        """Return boolean mask of non-dominated points."""
        n = len(F)
        if n == 0:
            return np.array([], dtype=bool)
        pareto = np.ones(n, dtype=bool)
        for i in range(n):
            if not pareto[i]:
                continue
            others = np.delete(F, i, axis=0)
            if len(others) == 0:
                continue
            dominated = np.any(np.all(others <= F[i], axis=1) & np.any(others < F[i], axis=1))
            if dominated:
                pareto[i] = False
        return pareto

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        ref = np.array([1705.0, 12.0, 0.3])

        rng = np.random

        # Population size for DE
        pop_size = 20
        n_generations = (budget - pop_size) // pop_size  # ~9 generations with 200 budget

        # --- Phase 1: Initial population ---
        # Use a mix of LHS and random for diversity
        n_lhs = 15
        n_rand = pop_size - n_lhs

        lhs_X = np.zeros((n_lhs, dim))
        for d in range(dim):
            perm = np.random.permutation(n_lhs)
            lhs_X[:, d] = lo[d] + (hi[d] - lo[d]) * (perm + np.random.uniform(size=n_lhs)) / n_lhs
        lhs_X = np.clip(lhs_X, lo, hi)

        rand_X = lo + np.random.uniform(size=(n_rand, dim)) * (hi - lo)
        rand_X = np.clip(rand_X, lo, hi)

        X_pop = np.vstack([lhs_X, rand_X])
        F_pop = np.array([problem(x) for x in X_pop])
        used = pop_size

        all_X = X_pop.copy()
        all_F = F_pop.copy()

        # Build initial Pareto archive
        valid = np.all(F_pop < ref, axis=1)
        if np.any(valid):
            pF = F_pop[valid]
            pX = X_pop[valid]
            p_mask = self._is_pareto(pF)
            pareto_X = pX[p_mask].copy()
            pareto_F = pF[p_mask].copy()
        else:
            pareto_X = np.empty((0, dim))
            pareto_F = np.empty((0, n_obj))

        # --- Phase 2: Differential Evolution ---
        F_de = 0.8  # Differential weight
        CR = 0.9    # Crossover rate

        gen = 0
        while used < budget and gen < n_generations:
            if time.time() > deadline - 10:
                break

            gen += 1
            X_new = np.zeros_like(X_pop)
            F_new = np.zeros_like(F_pop)

            for i in range(pop_size):
                if used >= budget or time.time() > deadline - 5:
                    break

                # Select 3 distinct random indices different from i
                candidates = list(range(pop_size))
                candidates.remove(i)
                a, b, c = np.random.choice(candidates, size=3, replace=False)

                # DE/rand/1/bin
                v = X_pop[a] + F_de * (X_pop[b] - X_pop[c])

                # Binary crossover
                r = np.random.random()
                j_rand = np.random.randint(dim)
                trial = np.where(
                    np.random.random(dim) <= CR,
                    v,
                    X_pop[i]
                )
                trial[j_rand] = v[j_rand]
                trial = np.clip(trial, lo, hi)

                ft = problem(trial)
                used += 1

                # Selection: keep the better of parent and trial
                # Compare using dominance; if non-dominated, use crowding or random
                if np.all(ft <= F_pop[i]) and np.any(ft < F_pop[i]):
                    # Trial dominates parent
                    X_new[i] = trial
                    F_new[i] = ft
                elif np.all(F_pop[i] <= ft) and np.any(F_pop[i] < ft):
                    # Parent dominates trial
                    X_new[i] = X_pop[i]
                    F_new[i] = F_pop[i]
                else:
                    # Non-dominated: keep the one with better HV contribution
                    # Simple heuristic: keep the one closer to reference in sum
                    if np.sum(ft) < np.sum(F_pop[i]):
                        X_new[i] = trial
                        F_new[i] = ft
                    else:
                        X_new[i] = X_pop[i]
                        F_new[i] = F_pop[i]

                all_X = np.vstack([all_X, [X_new[i]]])
                all_F = np.vstack([all_F, [F_new[i]]])

            X_pop = X_new
            F_pop = F_new

            # Update Pareto archive with new population
            valid_new = np.all(F_pop < ref, axis=1)
            if np.any(valid_new):
                new_pF = F_pop[valid_new]
                new_pX = X_pop[valid_new]
                # Merge with existing archive
                if len(pareto_F) > 0:
                    combined_F = np.vstack([pareto_F, new_pF])
                    combined_X = np.vstack([pareto_X, new_pX])
                else:
                    combined_F = new_pF
                    combined_X = new_pX

                p_mask = self._is_pareto(combined_F)
                pareto_X = combined_X[p_mask]
                pareto_F = combined_F[p_mask]

            # Prune archive if too large
            if len(pareto_F) > 30:
                valid_prune = np.all(pareto_F < ref, axis=1)
                if np.sum(valid_prune) > 30:
                    pf_valid = pareto_F[valid_prune]
                    px_valid = pareto_X[valid_prune]
                    # Greedy diversity selection
                    dists = np.sum(np.abs(pf_valid - ref), axis=1)
                    selected = [np.argmin(dists)]
                    remaining = set(range(len(pf_valid))) - set(selected)
                    while len(selected) < 30 and remaining:
                        best_idx = None
                        best_min_dist = -1
                        for idx in remaining:
                            min_dist = np.min(np.linalg.norm(pf_valid[idx] - pf_valid[selected], axis=1))
                            if min_dist > best_min_dist:
                                best_min_dist = min_dist
                                best_idx = idx
                        if best_idx is None:
                            break
                        selected.append(best_idx)
                        remaining.remove(best_idx)
                    pareto_X = px_valid[np.array(selected)]
                    pareto_F = pf_valid[np.array(selected)]

        # --- Phase 3: Local refinement on best points ---
        if used < budget and len(pareto_F) > 0:
            valid_final = np.all(pareto_F < ref, axis=1)
            if np.any(valid_final):
                pX_final = pareto_X[valid_final]
                pF_final = pareto_F[valid_final]
            else:
                pX_final = pareto_X
                pF_final = pareto_F

            if len(pX_final) > 0:
                sums = np.sum(pF_final, axis=1)
                top_k = min(3, len(pX_final))
                top_idx = np.argsort(sums)[:top_k]

                for pi in top_idx:
                    if used >= budget or time.time() > deadline - 3:
                        break
                    parent = pX_final[pi].copy()

                    # Coordinate-wise local search
                    for coord in range(dim):
                        if used >= budget or time.time() > deadline - 2:
                            break

                        # Try small perturbations
                        for delta in [-0.05, -0.02, 0.02, 0.05]:
                            if used >= budget or time.time() > deadline - 2:
                                break
                            trial = parent.copy()
                            trial[coord] = np.clip(
                                parent[coord] + delta * (hi[coord] - lo[coord]),
                                lo[coord], hi[coord]
                            )
                            if np.allclose(trial, parent):
                                continue
                            ft = problem(trial)
                            used += 1

                            # Check if ft improves parent in some sense
                            if np.all(ft <= pF_final[pi]) and np.any(ft < pF_final[pi]):
                                parent = trial
                                pF_final[pi] = ft
                                pX_final[pi] = trial

                                # Update archive
                                if np.all(ft < ref):
                                    dominated = False
                                    for pf in pareto_F:
                                        if np.all(pf <= ft) and np.any(pf < ft):
                                            dominated = True
                                            break
                                    if not dominated:
                                        pareto_X = np.vstack([pareto_X, [parent]])
                                        pareto_F = np.vstack([pareto_F, [ft]])
                                        keep = np.ones(len(pareto_F), dtype=bool)
                                        for i in range(len(pareto_F) - 1):
                                            if np.all(pareto_F[i] >= ft) and np.any(pareto_F[i] > ft):
                                                keep[i] = False
                                        pareto_X = pareto_X[keep]
                                        pareto_F = pareto_F[keep]

        return all_X

# EVOLVE-BLOCK-END
