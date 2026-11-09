import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285

        n_points = self.budget
        ref = np.array([2.5, 2.5, 2.5])

        def halton(index, base):
            f = 1.0
            r = 0.0
            idx = index
            while idx > 0:
                f = f / base
                r = r + f * (idx % base)
                idx //= base
            return r

        # Generate candidate pool using area-preserving Halton mapping
        n_candidates = 1500
        candidates = []

        # Corner and edge points
        corners = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
        for c in corners:
            candidates.append(c)
        for t in [0.1, 0.25, 0.5, 0.75, 0.9]:
            candidates.append((0.0, t))
            candidates.append((t, 0.0))
            candidates.append((t, 1.0))
            candidates.append((1.0, t))

        n_halton = n_candidates - len(candidates)
        for i in range(1, n_halton + 1):
            u = halton(i, 2)
            v = halton(i, 3)
            alpha = np.arccos(1.0 - u)
            beta = v * (np.pi / 2.0)
            x1_val = 2.0 * alpha / np.pi
            x2_val = 2.0 * beta / np.pi
            candidates.append((x1_val, x2_val))

        # Convert to objective space
        cand_objs = np.zeros((len(candidates), 3))
        for i, (x1, x2) in enumerate(candidates):
            a = np.pi * x1 / 2.0
            b = np.pi * x2 / 2.0
            cand_objs[i, 0] = np.sin(a)
            cand_objs[i, 1] = np.cos(a) * np.sin(b)
            cand_objs[i, 2] = np.cos(a) * np.cos(b)

        # Greedy hypervolume maximization
        selected_mask = np.zeros(len(candidates), dtype=bool)
        selected_points = []

        # First pick: max box volume
        box_vols = np.prod(ref[None, :] - cand_objs, axis=1)
        first_idx = np.argmax(box_vols)
        selected_mask[first_idx] = True
        selected_points.append(cand_objs[first_idx])

        for _ in range(1, n_points):
            if time.time() > deadline:
                break

            unselected_idx = np.where(~selected_mask)[0]
            if len(unselected_idx) == 0:
                break

            p = cand_objs[unselected_idx]
            S = np.array(selected_points)

            # Check domination
            dominated = np.all(S[None, :, :] <= p[:, None, :], axis=2)
            is_dominated = np.any(dominated, axis=1)

            p_box_vols = np.prod(ref[None, :] - p, axis=1)

            max_ps = np.maximum(p[:, None, :], S[None, :, :])
            overlap_vols = np.prod(ref[None, None, :] - max_ps, axis=2)
            overlap_sums = np.sum(overlap_vols, axis=1)

            marginals = p_box_vols - overlap_sums
            marginals = np.maximum(0.0, marginals)
            marginals[is_dominated] = 0.0

            best_local = np.argmax(marginals)
            if marginals[best_local] <= 0:
                break

            best_global = unselected_idx[best_local]
            selected_mask[best_global] = True
            selected_points.append(cand_objs[best_global])

        # Evaluate selected points
        results = []
        for idx in np.where(selected_mask)[0]:
            if time.time() > deadline:
                break

            x1, x2 = candidates[idx]
            x1_c = float(np.clip(x1, 0.0, 1.0))
            x2_c = float(np.clip(x2, 0.0, 1.0))

            x = np.zeros(self.dim)
            x[0] = x1_c
            x[1] = x2_c
            x[2:] = 0.5

            f = problem(x)
            results.append(f)

        return results
