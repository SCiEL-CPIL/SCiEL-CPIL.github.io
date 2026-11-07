import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        ref = np.array([1.1, 1.3, 1.1])  # scoring reference point

        eval_count = 0
        max_budget = self.budget

        def time_ok():
            return time.time() < deadline

        def evaluate(x):
            nonlocal eval_count
            if eval_count >= max_budget or not time_ok():
                return None
            eval_count += 1
            try:
                return np.asarray(problem(x), dtype=float).reshape(-1)[:n_obj]
            except Exception:
                return None

        # ---------- Pareto / hypervolume helpers (3 objectives) ----------
        def non_dominated_indices(F):
            F = np.asarray(F)
            n = len(F)
            if n == 0:
                return np.array([], dtype=int)
            mask = np.ones(n, dtype=bool)
            for i in range(n):
                if not mask[i]:
                    continue
                for j in range(n):
                    if i == j or not mask[j]:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        mask[i] = False
                        break
            return np.where(mask)[0]

        def hypervolume_3d(F, ref):
            """Exact 3D hypervolume of the non-dominated set F w.r.t. ref.
            Only points strictly better than ref in every objective count."""
            F = np.asarray(F, dtype=float)
            if len(F) == 0:
                return 0.0
            # keep points strictly better than ref in all objectives
            keep = np.all(F < ref, axis=1)
            F = F[keep]
            if len(F) == 0:
                return 0.0
            # non-dominated subset
            idx = non_dominated_indices(F)
            F = F[idx]
            n = len(F)
            if n == 0:
                return 0.0

            # Sort by objective 0 ascending
            order = np.argsort(F[:, 0])
            P = F[order]
            total = 0.0
            for i in range(n):
                p = P[i]
                # width in obj0: from p[0] to the next point's obj0 (or ref[0])
                if i + 1 < n:
                    next0 = P[i + 1][0]
                else:
                    next0 = ref[0]
                w0 = next0 - p[0]
                if w0 <= 0:
                    continue
                # 2D area in (obj1, obj2) plane of [p1, ref1] x [p2, ref2]
                # minus union of rectangles from points j with P[j][0] < p[0]
                # that also satisfy P[j][1] < p[1] and P[j][2] < p[2]
                p1, p2 = p[1], p[2]
                r1, r2 = ref[1], ref[2]
                area = (r1 - p1) * (r2 - p2)
                # collect blocking rectangles from earlier (smaller obj0) points
                rects = []
                for j in range(i):
                    q = P[j]
                    if q[1] < p1 and q[2] < p2:
                        # rectangle [q1, r1] x [q2, r2] but we only care about
                        # the part inside [p1, r1] x [p2, r2]
                        a1 = max(q[1], p1)
                        a2 = max(q[2], p2)
                        b1 = min(r1, p1)  # = p1
                        b2 = min(r2, p2)  # = p2
                        # rectangle to subtract: [a1, p1] x [a2, p2]
                        if a1 < p1 and a2 < p2:
                            rects.append((a1, a2, p1, p2))
                # subtract union area of these rectangles (they all share top-right corner (p1,p2))
                # Union of rectangles all anchored at (p1, p2) with varying bottom-left
                # = area of the bounding box determined by min a1 and min a2
                if rects:
                    min_a1 = min(r[0] for r in rects)
                    min_a2 = min(r[1] for r in rects)
                    # Since all rectangles are [a1,p1]x[a2,p2], their union is
                    # [min a1, p1] x [min a2, p2]
                    union = (p1 - min_a1) * (p2 - min_a2)
                    area -= union
                if area > 0:
                    total += w0 * area
            return total

        # ---------- hypervolume-guided local search ----------
        def local_search_hv(x_start, f_start, nd_F, n_steps):
            """Local search that accepts a move if it improves the hypervolume
            of the non-dominated set (nd_F plus the candidate)."""
            x_best = x_start.copy()
            f_best = f_start.copy()
            # current hypervolume baseline (nd_F may already include f_start)
            base_hv = hypervolume_3d(nd_F, ref)

            for step in range(n_steps):
                if not time_ok() or eval_count >= max_budget:
                    break
                scale = 0.12 * (1.0 - step / float(max(1, n_steps)))
                x_new = x_best + scale * np.random.standard_normal(dim)
                x_new = np.clip(x_new, lo, hi)
                f_new = evaluate(x_new)
                if f_new is None:
                    break
                # candidate non-dominated set: nd_F plus f_new (drop dominated)
                cand = np.vstack([nd_F, f_new])
                hv = hypervolume_3d(cand, ref)
                if hv > base_hv + 1e-12:
                    # accept: update best and update baseline to the new nd set
                    x_best = x_new
                    f_best = f_new
                    # recompute non-dominated set of cand to keep baseline consistent
                    nd_idx = non_dominated_indices(cand)
                    nd_F = cand[nd_idx]
                    base_hv = hv
            return x_best, f_best, nd_F

        # ---------- 1. LHS initialization ----------
        n_initial = 45
        if n_initial > max_budget:
            n_initial = max_budget
        samples = np.zeros((n_initial, dim))
        for d in range(dim):
            perm = np.random.permutation(n_initial)
            samples[:, d] = (perm + np.random.random(n_initial)) / n_initial
        samples = lo + (hi - lo) * samples

        X = []
        F = []
        for x in samples:
            if not time_ok():
                break
            f = evaluate(x)
            if f is None:
                break
            X.append(x)
            F.append(f)
        X = np.array(X)
        F = np.array(F)

        # ---------- 2. Build initial non-dominated set ----------
        nd_idx = non_dominated_indices(F)
        nd_F = F[nd_idx].copy() if len(nd_idx) > 0 else F.copy()
        # seed points: the non-dominated ones
        seed_idx = nd_idx if len(nd_idx) > 0 else np.arange(len(F))

        # ---------- 3. Hypervolume-guided local refinement ----------
        remaining = max_budget - eval_count
        if len(seed_idx) > 0:
            steps_per_point = max(3, remaining // (len(seed_idx) + 1))
        else:
            steps_per_point = 0

        refined_X = []
        refined_F = []
        for idx in seed_idx:
            if not time_ok() or eval_count >= max_budget:
                break
            x_start = X[idx]
            f_start = F[idx]
            x_ref, f_ref, nd_F = local_search_hv(x_start, f_start, nd_F, steps_per_point)
            refined_X.append(x_ref)
            refined_F.append(f_ref)

        # ---------- 4. Random restarts with hypervolume-guided refinement ----------
        remaining = max_budget - eval_count
        n_restarts = min(12, remaining // 6)
        for _ in range(n_restarts):
            if not time_ok() or eval_count >= max_budget:
                break
            x_rand = lo + np.random.random(dim) * (hi - lo)
            f_rand = evaluate(x_rand)
            if f_rand is None:
                break
            x_ref, f_ref, nd_F = local_search_hv(x_rand, f_rand, nd_F, 6)
            refined_X.append(x_ref)
            refined_F.append(f_ref)

        # ---------- 5. Final non-dominated set ----------
        if len(refined_F) > 0:
            refined_F = np.array(refined_F)
            refined_X = np.array(refined_X)
            final_nd = non_dominated_indices(refined_F)
            final_X = refined_X[final_nd]
            final_F = refined_F[final_nd]
        else:
            final_X = np.array([lo + 0.5 * (hi - lo)])
            f0 = evaluate(final_X[0])
            final_F = np.array([f0]) if f0 is not None else np.array([np.zeros(n_obj)])

        if len(final_X) > 0:
            return final_X[0]
        return X[0]
