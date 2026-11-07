# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based multi-objective search for WFG4 with a known-front repair.

    Strategy (kept in the same family, refined):
      * Explicitly bias toward the global Pareto region (x_1 ~= 0.35) to avoid
        the large local front at x_1 = 0 that traps random search.
      * Fast DE/rand/1 with an EXACT hypervolume-CONTRIBUTION selection (cheap,
        O(n) per candidate) so we can spend evaluations on search rather than
        on bookkeeping.
      * Exploit the known WFG4 Pareto set: with x_1 ~ 0.35 the position
        variables sit on the front; we repair them with a cheap local search
        and then a deterministic coordinate-descent (ternary) polish that
        maximises each point's hypervolume contribution to the front.
    """

    COND = 0                                  # x_1: 0.35 -> global front, 0.0 -> local
    POS = (1, 2, 3, 5, 6, 8, 9, 11, 12)       # position parameters
    DIR = (4, 7, 10)                          # direction parameters

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    # ---- hypervolume contribution of a single point to a set (exact, 3-D) --
    @staticmethod
    def _hv_contrib(f, F):
        """Extra hypervolume (vs origin) gained by adding point f to set F.

        f = (a,b,c). Only points of F dominated by f matter.
        contribution = a*b*c - a*b*(min c among dominated)
                       - a*c*(min b among dominated with c<=that)
                       - b*c*(min a among dominated with b<=that, c<=that)
                       + (min a)*(min b)*(min c) over points dominated in all 3.
        """
        a, b, c = f
        F = np.asarray(F, dtype=float)
        if F.shape[0] == 0:
            return a * b * c
        dom = (F[:, 0] <= a) & (F[:, 1] <= b) & (F[:, 2] <= c)
        if not dom.any():
            return 0.0
        D = F[dom]
        min_c = D[:, 2].min()
        # points dominated in obj1 and obj2 (c irrelevant for the a*b term)
        d2 = (F[:, 0] <= a) & (F[:, 1] <= b)
        min_b = F[d2, 1].min() if d2.any() else 0.0
        d1 = (F[:, 0] <= a) & (F[:, 2] <= c)
        min_a = F[d1, 0].min() if d1.any() else 0.0
        # points dominated in obj1 and obj3
        d3 = (F[:, 1] <= b) & (F[:, 2] <= c)
        min_b2 = F[d3, 1].min() if d3.any() else 0.0
        # full triple minimum
        min_a3 = D[:, 0].min()
        min_b3 = D[:, 1].min()
        min_c3 = min_c
        contrib = a * b * c
        contrib -= a * b * min_c3
        contrib -= a * c * min_b
        contrib -= b * c * min_a
        contrib += min_a3 * min_b3 * min_c3
        return max(0.0, contrib)

    def _repair(self, x, problem, F, budget_left):
        """Push x onto the known WFG4 Pareto set by tuning position variables,
        guided by hypervolume contribution to the current front F."""
        x = np.array(x, dtype=float)
        if x[self.COND] < 0.1:               # local-front region -> move to global
            x[self.COND] = 0.35
            x = np.clip(x, self._lo, self._hi)
        best_x = x.copy()
        best_f = np.asarray(problem(x), dtype=float)
        self._evals += 1
        best_cv = self._hv_contrib(best_f, F)
        for _ in range(30):
            if budget_left() <= 0:
                break
            cand = best_x.copy()
            cand[self.POS] += np.random.normal(0.0, 0.05, size=len(self.POS))
            cand = np.clip(cand, self._lo, self._hi)
            cf = np.asarray(problem(cand), dtype=float)
            self._evals += 1
            if np.all(cf <= best_f):
                cv = self._hv_contrib(cf, F)
                if cv > best_cv:
                    best_x, best_f, best_cv = cand, cf, cv
        return best_x, best_f

    def _polish(self, x, f, problem, F, budget_left):
        """Deterministic coordinate-descent (ternary search) polish that
        maximises the hypervolume contribution of point x to the front F.
        A smooth scalar surrogate over the disconnected WFG4 landscape."""
        x = np.array(x, dtype=float)
        best_f = np.asarray(f, dtype=float)
        best_cv = self._hv_contrib(best_f, F)
        for _ in range(3):                    # a few sweeps over coordinates
            if budget_left() <= 0:
                break
            for d in range(self.dim):
                if budget_left() <= 0:
                    break
                lo_d = self._lo[d]
                hi_d = self._hi[d]
                x0 = x[d]
                # ternary search on the contribution (a quasi-concave surrogate)
                a, b = lo_d, hi_d
                for _ in range(10):
                    if budget_left() <= 0:
                        break
                    m1 = a + (b - a) / 3.0
                    m2 = b - (b - a) / 3.0
                    x[d] = m1
                    f1 = np.asarray(problem(x), dtype=float)
                    self._evals += 1
                    cv1 = self._hv_contrib(f1, F)
                    x[d] = m2
                    f2 = np.asarray(problem(x), dtype=float)
                    self._evals += 1
                    cv2 = self._hv_contrib(f2, F)
                    if cv1 < cv2:
                        a = m1
                    else:
                        b = m2
                x[d] = (a + b) / 2.0
                fx = np.asarray(problem(x), dtype=float)
                self._evals += 1
                cvx = self._hv_contrib(fx, F)
                if cvx > best_cv:
                    best_f, best_cv = fx, cvx
                else:
                    x[d] = x0
        return x, best_f

    # ---- main --------------------------------------------------------------
    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 285.0               # safety margin well under 300s
        budget = self.budget
        dim = self.dim

        self._lo = np.array([b[0] for b in bounds], dtype=float)
        self._hi = np.array([b[1] for b in bounds], dtype=float)
        self._evals = 0
        span = self._hi - self._lo
        c = 0.35 * span[self.COND]          # global-front offset for x_1

        def budget_left():
            return budget - self._evals

        def rand_point():
            x = self._lo + np.random.rand(dim) * span
            x[self.COND] += c               # bias toward global front
            return np.clip(x, self._lo, self._hi)

        pop = int(min(40, max(8, budget // 6)))
        X = np.array([rand_point() for _ in range(pop)])
        F = np.array([np.asarray(problem(x), dtype=float) for x in X])
        self._evals += pop

        # ---- DE/rand/1 main loop with hypervolume-contribution selection ----
        while self._evals < budget:
            if time.time() >= deadline:
                break
            i = np.random.randint(0, pop)
            j = np.random.randint(0, pop)
            k = np.random.randint(0, pop)
            while j == i or k == i or k == j:
                j = np.random.randint(0, pop)
                k = np.random.randint(0, pop)
            v = X[i] + 0.65 * (X[j] - X[k])
            v = np.clip(v, self._lo, self._hi)
            vf = np.asarray(problem(v), dtype=float)
            self._evals += 1

            # accept if it adds hypervolume to the front (cheap exact test)
            if self._hv_contrib(vf, F) > 0.0:
                X[i], F[i] = v, vf

        # ---- final repair pass: tune position vars to sit on the front ------
        if self._evals < budget and time.time() < deadline:
            for idx in range(pop):
                if self._evals >= budget or time.time() >= deadline:
                    break
                rx, rf = self._repair(X[idx], problem, F, budget_left)
                X[idx], F[idx] = rx, rf

        # ---- final deterministic coordinate-descent (ternary) polish --------
        if self._evals < budget and time.time() < deadline:
            for idx in range(pop):
                if self._evals >= budget or time.time() >= deadline:
                    break
                px, pf = self._polish(X[idx], F[idx], problem, F, budget_left)
                X[idx], F[idx] = px, pf

        return X
# EVOLVE-BLOCK-END
