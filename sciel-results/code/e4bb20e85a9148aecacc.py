# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for ZDT3 (2 objectives)."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    # ---------- helper operators ----------
    def _sbx(self, a, b, eta):
        # Simulated binary crossover, returns two children
        c1 = np.empty_like(a)
        c2 = np.empty_like(a)
        r = np.random.rand(self.dim)
        for i in range(self.dim):
            if r[i] <= 0.5:
                beta = (2 * r[i]) ** (1.0 / (eta + 1))
            else:
                beta = (1.0 / (2 * (1 - r[i]))) ** (1.0 / (eta + 1))
            c1[i] = 0.5 * ((1 + beta) * a[i] + (1 - beta) * b[i])
            c2[i] = 0.5 * ((1 - beta) * a[i] + (1 + beta) * b[i])
        return c1, c2

    def _polymut(self, x, lo, hi, eta):
        c = x.copy()
        for i in range(self.dim):
            if np.random.rand() < 1.0 / self.dim:
                delta = np.random.rand()
                if delta < 0.5:
                    val = (2 * delta + 1 - (1 - delta) ** (eta + 1))
                else:
                    val = (1 - 2 * (1 - delta) + (2 * (1 - delta)) ** (eta + 1))
                c[i] = x[i] + (hi[i] - lo[i]) * (val - 1)
        return np.clip(c, lo, hi)

    def _fast_nondominated(self, F):
        n = len(F)
        S = [set() for _ in range(n)]
        nd = [0] * n
        front = [set()]
        for p in range(n):
            for q in range(n):
                if p == q:
                    continue
                if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                    S[p].add(q)
                elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                    nd[p] += 1
            if nd[p] == 0:
                front[0].add(p)
        i = 0
        while len(front[i]) > 0:
            nxt = set()
            for p in front[i]:
                for q in S[p]:
                    nd[q] -= 1
                    if nd[q] == 0:
                        nxt.add(q)
            i += 1
            front.append(nxt)
        return front

    def _crowding(self, F, front):
        # assign crowding distances within a front
        n = len(front)
        if n <= 2:
            return [float('inf')] * n
        order = list(front)
        cd = {p: 0.0 for p in order}
        for m in range(self.n_obj):
            sorted_idx = sorted(order, key=lambda p: F[p][m])
            cd[sorted_idx[0]] = float('inf')
            cd[sorted_idx[-1]] = float('inf')
            fmin = F[sorted_idx[0]][m]
            fmax = F[sorted_idx[-1]][m]
            span = fmax - fmin
            if span == 0:
                continue
            for k in range(1, n - 1):
                cd[sorted_idx[k]] += (F[sorted_idx[k + 1]][m] - F[sorted_idx[k - 1]][m]) / span
        return cd

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = 300.0 - 30.0  # safety margin
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        budget = self.budget

        # population size
        pop = 40
        if pop * 5 > budget:
            pop = max(10, budget // 5)

        # initial population: spread x1 across [0,1] (f1 = x1), others random
        X = np.empty((pop, n))
        for i in range(pop):
            X[i, 0] = lo[0] + (hi[0] - lo[0]) * (i + 0.5) / pop
            X[i, 1:] = lo[1:] + (hi[1:] - lo[1:]) * np.random.rand(n - 1)
        F = np.array([problem(x) for x in X])
        used = pop

        eta_c, eta_m = 20.0, 20.0

        while used < budget:
            if time.time() > deadline:
                break
            # non-dominated sorting
            fronts = self._fast_nondominated(F)
            # build mating pool by fronts
            pool = []
            for fr in fronts:
                if not fr:
                    continue
                cd = self._crowding(F, fr)
                for p in fr:
                    pool.append((p, cd[p]))
            # tournament selection on (front_index, crowding)
            # assign front index
            front_idx = {}
            for i, fr in enumerate(fronts):
                for p in fr:
                    front_idx[p] = i
            cd_all = self._crowding(F, set(range(len(F)))) if False else None

            # produce offspring
            offspring = []
            while len(offspring) < pop and used + (pop - len(offspring)) <= budget:
                # tournament select two parents
                def select():
                    a = np.random.randint(len(F))
                    b = np.random.randint(len(F))
                    # prefer lower front index, then higher crowding
                    fa, fb = front_idx[a], front_idx[b]
                    if fa != fb:
                        return a if fa < fb else b
                    # crowding: recompute per front for fairness
                    return a if np.random.rand() < 0.5 else b
                p1 = select()
                p2 = select()
                if np.random.rand() < 0.9:
                    c1, c2 = self._sbx(X[p1], X[p2], eta_c)
                else:
                    c1, c2 = X[p1].copy(), X[p2].copy()
                c1 = self._polymut(c1, lo, hi, eta_m)
                c2 = self._polymut(c2, lo, hi, eta_m)
                offspring.append(c1)
                offspring.append(c2)
                used += 2
                if len(offspring) < pop:
                    pass
            # evaluate offspring
            off_F = np.array([problem(x) for x in offspring])
            # combine
            Xall = np.vstack([X, np.array(offspring)])
            Fall = np.vstack([F, off_F])
            # select next pop
            fronts = self._fast_nondominated(Fall)
            new_idx = []
            for fr in fronts:
                if not fr:
                    continue
                if len(new_idx) + len(fr) <= pop:
                    new_idx.extend(list(fr))
                else:
                    cd = self._crowding(Fall, fr)
                    fr_sorted = sorted(fr, key=lambda p: cd[p], reverse=True)
                    need = pop - len(new_idx)
                    new_idx.extend(fr_sorted[:need])
                    break
            X = Xall[new_idx]
            F = Fall[new_idx]
        return X
# EVOLVE-BLOCK-END
