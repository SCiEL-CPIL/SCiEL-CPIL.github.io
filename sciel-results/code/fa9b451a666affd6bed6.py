# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer: exact-HV-guided adaptive mutation + polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    # ---------- exact 3D hypervolume (minimisation, reference point) ----------
    def _hv_2d(self, F, ref):
        """Exact 2D HV for points strictly below ref. F: (n,2)."""
        n = len(F)
        if n == 0:
            return 0.0
        order = np.argsort(F[:, 0])
        F = F[order]
        hv = 0.0
        max_y = -np.inf
        for i in range(n):
            dx = ref[0] - F[i, 0]
            if dx <= 0:
                break
            dy = ref[1] - F[i, 1]
            if dy > max_y:
                max_y = dy
                if max_y > 0:
                    hv += dx * max_y
        return hv

    def _hv_3d(self, F, ref):
        """Exact 3D HV for points strictly below ref. F: (n,3)."""
        valid = np.all(F < ref, axis=1)
        Fv = F[valid]
        n = len(Fv)
        if n == 0:
            return 0.0
        if n == 1:
            return float(np.prod(np.maximum(0.0, ref - Fv[0])))

        order = np.argsort(Fv[:, 0])
        Fv = Fv[order]

        # Keep only the Pareto front (points not dominated by an earlier point).
        keep = np.ones(n, dtype=bool)
        best_y = np.inf
        best_z = np.inf
        for i in range(n):
            y = Fv[i, 1]
            z = Fv[i, 2]
            if y <= best_y and z <= best_z and (y < best_y or z < best_z):
                # strictly better in at least one -> later points with >= are dominated
                best_y, best_z = y, z
            else:
                keep[i] = False
        Fv = Fv[keep]
        n = len(Fv)
        if n == 0:
            return 0.0
        if n == 1:
            return float(np.prod(np.maximum(0.0, ref - Fv[0])))

        # Sweep along objective 0.
        hv = 0.0
        prev_x = -np.inf
        prev = Fv[0]
        for i in range(1, n):
            cur = Fv[i]
            dx = cur[0] - prev_x
            if dx > 0:
                hv += dx * self._hv_2d(prev.reshape(1, 2), ref[1:])
            prev = cur
            prev_x = cur[0]
        dx = ref[0] - prev_x
        if dx > 0:
            hv += dx * self._hv_2d(prev.reshape(1, 2), ref[1:])
        return hv

    # ---------- main entry point (interface preserved) ----------
    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # safety margin below 330s kill

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        ref = np.array([1705.0, 12.0, 0.3])

        X_list = []
        F_list = []

        def evaluate(x):
            f = np.asarray(problem(x), dtype=float)
            if f.shape != (self.n_obj,):
                f = f.reshape(-1)[:self.n_obj]
            X_list.append(np.asarray(x, dtype=float).copy())
            F_list.append(f)
            return f

        # ---- initial random sampling ----
        n_init = min(50, self.budget // 4)
        for i in range(n_init):
            if time.time() > deadline:
                break
            x = lo + span * rng.random(self.dim)
            evaluate(x)

        used = len(X_list)
        step_size = 0.22

        # ---- HV-guided adaptive mutation ----
        while used < self.budget:
            if time.time() > deadline:
                break

            F_arr = np.array(F_list)
            n = len(F_arr)
            valid = np.all(F_arr < ref, axis=1)
            nv = int(np.sum(valid))

            if nv == 0:
                parent_idx = int(rng.integers(n))
            else:
                # exact marginal HV contribution of each valid point
                contrib = np.zeros(n)
                if nv == 1:
                    vi = int(np.where(valid)[0][0])
                    contrib[vi] = self._hv_3d(F_arr[[vi]], ref)
                else:
                    valid_idx = np.where(valid)[0]
                    Fv = F_arr[valid_idx]
                    hv_all = self._hv_3d(Fv, ref)
                    for k, vi in enumerate(valid_idx):
                        mask = np.ones(nv, dtype=bool)
                        mask[k] = False
                        hv_without = self._hv_3d(Fv[mask], ref)
                        contrib[vi] = hv_all - hv_without

                pos = contrib > 0
                if np.any(pos):
                    pos_idx = np.where(pos)[0]
                    top_k = min(6, len(pos_idx))
                    top = pos_idx[np.argsort(contrib[pos_idx])][-top_k:]
                    w = contrib[top]
                    w = w / w.sum()
                    parent_idx = int(rng.choice(top, p=w))
                else:
                    parent_idx = int(rng.integers(n))

            parent = X_list[parent_idx]
            child = parent + step_size * span * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            fc = evaluate(child)
            used += 1

            # adapt step size
            if np.all(fc < ref):
                old_F = np.array(F_list[:-1])
                dominated = np.any(
                    np.all(old_F <= fc, axis=1) & np.any(old_F < fc, axis=1)
                )
                if not dominated:
                    step_size = max(0.05, step_size * 0.92)
                else:
                    step_size = min(1.0, step_size * 1.08)
            else:
                step_size = min(1.0, step_size * 1.15)

        # ---- polish phase: refine top HV contributors with small local steps ----
        if used < self.budget and len(F_list) > 0:
            F_arr = np.array(F_list)
            valid = np.all(F_arr < ref, axis=1)
            nv = int(np.sum(valid))
            if nv > 0:
                valid_idx = np.where(valid)[0]
                Fv = F_arr[valid_idx]
                hv_all = self._hv_3d(Fv, ref)
                contrib = np.zeros(nv)
                if nv == 1:
                    contrib[0] = hv_all
                else:
                    for k in range(nv):
                        mask = np.ones(nv, dtype=bool)
                        mask[k] = False
                        contrib[k] = hv_all - self._hv_3d(Fv[mask], ref)

                top_k_polish = min(10, nv, max(1, (self.budget - used) // 4))
                if top_k_polish > 0:
                    order = np.argsort(contrib)[::-1][:top_k_polish]
                    for k in order:
                        if used >= self.budget:
                            break
                        if time.time() > deadline - 5.0:
                            break
                        base = X_list[valid_idx[k]]
                        for _ in range(4):
                            if used >= self.budget:
                                break
                            if time.time() > deadline - 5.0:
                                break
                            child = base + 0.05 * span * rng.standard_normal(self.dim)
                            child = np.clip(child, lo, hi)
                            evaluate(child)
                            used += 1

        return X_list
# EVOLVE-BLOCK-END
