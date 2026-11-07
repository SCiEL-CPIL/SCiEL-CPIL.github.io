"""CMA-ES based optimizer for BBOB bent cigar function."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Full CMA-ES with restart strategy for anisotropic optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        hard_deadline = t0 + 275.0
        d = self.dim
        B = self.budget
        rng = np.random.default_rng(42)

        global_best_x = None
        global_best_f = float('inf')
        total_calls = 0

        def _eval(x):
            nonlocal global_best_x, global_best_f, total_calls
            xc = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            val = float(func(xc))
            total_calls += 1
            if val < global_best_f:
                global_best_f = val
                global_best_x = xc.copy()
            return val

        def _remaining():
            return hard_deadline - time.time()

        # Phase 1: Global exploration with large CMA-ES
        # Phase 2: Focused exploitation
        # Phase 3: Fine-tuning

        def _cma_phase(start_x, init_sigma, n_calls):
            nonlocal total_calls, global_best_x, global_best_f
            if n_calls < 12:
                return

            # CMA-ES hyperparameters
            lam = max(4, 4 + int(3.0 * np.log(d)))
            mu = lam // 2
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / w.sum()
            mueff = 1.0 / np.dot(w, w)

            # Strategy parameters
            cs = (mueff + 2.0) / ((d + 1.0)**2 + mueff)
            ds = 1.0 + 2.0 * max(0.0, (mueff - 1.0) / (d + 1.0)) + cs
            cc = (4.0 + mueff / d) / (d + 4.0 + 2.0 * mueff / d)
            c1 = 2.0 / ((d + 1.3)**2 + mueff)
            cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((d + 2.0)**2 + mueff))
            chis = np.sqrt(d) * (1.0 - 1.0 / (4.0 * d) + 1.0 / (21.0 * d**2))

            # State
            mean = np.array(start_x, dtype=np.float64)
            sigma = init_sigma
            C = np.eye(d)
            Cinv = np.eye(d)
            Csq = np.eye(d)
            Csq_inv = np.eye(d)
            pc = np.zeros(d)
            ps = np.zeros(d)

            def _refresh_matrices():
                nonlocal Cinv, Csq, Csq_inv
                try:
                    ev, evv = np.linalg.eigh(C)
                    ev = np.maximum(ev, 1e-14)
                    Cinv = evv @ np.diag(1.0 / ev) @ evv.T
                    Csq = evv @ np.diag(np.sqrt(ev)) @ evv.T
                    Csq_inv = evv @ np.diag(1.0 / np.sqrt(ev)) @ evv.T
                except Exception:
                    Cinv = np.eye(d)
                    Csq = np.eye(d)
                    Csq_inv = np.eye(d)

            _refresh_matrices()

            f_mean = _eval(mean)

            while total_calls < n_calls + (n_calls - (total_calls - 1)) and _remaining() > 1.5:
                if total_calls >= self.budget or _remaining() < 0.8:
                    break

                # Generate population
                zs = rng.standard_normal((lam, d))
                ys = (zs @ Csq.T)  # Csq @ z for each z
                xs = mean[None, :] + sigma * ys
                fs = np.array([_eval(x) for x in xs])

                if total_calls >= self.budget:
                    break

                # Select best mu
                order = np.argsort(fs)
                ys_sel = ys[order[:mu]]
                xs_sel = xs[order[:mu]]
                fs_sel = fs[order[:mu]]
                ws_sel = w[:mu]

                # Weighted mean
                new_mean = np.sum(ws_sel[:, None] * xs_sel, axis=0)
                y_w = np.sum(ws_sel[:, None] * ys_sel, axis=0)

                # CSA: step size adaptation
                ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) / sigma * y_w
                ps_norm = np.linalg.norm(ps)
                sigma *= np.exp(cs / ds * (ps_norm / chis - 1.0))
                sigma = max(sigma, 1e-9)
                sigma = min(sigma, 20.0)

                # CMA: covariance adaptation
                pc = (1.0 - cc) * pc + np.sqrt(cc * (2.0 - cc) * mueff) * y_w
                delta1 = c1 * ((1.0 + (1.0 - cc) / cc) / sigma * np.outer(pc, pc) - C)
                delta_m = cmu * (1.0 / sigma) * np.sum(
                    ws_sel[:, None] * np.outer(ys_sel, ys_sel), axis=0
                )
                C = C + delta1 + delta_m
                C = (C + C.T) * 0.5

                # Ensure PD
                ev_chk = np.linalg.eigvalsh(C)
                if ev_chk[0] < 1e-12:
                    C += (1e-10 - ev_chk[0]) * np.eye(d)

                mean = new_mean
                _refresh_matrices()

                # Stopping: sigma too small
                if sigma < 1e-7:
                    break

        # Phase 1: Global search
        phase1_calls = int(B * 0.55)
        _cma_phase(np.zeros(d), 3.5, phase1_calls)

        # Phase 2: Medium-scale refinement
        if total_calls < B and _remaining() > 2.0:
            phase2_calls = int(B * 0.25)
            if global_best_x is not None:
                x_start = global_best_x.copy()
            else:
                x_start = np.zeros(d)
            _cma_phase(x_start, 0.8, phase2_calls)

        # Phase 3: Fine local search
        if total_calls < B and _remaining() > 1.5:
            phase3_calls = B - total_calls
            if phase3_calls > 8:
                if global_best_x is not None:
                    x_start = global_best_x.copy()
                else:
                    x_start = np.zeros(d)
                _cma_phase(x_start, 0.2, phase3_calls)

        # Final: a few random perturbation checks
        while total_calls < B and _remaining() > 0.5:
            if global_best_x is not None:
                z = rng.standard_normal(d)
                xt = np.clip(global_best_x + 0.05 * z, -5, 5)
                _eval(xt)
            else:
                break

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
