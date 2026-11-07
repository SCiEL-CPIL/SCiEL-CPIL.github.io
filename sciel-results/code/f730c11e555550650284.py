"""Combined optimizer: CMA-ES with LHS initialization and Nelder-Mead refinement for BBOB f7."""
import time
import numpy as np


class Optimizer:
    """CMA-ES with stagnation restart, LHS initialization, and Nelder-Mead polishing."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cmaes(self, func, x0, sigma, budget, deadline, rng, state):
        """Run CMA-ES until budget or deadline. Returns (best_x, best_f)."""
        dim = self.dim
        lam = 4 + int(3 * np.log(dim))
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = 1.0 / np.sum(w ** 2)
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, (mueff - 1) / (dim + 2)) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = (2 / ((dim + 1.3) ** 2 + mueff)) * (2 - c_c)
        c_mu = min(1 - c_1, (2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff)) * (c_c + c_1))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        C = np.eye(dim)
        Csqrt = np.eye(dim)
        Cinv_sqrt = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        best_x = x0.copy()
        best_f = func(x0)
        state["calls"] += 1
        gen = 0
        no_improve = 0

        while state["calls"] < budget:
            if time.time() > deadline:
                break
            gen += 1
            z = rng.standard_normal((lam, dim))
            y = np.array([z[i] @ Cinv_sqrt for i in range(lam)])
            xk = np.array([x0 + sigma * y[i] for i in range(lam)])
            xk = np.clip(xk, -5.0, 5.0)
            fk = np.empty(lam)
            for i in range(lam):
                fk[i] = func(xk[i])
                state["calls"] += 1
                if state["calls"] >= budget or time.time() > deadline:
                    break
            if state["calls"] >= budget:
                for i in range(i + 1, lam):
                    fk[i] = np.inf

            order = np.argsort(fk)
            if fk[order[0]] < best_f:
                best_f = fk[order[0]]
                best_x = xk[order[0]].copy()
                no_improve = 0
            else:
                no_improve += 1

            xnew = np.zeros(dim)
            for i in range(mu):
                xnew += w[i] * xk[order[i]]

            yinv = np.array([Cinv_sqrt @ (xk[order[i]] - x0) / sigma for i in range(mu)])
            v = np.zeros((dim, dim))
            for i in range(mu):
                v += np.outer(w[i] * yinv[i], yinv[i])
            Csqrt = (1 - c_1 - c_mu) * Csqrt + c_mu * (Csqrt @ v + v @ Csqrt) + c_1 * (
                (1 - c_c) * Csqrt + c_c * (xnew - x0) / sigma
            )
            C = Csqrt @ Csqrt.T
            C += 1e-12 * np.eye(dim)
            try:
                L = np.linalg.cholesky(C)
                Cinv_sqrt = np.linalg.inv(L).T
            except np.linalg.LinAlgError:
                C = np.linalg.pinv(C) + 1e-8 * np.eye(dim)
                Cinv_sqrt = np.linalg.inv(np.linalg.cholesky(C + 1e-6 * np.eye(dim))).T

            pc = (1 - c_c) * pc + c_c * (xnew - x0) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_c * (2 - c_c) * mueff) * Cinv_sqrt @ (xnew - x0) / sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 1e3)

            x0 = xnew

            if no_improve > max(150, 20 * dim):
                break

        return best_x, best_f

    def _nelder_mead(self, func, x0, budget, deadline, state):
        """Nelder-Mead local refinement."""
        dim = self.dim
        # Initialize simplex
        simplex = [x0.copy()]
        fvals = [func(x0)]
        state["calls"] += 1
        
        for i in range(dim):
            x_new = x0.copy()
            x_new[i] += 0.5
            x_new = np.clip(x_new, -5, 5)
            simplex.append(x_new)
            fvals.append(func(x_new))
            state["calls"] += 1
            if state["calls"] >= budget or time.time() > deadline:
                break
        
        if state["calls"] >= budget or time.time() > deadline:
            idx = np.argmin(fvals)
            return simplex[idx], fvals[idx]
        
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma_nm = 0.5
        max_iter = 200
        
        for _ in range(max_iter):
            if state["calls"] >= budget or time.time() > deadline:
                break
            
            # Sort simplex
            idx = np.argsort(fvals)
            simplex = [simplex[i] for i in idx]
            fvals = [fvals[i] for i in idx]
            
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, -5, 5)
            f_r = func(x_r)
            state["calls"] += 1
            if state["calls"] >= budget or time.time() > deadline:
                break
            
            if f_r < fvals[0]:
                # Expansion
                x_e = centroid + gamma * (centroid - simplex[-1])
                x_e = np.clip(x_e, -5, 5)
                f_e = func(x_e)
                state["calls"] += 1
                if state["calls"] >= budget or time.time() > deadline:
                    break
                if f_e < f_r:
                    simplex[-1] = x_e
                    fvals[-1] = f_e
                else:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
            elif f_r < fvals[-2]:
                simplex[-1] = x_r
                fvals[-1] = f_r
            else:
                # Contraction
                x_c = centroid + rho * (simplex[-1] - centroid)
                x_c = np.clip(x_c, -5, 5)
                f_c = func(x_c)
                state["calls"] += 1
                if state["calls"] >= budget or time.time() > deadline:
                    break
                if f_c < f_r:
                    simplex[-1] = x_c
                    fvals[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fvals[i] = func(simplex[i])
                        state["calls"] += 1
                        if state["calls"] >= budget or time.time() > deadline:
                            break
                    if state["calls"] >= budget or time.time() > deadline:
                        break
            
            # Check for convergence
            if max(fvals) - min(fvals) < 1e-10:
                break
        
        idx = np.argmin(fvals)
        return simplex[idx], fvals[idx]

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)
        state = {"calls": 0}
        deadline = time.time() + 290.0  # Safety margin

        best_x = None
        best_f = np.inf

        # Phase 1: LHS Initialization
        n_init = 20
        if budget <= n_init:
            n_init = max(1, budget // 2)
        
        lhs_points = np.zeros((n_init, dim))
        for d in range(dim):
            perm = rng.permutation(n_init)
            for i in range(n_init):
                lhs_points[i, d] = (perm[i] + rng.uniform(0, 1)) / n_init
        
        lhs_points = lhs_points * 10.0 - 5.0
        
        for i in range(n_init):
            if time.time() > deadline:
                break
            x = lhs_points[i]
            f = func(x)
            state["calls"] += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            if state["calls"] >= budget:
                break
        
        if best_x is None:
            x0 = np.zeros(dim)
            best_f = func(x0)
            state["calls"] += 1
            best_x = x0

        # Phase 2: CMA-ES with multi-restart
        n_restarts = 3
        budget_per = max(10, (budget - state["calls"]) // n_restarts)
        
        for r in range(n_restarts):
            if state["calls"] >= budget or time.time() > deadline:
                break
            
            if r == 0:
                x0 = best_x.copy()
                sigma = 1.0  # Reduced initial sigma for better step boundary resolution
            else:
                x0 = best_x + rng.standard_normal(dim) * 0.5
                x0 = np.clip(x0, -5, 5)
                sigma = 1.0
            
            bx, bf = self._cmaes(func, x0, sigma, budget, deadline, rng, state)
            if bf < best_f:
                best_f = bf
                best_x = bx.copy()

        # Phase 3: Nelder-Mead polishing
        remaining = budget - state["calls"]
        if remaining > 10:
            bx, bf = self._nelder_mead(func, best_x, budget, deadline, state)
            if bf < best_f:
                best_f = bf
                best_x = bx.copy()

        return best_x, best_f
