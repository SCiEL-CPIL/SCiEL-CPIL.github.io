"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with time guard and Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 330s kill

        # Initial point: center of the box
        x = np.zeros(self.dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # CMA-ES parameters
        n = self.dim
        # Initial sigma: reasonable for box [-5,5]^5
        sigma = 2.0
        # Covariance matrix C (identity initially)
        C = np.eye(n)
        # Inverse square root of C
        Cs = C.copy()
        # Step sizes for mean and covariance
        cm = (1 + 2 * max(0, np.log((n - 1) / 2))) / (n + 2)
        cs = (n + 1.3) / (n + 3 + 2 * n / (n - 1))
        c1 = 2 / ((n + 1.3) ** 2 + 10)
        cmu = min(0.4 - 0.2 / n, (2 * (2 / n - 1) - c1) / ((n + 2) ** 2 + 2))
        # Damping
        damps = 1 + 2 * max(0, np.sqrt((n + 1) / (n - 1) - 1)) + cs
        # Normalization factor for step size control
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        # Evolution paths
        ps = np.zeros(n)
        ph = np.zeros(n)

        # Population size
        lambda_pop = 4 + int(np.floor(3 * np.log(n)))
        mu = lambda_pop // 2  # number of recombination weights

        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1 / (weights ** 2).sum()
        # Effective population size for covariance update
        # We use the full lambda for sampling but only top mu for recombination

        # Sampling: generate lambda_pop points
        # We'll keep a list of (fx, x) and sort

        call_count = 1  # we already made 1 call
        remaining_budget = self.budget - call_count

        # Main CMA-ES loop
        while remaining_budget > 0:
            # Check time
            if time.time() > deadline - 5.0:
                break

            # Generate lambda_pop candidates
            # Use current mean x and step size sigma
            # Standard normal samples
            z = rng_standard_normal(lambda_pop, n)
            # New points
            y = x[None, :] + sigma * (z @ Cs.T)  # shape (lambda_pop, n)
            y = np.clip(y, -5, 5)

            # Evaluate all
            fvals = np.empty(lambda_pop)
            for i in range(lambda_pop):
                if remaining_budget <= 0:
                    break
                fvals[i] = func(y[i])
                remaining_budget -= 1
                call_count += 1

            if remaining_budget <= 0:
                break

            # Update best
            min_idx = np.argmin(fvals)
            if fvals[min_idx] < best_fx:
                best_fx = fvals[min_idx]
                best_x = y[min_idx].copy()

            # Sort by fitness
            order = np.argsort(fvals)
            y_sorted = y[order]
            fvals_sorted = fvals[order]

            # Update mean
            new_x = np.sum(weights[:, None] * y_sorted[:mu], axis=0)

            # Update evolution paths
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / sigma * (new_x - x) @ Cs
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (call_count - 1)))) \
                   < (1.4 + 2 / (n + 1)) * chi_n
            ph = (1 - cs) * ph + hsig * np.sqrt(cs * (2 - cs) * mueff)

            # Update covariance
            delta_hsig = (1 - hsig) * cs * (2 - cs)
            C = (1 + c1 * delta_hsig - c1 - cmu) * C \
                + c1 * (np.outer(ps, ps) - (1 - delta_hsig) * C) \
                + cmu * np.sum(weights[:, None, None] * np.outer(y_sorted[:mu] - x, y_sorted[:mu] - x), axis=0)

            # Ensure positive definiteness
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = (eigvecs * eigvals) @ eigvecs.T

            # Update inverse square root
            Cs = invsqrt(C)

            # Update sigma
            sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 100.0)

            # Update mean
            x = new_x

            # Check time again
            if time.time() > deadline - 5.0:
                break

        # Nelder-Mead polish with remaining budget
        if remaining_budget > 5 and time.time() < deadline - 2.0:
            # Start from best_x
            x0 = best_x.copy()
            # Initial simplex
            simplex = [x0]
            for i in range(n):
                xi = x0.copy()
                xi[i] += 0.05 if x0[i] < 5 else -0.05
                simplex.append(xi)
            simplex = np.array(simplex)
            fvals_nm = np.array([func(pt) for pt in simplex])
            remaining_budget -= (n + 1)
            call_count += (n + 1)

            # Nelder-Mead
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            eps = 1e-12
            max_iter = min(remaining_budget, 200)

            for _ in range(max_iter):
                if time.time() > deadline - 1.0:
                    break
                # Sort
                order = np.argsort(fvals_nm)
                simplex = simplex[order]
                fvals_nm = fvals_nm[order]

                # Check convergence
                if np.max(fvals_nm) - np.min(fvals_nm) < eps:
                    break

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                remaining_budget -= 1
                call_count += 1
                if fr < fvals_nm[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    remaining_budget -= 1
                    call_count += 1
                    if fe < fr:
                        simplex[-1] = xe
                        fvals_nm[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals_nm[-1] = fr
                elif fr < fvals_nm[-2]:
                    simplex[-1] = xr
                    fvals_nm[-1] = fr
                else:
                    # Contraction
                    xc = centroid + rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    remaining_budget -= 1
                    call_count += 1
                    if fc < fvals_nm[-1]:
                        simplex[-1] = xc
                        fvals_nm[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            fvals_nm[i] = func(simplex[i])
                            remaining_budget -= 1
                            call_count += 1
                        if remaining_budget <= 0:
                            break

                # Update best
                min_idx = np.argmin(fvals_nm)
                if fvals_nm[min_idx] < best_fx:
                    best_fx = fvals_nm[min_idx]
                    best_x = simplex[min_idx].copy()

        return best_x, best_fx


def rng_standard_normal(n_samples, dim):
    """Generate standard normal samples."""
    # Use a fixed seed for reproducibility
    rng = np.random.default_rng(42)
    return rng.standard_normal((n_samples, dim))


def invsqrt(C):
    """Compute inverse square root of a symmetric positive definite matrix."""
    eigvals, eigvecs = np.linalg.eigh(C)
    eigvals = np.maximum(eigvals, 1e-12)
    inv_sqrt_eigvals = 1.0 / np.sqrt(eigvals)
    return eigvecs @ np.diag(inv_sqrt_eigvals) @ eigvecs.T
# EVOLVE-BLOCK-END
