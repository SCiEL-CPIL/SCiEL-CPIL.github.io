"""Optimized optimizer for BBOB f13 (Sharp ridge)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead simplex method initialized at box center for sharp ridge tracking."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Initialize simplex at the box center
        # Nelder-Mead needs dim+1 points
        x0 = np.zeros(dim)
        f0 = func(x0)
        best_x = x0.copy()
        best_fx = f0

        # Build initial simplex
        # Start with a small perturbation in each direction
        simplex = [x0.copy()]
        fvals = [f0]

        # Add dim more points, each differing from x0 in one dimension
        initial_step = 0.5
        for i in range(dim):
            xi = x0.copy()
            xi[i] += initial_step
            xi = np.clip(xi, -5, 5)
            fi = func(xi)
            simplex.append(xi)
            fvals.append(fi)
            if fi < best_fx:
                best_fx = fi
                best_x = xi.copy()

        calls_made = len(simplex)

        # Nelder-Mead parameters
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma_nm = 0.5  # shrink

        # Adaptive initial step size
        step_size = initial_step

        while calls_made < budget:
            if time.time() > deadline:
                break

            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]

            # Compute centroid of all points except the worst
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            x_worst = simplex[-1]
            f_worst = fvals[-1]
            x_reflect = centroid + alpha * (centroid - x_worst)
            x_reflect = np.clip(x_reflect, -5, 5)
            f_reflect = func(x_reflect)
            calls_made += 1

            if f_reflect < fvals[0]:
                # Try expansion
                x_expand = centroid + gamma * (x_reflect - centroid)
                x_expand = np.clip(x_expand, -5, 5)
                f_expand = func(x_expand)
                calls_made += 1

                if f_expand < f_reflect:
                    simplex[-1] = x_expand
                    fvals[-1] = f_expand
                    if f_expand < best_fx:
                        best_fx = f_expand
                        best_x = x_expand.copy()
                else:
                    simplex[-1] = x_reflect
                    fvals[-1] = f_reflect
                    if f_reflect < best_fx:
                        best_fx = f_reflect
                        best_x = x_reflect.copy()
            elif f_reflect < fvals[-2]:
                # Reflection is good but not great
                simplex[-1] = x_reflect
                fvals[-1] = f_reflect
                if f_reflect < best_fx:
                    best_fx = f_reflect
                    best_x = x_reflect.copy()
            else:
                # Contraction
                if f_reflect < f_worst:
                    # Outside contraction
                    x_contract = centroid + rho * (x_reflect - centroid)
                    x_contract = np.clip(x_contract, -5, 5)
                    f_contract = func(x_contract)
                    calls_made += 1

                    if f_contract < f_reflect:
                        simplex[-1] = x_contract
                        fvals[-1] = f_contract
                        if f_contract < best_fx:
                            best_fx = f_contract
                            best_x = x_contract.copy()
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fi = func(simplex[i])
                            calls_made += 1
                            fvals[i] = fi
                            if fi < best_fx:
                                best_fx = fi
                                best_x = simplex[i].copy()
                else:
                    # Inside contraction
                    x_contract = centroid - rho * (centroid - x_worst)
                    x_contract = np.clip(x_contract, -5, 5)
                    f_contract = func(x_contract)
                    calls_made += 1

                    if f_contract < f_worst:
                        simplex[-1] = x_contract
                        fvals[-1] = f_contract
                        if f_contract < best_fx:
                            best_fx = f_contract
                            best_x = x_contract.copy()
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fi = func(simplex[i])
                            calls_made += 1
                            fvals[i] = fi
                            if fi < best_fx:
                                best_fx = fi
                                best_x = simplex[i].copy()

            # Check if simplex is too small (convergence)
            if calls_made >= budget:
                break

            # Adaptive step size: if progress is slow, try a larger perturbation
            # Check if best has improved in last few iterations
            spread = np.max([np.linalg.norm(simplex[0] - simplex[i]) for i in range(1, dim + 1)])
            if spread < 1e-6:
                # Reset with a new random perturbation around best
                for i in range(1, dim + 1):
                    simplex[i] = best_x + step_size * rng.standard_normal(dim)
                    simplex[i] = np.clip(simplex[i], -5, 5)
                    fi = func(simplex[i])
                    calls_made += 1
                    fvals[i] = fi
                    if fi < best_fx:
                        best_fx = fi
                        best_x = simplex[i].copy()
                step_size *= 0.5
                if step_size < 1e-4:
                    break

        return best_x, best_fx
# EVOLVE-BLOCK-END
