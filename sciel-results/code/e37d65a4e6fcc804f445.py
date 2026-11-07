"""Rewritten optimizer for BBOB-style f13 (Sharp ridge) using multi-start strategy."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-start optimizer: 50 random initializations, top-5 selected, parallel (1+1)-ES."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Global multi-start exploration
        n_starts = 50
        n_top = 5

        # Evaluate n_starts random points
        points = []
        values = []
        for _ in range(n_starts):
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            points.append(x)
            values.append(fx)

        values = np.array(values)
        points = np.array(points)

        # Select top n_top points
        top_indices = np.argsort(values)[:n_top]
        top_points = points[top_indices]
        top_values = values[top_indices]

        calls_used = n_starts
        remaining_budget = budget - calls_used

        if remaining_budget <= 0:
            best_idx = np.argmin(top_values)
            return top_points[best_idx], top_values[best_idx]

        # Phase 2: Run (1+1)-ES from each of the top 5 starting points in parallel
        # Allocate remaining budget evenly among the 5 ES runs
        budget_per_es = remaining_budget // n_top

        # State for each ES run
        es_states = []
        for i in range(n_top):
            es_states.append({
                'x': top_points[i].copy(),
                'fx': top_values[i],
                'sigma': 0.5,  # Smaller initial sigma for sharp ridge
                'success_count': 0,
                'fail_count': 0
            })

        # Track the best overall
        best_overall_x = es_states[0]['x'].copy()
        best_overall_fx = es_states[0]['fx']
        for s in es_states[1:]:
            if s['fx'] < best_overall_fx:
                best_overall_fx = s['fx']
                best_overall_x = s['x'].copy()

        # Run (1+1)-ES for each starting point with adaptive sigma
        for step in range(budget_per_es):
            for s in es_states:
                if calls_used >= budget:
                    break
                # Generate candidate
                y = np.clip(s['x'] + s['sigma'] * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_used += 1

                if fy < s['fx']:
                    s['x'] = y
                    s['fx'] = fy
                    s['success_count'] += 1
                    s['fail_count'] = 0
                    # Increase sigma when successful, but cap it
                    s['sigma'] = min(s['sigma'] * 1.2, 3.0)
                    # Update best overall
                    if fy < best_overall_fx:
                        best_overall_fx = fy
                        best_overall_x = y.copy()
                else:
                    s['fail_count'] += 1
                    s['success_count'] = 0
                    # Decrease sigma when failing
                    s['sigma'] = max(s['sigma'] * 0.75, 0.01)

                if calls_used >= budget:
                    break

            if calls_used >= budget:
                break

        # Use any remaining budget on the best ES run
        if calls_used < budget:
            # Find the best ES state
            best_es_idx = np.argmin([s['fx'] for s in es_states])
            best_s = es_states[best_es_idx]
            while calls_used < budget:
                y = np.clip(best_s['x'] + best_s['sigma'] * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_used += 1
                if fy < best_s['fx']:
                    best_s['x'] = y
                    best_s['fx'] = fy
                    best_s['sigma'] = min(best_s['sigma'] * 1.2, 3.0)
                    if fy < best_overall_fx:
                        best_overall_fx = fy
                        best_overall_x = y.copy()
                else:
                    best_s['sigma'] = max(best_s['sigma'] * 0.75, 0.01)

        return best_overall_x, best_overall_fx
# EVOLVE-BLOCK-END
