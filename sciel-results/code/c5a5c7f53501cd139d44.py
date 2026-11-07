<<<<<<< SEARCH
"""Rewritten optimizer for BBOB-style f13 (Sharp ridge) using multi-start strategy."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-start optimizer with adaptive step size and dynamic budget allocation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Global multi-start exploration - use fewer starts to preserve budget
        n_starts = 30
        n_top = 3

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

        # Phase 2: Run (1+1)-ES with success-based adaptation from top starting points
        # State for each ES run
        es_states = []
        for i in range(n_top):
            es_states.append({
                'x': top_points[i].copy(),
                'fx': top_values[i],
                'sigma': 2.0,  # Start with larger step size for ridge exploration
                'success_count': 0,
                'stall_count': 0,
                'last_improvement_fx': top_values[i]
            })

        # Track the best overall
        best_overall_x = es_states[0]['x'].copy()
        best_overall_fx = es_states[0]['fx']
        for s in es_states[1:]:
            if s['fx'] < best_overall_fx:
                best_overall_fx = s['fx']
                best_overall_x = s['x'].copy()

        # Run (1+1)-ES with success-based step size adaptation
        # Use a sliding window of successes to adapt sigma
        window_size = 15
        success_rates = {i: [] for i in range(n_top)}

        while calls_used < budget:
            # For each ES run, generate a candidate
            for i in range(n_top):
                if calls_used >= budget:
                    break
                s = es_states[i]
                
                # Generate candidate
                y = np.clip(s['x'] + s['sigma'] * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_used += 1

                if fy < s['fx']:
                    s['x'] = y
                    s['fx'] = fy
                    s['success_count'] += 1
                    s['stall_count'] = 0
                    success_rates[i].append(1)
                    # Update best overall
                    if fy < best_overall_fx:
                        best_overall_fx = fy
                        best_overall_x = y.copy()
                        s['last_improvement_fx'] = fy
                else:
                    s['stall_count'] += 1
                    success_rates[i].append(0)

                # Keep only last window_size entries
                if len(success_rates[i]) > window_size:
                    success_rates[i] = success_rates[i][-window_size:]

                # Adapt sigma based on success rate in window
                if len(success_rates[i]) >= 5:
                    success_rate = sum(success_rates[i]) / len(success_rates[i])
                    # Target success rate ~0.2 for (1+1)-ES
                    if success_rate > 0.3:
                        s['sigma'] *= 1.2
                    elif success_rate < 0.1:
                        s['sigma'] *= 0.8
                    else:
                        s['sigma'] *= 1.05  # Slight increase for ridge exploration

                    # Clamp sigma
                    s['sigma'] = np.clip(s['sigma'], 0.01, 10.0)

                # Restart if stalled too long
                if s['stall_count'] > 20:
                    # Restart from best point with larger sigma
                    if s['fx'] > best_overall_fx * 1.01 and best_overall_fx > 0:
                        s['x'] = best_overall_x.copy() + 0.5 * rng.standard_normal(dim)
                        s['x'] = np.clip(s['x'], -5, 5)
                        s['sigma'] = 3.0
                        s['stall_count'] = 0
                        success_rates[i] = []

            # Dynamic allocation: if one ES is much better, give it more budget
            # Check if we should focus on the best performer
            if calls_used < budget and calls_used > budget * 0.5:
                # After 50% of budget, focus more on the best ES
                best_es_idx = np.argmin([s['fx'] for s in es_states])
                # Run 3 times on best, 1 time on others
                for round_num in range(4):
                    if calls_used >= budget:
                        break
                    if round_num < 3:
                        idx = best_es_idx
                    else:
                        # Pick a random other ES
                        other_indices = [j for j in range(n_top) if j != best_es_idx]
                        idx = rng.choice(other_indices)
                    
                    s = es_states[idx]
                    y = np.clip(s['x'] + s['sigma'] * rng.standard_normal(dim), -5, 5)
                    fy = func(y)
                    calls_used += 1

                    if fy < s['fx']:
                        s['x'] = y
                        s['fx'] = fy
                        s['success_count'] += 1
                        s['stall_count'] = 0
                        success_rates[idx].append(1)
                        if fy < best_overall_fx:
                            best_overall_fx = fy
                            best_overall_x = y.copy()
                            s['last_improvement_fx'] = fy
                    else:
                        s['stall_count'] += 1
                        success_rates[idx].append(0)

                    if len(success_rates[idx]) > window_size:
                        success_rates[idx] = success_rates[idx][-window_size:]

                    if len(success_rates[idx]) >= 5:
                        success_rate = sum(success_rates[idx]) / len(success_rates[idx])
                        if success_rate > 0.3:
                            s['sigma'] *= 1.2
                        elif success_rate < 0.1:
                            s['sigma'] *= 0.8
                        else:
                            s['sigma'] *= 1.05
                        s['sigma'] = np.clip(s['sigma'], 0.01, 10.0)

                    if s['stall_count'] > 20:
                        if s['fx'] > best_overall_fx * 1.01 and best_overall_fx > 0:
                            s['x'] = best_overall_x.copy() + 0.5 * rng.standard_normal(dim)
                            s['x'] = np.clip(s['x'], -5, 5)
                            s['sigma'] = 3.0
                            s['stall_count'] = 0
                            success_rates[idx] = []

        return best_overall_x, best_overall_fx
# EVOLVE-BLOCK-END
=======
"""Optimizer for BBOB-style f13 (Sharp ridge) using global multi-start and parallel (1+1)-ES."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Global multi-start with 50 initial points, top 5 selected, parallel (1+1)-ES."""

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

        # Phase 2: Parallel (1+1)-ES from top starting points
        # Initialize states
        es_states = []
        for i in range(n_top):
            es_states.append({
                'x': top_points[i].copy(),
                'fx': top_values[i],
                'sigma': 2.0,  # Initial step size
                'successes': 0,
                'failures': 0,
                'stall_count': 0
            })

        # Track the best overall
        best_overall_x = es_states[0]['x'].copy()
        best_overall_fx = es_states[0]['fx']
        for s in es_states[1:]:
            if s['fx'] < best_overall_fx:
                best_overall_fx = s['fx']
                best_overall_x = s['x'].copy()

        # Round-robin execution
        # We will cycle through the 5 states until budget is exhausted
        # This ensures even allocation
        
        while calls_used < budget:
            for i in range(n_top):
                if calls_used >= budget:
                    break
                
                s = es_states[i]
                
                # Generate candidate
                y = s['x'] + s['sigma'] * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_used += 1

                if fy < s['fx']:
                    # Success
                    s['x'] = y
                    s['fx'] = fy
                    s['successes'] += 1
                    s['stall_count'] = 0
                    
                    # Update best overall
                    if fy < best_overall_fx:
                        best_overall_fx = fy
                        best_overall_x = y.copy()
                else:
                    # Failure
                    s['failures'] += 1
                    s['stall_count'] += 1

                # Adapt sigma
                # Use a simple success rate based on recent history
                # To keep it simple and fast, we use a moving window of last 20 evaluations
                # But storing history for 5 runs is fine.
                # Alternatively, use a simpler rule:
                # If success, increase sigma slightly. If fail, decrease.
                # But for ridges, we want to keep sigma large enough to move along the ridge.
                # A common robust approach:
                # Target success rate ~0.2.
                
                # Let's use a simple counter-based adaptation with decay
                # Or just a simple multiplicative factor based on last outcome?
                # No, that's noisy.
                
                # Let's use a window of size 10 for each run.
                # We didn't store history in the state dict above. Let's add it or use a simpler heuristic.
                # Given the budget is 500 and n_top=5, we have ~450/5 = 90 steps per run.
                # A window of 10 is reasonable.
                
                # To avoid complex state management in this edit, I'll use a slightly different adaptation:
                # Keep a running success count for the last 20 steps using a deque or list.
                # Since I can't easily edit the state dict structure in a simple replace without redefining it,
                # I will modify the state dict creation and the loop.
                
                # Wait, I am replacing the whole block. I can define the state dict with a history list.
                
                # Let's restart the state definition part in my head to ensure correctness.
                # I'll do a full replacement of the class body to be safe and clean.
                
                pass # Placeholder, actual logic below in the replacement block

        return best_overall_x, best_overall_fx
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

Wait, the `pass` in the middle of the `while` loop in my replacement logic above is bad. I need to provide the *full* correct implementation in the replacement. The SEARCH block covers the entire class. I will replace the entire class definition.

Here is the corrected replacement logic:
1.  `n_starts = 50`, `n_top = 5`.
2.  State includes `history` list for sigma adaptation.
3.  Round-robin loop.
4.  Sigma adaptation based on success rate in last 10-20 evaluations.
5.  Restart logic: if `stall_count` > 30, reset this run to global best + noise, and reset sigma.

