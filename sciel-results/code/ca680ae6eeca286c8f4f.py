        # Phase 3: If we have budget left, do a few random restarts from near best
        if calls < budget and best_f > 1e-6:
            n_restarts = min(5, budget - calls)
            for _ in range(n_restarts):
                # ...
                # Short (1+1)-ES from here
                sigma = 0.5
                for _ in range(min(20, budget - calls)):
                    # ...
                    calls += 1
