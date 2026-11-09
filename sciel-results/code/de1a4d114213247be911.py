        # Phase 2: Multi-start ASA for global optimization
        budget_asa_total = int(0.75 * budget)
        n_starts = 2
        budget_per_start = budget_asa_total // n_starts
        
        x_current = best_x.copy()
        for i in range(n_starts):
            if calls >= budget:
                break
            x_current, _ = run_asa_segment(x_current, budget_per_start, rng)
            x_current = best_x.copy()
