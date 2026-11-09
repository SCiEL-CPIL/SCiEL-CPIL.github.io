<<<<<<< SEARCH
    for r in range(num_restarts):
        # Initialize random weights
        w = np.random.rand(5)
        # Ensure non-negative
        w = np.abs(w)
        if np.sum(w) < 1e-6:
            w = np.ones(5)
            
        # Hill climbing
        current_stations = _simulate(features @ w, times, predecessors, cycle_time, n)
        
        for step in range(steps_per_restart):
            if time.time() - start_search_time > time_budget_per_instance:
                break
                
            # Perturb weights
            new_w = w.copy()
            idx = random.randint(0, 4)
            delta = random.gauss(0, 0.2)
            new_w[idx] += delta
            if new_w[idx] < 0:
                new_w[idx] = 0
                
            new_stations = _simulate(features @ new_w, times, predecessors, cycle_time, n)
            
            if new_stations < current_stations:
                w = new_w
                current_stations = new_stations
                if current_stations < best_stations:
                    best_stations = current_stations
                    best_weights = w.copy()
            elif new_stations == current_stations:
                # Accept equal with some probability to escape plateaus? 
                # Or just move sideways.
                if random.random() < 0.5:
                    w = new_w
                    # current_stations stays same
            
        if time.time() - start_search_time > time_budget_per_instance:
            break
=======
    for r in range(num_restarts):
        # Initialize random weights
        w = np.random.rand(7)
        # Ensure non-negative
        w = np.abs(w)
        if np.sum(w) < 1e-6:
            w = np.ones(7)
            
        # Hill climbing
        current_stations = _simulate(features @ w, times, predecessors, cycle_time, n)
        
        for step in range(steps_per_restart):
            if time.time() - start_search_time > time_budget_per_instance:
                break
                
            # Perturb weights
            new_w = w.copy()
            idx = random.randint(0, 6)
            delta = random.gauss(0, 0.2)
            new_w[idx] += delta
            if new_w[idx] < 0:
                new_w[idx] = 0
                
            new_stations = _simulate(features @ new_w, times, predecessors, cycle_time, n)
            
            if new_stations < current_stations:
                w = new_w
                current_stations = new_stations
                if current_stations < best_stations:
                    best_stations = current_stations
                    best_weights = w.copy()
            elif new_stations == current_stations:
                # Accept equal with some probability to escape plateaus? 
                # Or just move sideways.
                if random.random() < 0.5:
                    w = new_w
                    # current_stations stays same
            
        if time.time() - start_search_time > time_budget_per_instance:
            break
>>>>>>> REPLACE
