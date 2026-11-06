<<<<<<< SEARCH
    def simulated_annealing(coloring, k, max_steps=50000, deadline_t=None):
        """Refine coloring using SA with incremental violation counting."""
        current_violations = count_violations(coloring)
        best_coloring = coloring[:]
        best_violations = current_violations
        
        T = 5.0
        cooling = 0.9995
        steps = 0
        
        while steps < max_steps and (deadline_t is None or time.time() < deadline_t):
            steps += 1
            idx = random.randint(0, N - 1)
            old_color = coloring[idx]
            new_color = random.randint(0, k - 1)
            
            if old_color == new_color:
                continue
            
            # Calculate change in violations
            delta_violations = 0
            for ap_id in ap_indices[idx]:
                ap = aps[ap_id]
                c0 = coloring[ap[0]]
                c1 = coloring[ap[1]]
                c2 = coloring[ap[2]]
                c3 = coloring[ap[3]]
                
                old_set_size = len({c0, c1, c2, c3})
                
                new_c0 = new_color if ap[0] == idx else c0
                new_c1 = new_color if ap[1] == idx else c1
                new_c2 = new_color if ap[2] == idx else c2
                new_c3 = new_color if ap[3] == idx else c3
                
                new_set_size = len({new_c0, new_c1, new_c2, new_c3})
                
                if old_set_size < 3 and new_set_size >= 3:
                    delta_violations -= 1
                elif old_set_size >= 3 and new_set_size < 3:
                    delta_violations += 1
            
            if delta_violations < 0:
                coloring[idx] = new_color
                current_violations += delta_violations
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_coloring = coloring[:]
            else:
                if delta_violations == 0:
                    accept = random.random() < 0.5
                else:
                    if T > 0:
                        accept = random.random() < np.exp(-delta_violations / T)
                    else:
                        accept = False
                
                if accept:
                    coloring[idx] = new_color
                    current_violations += delta_violations
            
            T *= cooling
        
        return best_coloring, best_violations
=======
    def simulated_annealing(coloring, k, max_steps=50000, deadline_t=None):
        """Refine coloring using SA with penalty-based objective: violations + weight * distinct_colors."""
        current_violations = count_violations(coloring)
        current_distinct = count_distinct_colors(coloring)
        best_coloring = coloring[:]
        best_violations = current_violations
        best_distinct = current_distinct
        
        # Penalty weight: large enough to strongly prefer fewer colors when violations are equal
        color_penalty = 100.0
        T = 10.0
        cooling = 0.9998
        steps = 0
        
        def energy(violations, distinct):
            return violations + color_penalty * max(0, distinct - 3)
        
        while steps < max_steps and (deadline_t is None or time.time() < deadline_t):
            steps += 1
            idx = random.randint(0, N - 1)
            old_color = coloring[idx]
            new_color = random.randint(0, k - 1)
            
            if old_color == new_color:
                continue
            
            # Calculate change in violations
            delta_violations = 0
            for ap_id in ap_indices[idx]:
                ap = aps[ap_id]
                c0 = coloring[ap[0]]
                c1 = coloring[ap[1]]
                c2 = coloring[ap[2]]
                c3 = coloring[ap[3]]
                
                old_set_size = len({c0, c1, c2, c3})
                
                new_c0 = new_color if ap[0] == idx else c0
                new_c1 = new_color if ap[1] == idx else c1
                new_c2 = new_color if ap[2] == idx else c2
                new_c3 = new_color if ap[3] == idx else c3
                
                new_set_size = len({new_c0, new_c1, new_c2, new_c3})
                
                if old_set_size < 3 and new_set_size >= 3:
                    delta_violations -= 1
                elif old_set_size >= 3 and new_set_size < 3:
                    delta_violations += 1
            
            # Track if this position's old color was unique (affects distinct count)
            old_color_count = coloring.count(old_color)
            new_color_count = coloring.count(new_color)
            
            delta_distinct = 0
            if old_color_count == 1:
                delta_distinct -= 1
            if new_color_count == 0:
                delta_distinct += 1
            
            delta_energy = delta_violations + color_penalty * max(0, current_distinct + delta_distinct - 3) - color_penalty * max(0, current_distinct - 3)
            
            if delta_energy < 0:
                coloring[idx] = new_color
                current_violations += delta_violations
                current_distinct += delta_distinct
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_distinct = current_distinct
                    best_coloring = coloring[:]
            else:
                if T > 0.01:
                    accept = random.random() < np.exp(-delta_energy / T)
                else:
                    accept = False
                
                if accept:
                    coloring[idx] = new_color
                    current_violations += delta_violations
                    current_distinct += delta_distinct
                    if current_violations < best_violations:
                        best_violations = current_violations
                        best_distinct = current_distinct
                        best_coloring = coloring[:]
            
            T *= cooling
        
        return best_coloring, best_violations
>>>>>>> REPLACE

