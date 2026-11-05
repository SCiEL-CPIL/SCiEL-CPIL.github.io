            # Move: Flip a random bit
            idx = random.randint(0, M-1)
            old_val = x[idx]
            x[idx] = 1 - old_val
            
            # Calculate delta in score and conflicts
            delta_score = 1 if x[idx] else -1
            
            # Delta conflicts:
            # If x[idx] became 1, conflicts increase by number of neighbors that are 1.
            # If x[idx] became 0, conflicts decrease by number of neighbors that are 1.
            neighbor_conflicts = sum(1 for j in conflicts[idx] if x[j])
            
            if x[idx] == 1:
                delta_conflicts = neighbor_conflicts
            else:
                delta_conflicts = -neighbor_conflicts
                
            new_score = current_score + delta_score
            new_conflicts = current_conflicts + delta_conflicts
            
            P = 100
            delta_pen = delta_score - P * delta_conflicts
            
            if delta_pen > 0 or random.random() < math.exp(delta_pen / T):
                current_score = new_score
                current_conflicts = new_conflicts
                if new_conflicts == 0 and new_score > best_score:
                    best_score = new_score
                    best_x = x.copy()
            else:
                x[idx] = old_val
