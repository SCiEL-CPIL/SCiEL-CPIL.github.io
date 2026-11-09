        while time.time() < deadline and T > min_T:
            # ...
            move_type = random.random()
            current_peak = get_peak(current_reps)
            current_ss = get_sum_squares(current_reps)
            
            if move_type < 0.3 and len(current_A) > 15:
                # Remove
                item_to_remove = random.choice(list(current_A))
                new_A = set(current_A)
                new_A.remove(item_to_remove)
                new_reps = calc_reps(new_A)
                
                if is_covered(new_reps):
                    new_peak = get_peak(new_reps)
                    new_ss = get_sum_squares(new_reps)
                    
                    if new_peak < current_peak or (new_peak == current_peak and new_ss < current_ss):
                        current_A = new_A
                        current_reps = new_reps
                        update_best(current_A, current_reps)
                        stuck_counter = 0
                    else:
                        # Accept with prob exp(-delta/T)
                        # Delta is based on peak and ss
                        # If peak increased, penalize heavily.
                        if new_peak > current_peak:
                            delta = 10.0 + (new_peak - current_peak)
                        else:
                            delta = (new_ss - current_ss) / 100.0 # Scale SS
                        prob = np.exp(-delta / T)
                        if random.random() < prob:
                            current_A = new_A
                            current_reps = new_reps
                            stuck_counter = 0
                        else:
                            stuck_counter += 1
                else:
                    stuck_counter += 1
            elif move_type < 0.8:
                # Swap
                item_to_remove = random.choice(list(current_A))
                new_item = random.randint(0, 128)
                if new_item == item_to_remove: continue
                new_A = set(current_A)
                new_A.remove(item_to_remove)
                new_A.add(new_item)
                new_reps = calc_reps(new_A)
                
                if is_covered(new_reps):
                    new_peak = get_peak(new_reps)
                    new_ss = get_sum_squares(new_reps)
                    
                    if new_peak < current_peak or (new_peak == current_peak and new_ss < current_ss):
                        current_A = new_A
                        current_reps = new_reps
                        update_best(current_A, current_reps)
                        stuck_counter = 0
                    else:
                        if new_peak > current_peak:
                            delta = 10.0 + (new_peak - current_peak)
                        else:
                            delta = (new_ss - current_ss) / 100.0
                        prob = np.exp(-delta / T)
                        if random.random() < prob:
                            current_A = new_A
                            current_reps = new_reps
                            stuck_counter = 0
                        else:
                            stuck_counter += 1
                else:
                    stuck_counter += 1
            else:
                # Add (if we want to grow? Usually we want to shrink or swap)
                # Let's stick to Remove/Swap.
                pass
            
            T *= cooling
