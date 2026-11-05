    def local_search(a, sums_bits, deadline, temperature=1.0):
        current_a = list(a)
        current_bits = sums_bits
        best_a = list(current_a)
        best_bits = current_bits
        best_size = len(current_a)
        
        current_temp = temperature
        min_temp = 0.01
        
        while time.time() < deadline:
            if len(current_a) <= len(seed_elements):
                break
            
            # Pick a random element to remove
            removable = [x for x in current_a if x not in seed_set]
            if not removable:
                break
            
            x_remove = random.choice(removable)
            
            # Save state for potential revert
            prev_a = list(current_a)
            prev_bits = current_bits
            
            # Perform move: Remove x_remove
            current_a.remove(x_remove)
            
            # Recompute sums
            new_bits = compute_sums_bitset(current_a)
            
            # Greedy fill
            a_set = set(current_a)
            candidates = [x for x in range(1, max_val + 1) if x not in a_set]
            random.shuffle(candidates)
            
            new_a, new_bits = greedy_fill(current_a, new_bits, candidates)
            
            new_size = len(new_a)
            prev_size = len(prev_a)
            
            # Decision
            if new_size > best_size:
                best_a = new_a
                best_bits = new_bits
                best_size = new_size
                current_a = new_a
                current_bits = new_bits
            else:
                # Accept with probability
                # If new_size < prev_size, it's a loss.
                # If new_size == prev_size, it's a swap (neutral).
                # If new_size > prev_size, it's an improvement (should have been caught above? No, best_size might be higher).
                
                # Objective: Maximize Size.
                # Cost function: -Size.
                # Delta Cost = -new_size - (-prev_size) = prev_size - new_size.
                # If Delta Cost > 0 (loss), accept with prob exp(-Delta/T).
                
                delta_cost = prev_size - new_size
                
                if delta_cost <= 0:
                    # Improvement or neutral relative to previous state
                    current_a = new_a
                    current_bits = new_bits
                else:
                    prob = pow(2.71828, -delta_cost / current_temp)
                    if random.random() < prob:
                        current_a = new_a
                        current_bits = new_bits
                    else:
                        # Revert
                        current_a = prev_a
                        current_bits = prev_bits
            
            # Cool down
            current_temp = max(min_temp, current_temp * 0.99)
            
        return best_a, best_bits
