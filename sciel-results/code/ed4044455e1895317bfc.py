import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115s limit to be safe
    
    required = [1, 2, 5, 11]
    max_val = 284
    required_set = set(required)
    max_sum = 2 * max_val + 2
    
    # Bitset-based sum management for speed
    def compute_sums_bitset(seq):
        bits = 0
        n = len(seq)
        for i in range(n):
            for j in range(i, n):
                bits |= 1 << (seq[i] + seq[j])
        return bits
    
    def can_add(x, seq, sums_bits):
        """Check if x can be added to seq given current sums bitset."""
        if (sums_bits >> (2 * x)) & 1:
            return False
        for y in seq:
            if (sums_bits >> (x + y)) & 1:
                return False
        return True
    
    def add_element(seq, sums_bits, x):
        """Add x to seq and update sums bitset."""
        for y in seq:
            sums_bits |= 1 << (x + y)
        sums_bits |= 1 << (2 * x)
        seq.append(x)
        return sums_bits
    
    def remove_element(seq, x):
        """Remove x from seq and recompute sums bitset."""
        seq.remove(x)
        new_sums = compute_sums_bitset(seq)
        return new_sums
    
    # Initial state
    curr_set = required[:]
    curr_sums = compute_sums_bitset(curr_set)
    
    best_set = curr_set[:]
    best_size = len(curr_set)
    best_sums = curr_sums
    
    # Candidates
    candidates = [x for x in range(1, max_val + 1) if x not in required_set]
    
    # Strategy 1: Randomized Greedy with Bitsets
    def randomized_greedy():
        curr = required[:]
        curr_sums = compute_sums_bitset(curr)
        pool = candidates[:]
        random.shuffle(pool)
        
        for x in pool:
            if x in curr:
                continue
            if can_add(x, curr, curr_sums):
                curr_sums = add_element(curr, curr_sums, x)
        
        return curr, curr_sums
    
    # Strategy 2: Simulated Annealing with Bitsets
    def simulated_annealing(initial_set, initial_sums, t_start=50.0, t_end=0.1, cooling_rate=0.999):
        curr = initial_set[:]
        curr_sums = initial_sums
        best_local = curr[:]
        best_local_size = len(curr)
        best_local_sums = curr_sums
        
        temperature = t_start
        no_improve = 0
        max_no_improve = 3000
        
        while time.time() < deadline:
            move_type = random.random()
            
            if move_type < 0.5:
                # Add a random candidate
                available = [x for x in candidates if x not in curr]
                if not available:
                    continue
                x = random.choice(available)
                if not can_add(x, curr, curr_sums):
                    continue
                
                old_sums = curr_sums
                curr_sums = add_element(curr, curr_sums, x)
                
                if len(curr) > best_local_size:
                    best_local_size = len(curr)
                    best_local = curr[:]
                    best_local_sums = curr_sums
                    no_improve = 0
                else:
                    # Accept with probability based on temperature
                    # Since size increases, always accept for simplicity in this phase
                    no_improve += 1
                    
            elif move_type < 0.7:
                # Remove a random non-required element
                removable = [x for x in curr if x not in required_set]
                if not removable:
                    continue
                x = random.choice(removable)
                curr_sums = remove_element(curr, x)
                no_improve += 1
                
            else:
                # Swap: remove one, add one
                removable = [x for x in curr if x not in required_set]
                available = [x for x in candidates if x not in curr]
                if not removable or not available:
                    continue
                
                x_remove = random.choice(removable)
                x_add = random.choice(available)
                
                # Try the swap
                old_sums = curr_sums
                old_set = curr[:]
                
                # Remove x_remove
                curr_sums = remove_element(curr, x_remove)
                
                # Try to add x_add
                if can_add(x_add, curr, curr_sums):
                    curr_sums = add_element(curr, curr_sums, x_add)
                    # Size stays same, but structure changed
                    no_improve += 1
                else:
                    # Revert
                    curr = old_set[:]
                    curr_sums = old_sums
                    no_improve += 1
            
            # Cooling
            temperature *= cooling_rate
            if temperature < t_end:
                temperature = t_end
            
            # Restart if stuck
            if no_improve > max_no_improve:
                curr = best_local[:]
                curr_sums = best_local_sums
                temperature = max(temperature, t_start * 0.5)
                no_improve = 0
        
        return best_local, best_local_sums
    
    # Main Loop
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        
        # Phase 1: Randomized Greedy
        if iterations % 2 == 1:
            curr_set, curr_sums = randomized_greedy()
            if len(curr_set) > best_size:
                best_size = len(curr_set)
                best_set = curr_set[:]
                best_sums = curr_sums
        else:
            # Phase 2: Simulated Annealing starting from current best
            # Use a shorter time budget for each SA run to allow multiple restarts
            sa_deadline = time.time() + min(10, deadline - time.time())
            # Temporarily adjust deadline for SA
            old_deadline = deadline
            deadline = sa_deadline
            
            improved_set, improved_sums = simulated_annealing(best_set, best_sums)
            
            deadline = old_deadline
            
            if len(improved_set) > best_size:
                best_size = len(improved_set)
                best_set = improved_set[:]
                best_sums = improved_sums
        
        # Early exit if we reach theoretical max
        if best_size >= 20:
            break
        
        # Additional Local Search: Kick-out and Greedy Fill
        if iterations % 5 == 0 and best_size > len(required):
            # Try to improve best_set by kicking out one element and refilling
            removable = [x for x in best_set if x not in required_set]
            if removable:
                random.shuffle(removable)
                for rem_val in removable:
                    new_set = [x for x in best_set if x != rem_val]
                    new_sums = compute_sums_bitset(new_set)
                    
                    # Greedy fill
                    pool = [x for x in candidates if x not in new_set]
                    random.shuffle(pool)
                    
                    for x in pool:
                        if can_add(x, new_set, new_sums):
                            new_sums = add_element(new_set, new_sums, x)
                    
                    if len(new_set) > best_size:
                        best_size = len(new_set)
                        best_set = new_set[:]
                        best_sums = new_sums
                        break  # Found an improvement, stop trying other kicks for this iteration

    return sorted(best_set)
