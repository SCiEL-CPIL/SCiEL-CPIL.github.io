import time

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    best_range = float('inf')
    best_vals = None
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50.0
    
    # Search for a starting point A
    for A in range(1, 5000):
        if time.time() > deadline:
            break
            
        vals = []
        for d in divisors:
            rem = A % d
            offset = (d - rem) % d
            vals.append(A + offset)
        
        # Check distinctness
        if len(set(vals)) == n:
            current_range = max(vals) - min(vals)
            if current_range < best_range:
                best_range = current_range
                best_vals = vals[:]
                if best_range <= 10:
                    break
        else:
            # Try to fix collisions
            # Simple strategy: for each duplicate value, try incrementing one of the indices
            # This is O(n^2) per A, which is fine for small n.
            
            # Group indices by value
            from collections import defaultdict
            groups = defaultdict(list)
            for i, v in enumerate(vals):
                groups[v].append(i)
                
            found_fix = False
            for v, indices in groups.items():
                if len(indices) > 1:
                    # Try shifting each index in the group to its next multiple
                    for idx_to_shift in indices:
                        new_vals = vals[:]
                        d_shift = divisors[idx_to_shift]
                        # Next multiple is current + d
                        new_vals[idx_to_shift] += d_shift
                        
                        # Check if new_vals are distinct
                        if len(set(new_vals)) == n:
                            current_range = max(new_vals) - min(new_vals)
                            if current_range < best_range:
                                best_range = current_range
                                best_vals = new_vals[:]
                                found_fix = True
                                if best_range <= 10:
                                    break
                    if best_range <= 10:
                        break
            if best_range <= 10:
                break

    if best_vals is None:
        return divisors[:]
        
    return best_vals
