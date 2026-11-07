import time
import random

def construct():
    start = time.time()
    deadline = start + 85.0

    def is_valid(sums):
        """Check if any two distinct sums divide each other."""
        if len(sums) < 2:
            return True
        sum_list = sorted(sums)
        max_sum = sum_list[-1]
        # We only need to check u where 2u <= max_sum
        for u in sum_list:
            if u * 2 > max_sum:
                break
            # Check multiples of u
            # Using a set for fast lookup
            for v in range(u * 2, max_sum + 1, u):
                if v in sums:
                    return False
        return True

    def try_add_element(current_sums, c):
        """Try adding element c. Returns new_sums if valid, else None."""
        # Generate new sums
        # new_sums = current_sums U {s+c for s in current_sums} U {c}
        new_sums = set(current_sums)
        for s in current_sums:
            new_sums.add(s + c)
        new_sums.add(c)
        
        # Check for distinct sums
        # If the set size is not 2*len(current_sums) + 1, there are duplicates
        if len(new_sums) != 2 * len(current_sums) + 1:
            return None
            
        # Check divisibility
        if not is_valid(new_sums):
            return None
            
        return new_sums

    best_set = []
    best_size = 0

    # Strategy: Randomized Greedy with Restart
    # Try to build large sets using large numbers
    # We try different ranges and orderings
    
    # Pool of candidates
    # Focus on larger numbers to keep sums large and checks fast
    # Also try mixed ranges
    
    ranges = [
        list(range(128, 257)),
        list(range(100, 257)),
        list(range(50, 257)),
        list(range(1, 257))
    ]
    
    for attempt in range(1000):
        if time.time() > deadline - 5.0:
            break
            
        # Pick a range
        r_idx = random.randint(0, len(ranges) - 1)
        candidates = ranges[r_idx][:]
        random.shuffle(candidates)
        
        current = []
        current_sums = set()
        
        for c in candidates:
            if len(current) >= 30:
                break
            new_sums = try_add_element(current_sums, c)
            if new_sums is not None:
                current.append(c)
                current_sums = new_sums
                
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
            if best_size >= 30:
                break

    # If we found a good set, return it
    # Ensure types are correct
    return [int(x) for x in best_set]
