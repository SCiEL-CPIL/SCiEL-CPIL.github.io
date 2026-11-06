import time

def construct():
    deadline = time.time() + 85  # 85s safety margin for 90s limit
    
    # Greedy: pick smallest positive integer such that adding it
    # keeps all subset sums distinct.
    # Condition: for all s in subset_sums, s + candidate not in subset_sums
    chosen = []
    subset_sums = {0}
    
    for i in range(12):
        if time.time() > deadline:
            break
        candidate = 1
        while True:
            # Check that no s + candidate collides with an existing subset sum
            valid = True
            for s in subset_sums:
                if (s + candidate) in subset_sums:
                    valid = False
                    break
            if valid:
                # Also ensure candidate itself is not already chosen (distinct positive integers)
                if candidate not in chosen:
                    chosen.append(candidate)
                    # Update subset sums
                    new_sums = set()
                    for s in subset_sums:
                        new_sums.add(s + candidate)
                    subset_sums = subset_sums | new_sums
                    break
            candidate += 1
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Fallback to powers of 2 if we didn't complete 12 elements
    if len(chosen) < 12:
        return [1 << i for i in range(12)]
    
    return chosen
