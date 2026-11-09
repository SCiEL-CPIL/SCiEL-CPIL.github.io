import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s budget
    
    # Known 12-term solution fallback
    fallback = [5, 6, 7, 10, 14, 15, 21, 35, 50, 70, 105, 210]
    # Verify fallback
    # 1/5+1/6+1/7+1/10+1/14+1/15+1/21+1/35+1/50+1/70+1/105+1/210
    # Sum = 420+350+300+210+150+140+100+60+42+30+20+10 = 1832/2100 != 1
    # The seed program found a valid one. I'll just run the seed logic if search fails.
    
    # Search for k=10, 11
    # Use a simple backtracking with strong pruning
    # Target sum = 1.0
    # Denominators >= 5
    
    def search(start_d, remaining_sum, current_set, max_depth):
        if time.time() > deadline:
            return None
        if not current_set:
            return None
        if len(current_set) == max_depth:
            if abs(remaining_sum - 1.0/current_set[-1]) < 1e-10:
                return current_set[:]
            return None
        if remaining_sum < 1.0 / (start_d + (max_depth - len(current_set)) * 1000): # Rough bound
            return None
            
        # Next denominator d must be >= start_d
        # 1/d <= remaining_sum / (max_depth - len(current_set) + 1) is not right
        # Sum of remaining terms = remaining_sum
        # Next term 1/d <= remaining_sum / (max_depth - len(current_set))
        # So d >= (max_depth - len(current_set)) / remaining_sum
        
        min_d = max(start_d, int((max_depth - len(current_set)) / remaining_sum) + 1)
        
        # Also 1/d < remaining_sum
        # d > 1/remaining_sum
        
        # Iterate d
        # Upper bound for d?
        # If we pick d, remaining sum becomes remaining_sum - 1/d
        # We need to fill max_depth - len(current_set) - 1 terms
        # Each term < 1/d (since denominators increase)
        # So remaining_sum - 1/d < (max_depth - len(current_set) - 1) * (1/(d+1))
        
        for d in range(min_d, 10000):
            if time.time() > deadline:
                return None
            if 1.0 / d >= remaining_sum:
                break
            new_sum = remaining_sum - 1.0 / d
            if new_sum < 0:
                break
            res = search(d + 1, new_sum, current_set + [d], max_depth)
            if res:
                return res
        return None

    # Try k=10
    res = search(5, 1.0, [], 10)
    if res:
        return res

    # Try k=11
    res = search(5, 1.0, [], 11)
    if res:
        return res

    # Fallback to seed logic
    target, reached = 2520, {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]
    raise RuntimeError("no baseline representation")
