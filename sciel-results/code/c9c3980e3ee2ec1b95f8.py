import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    max_n = 512
    target_size = 48
    
    best_set = []
    candidates_base = list(range(1, max_n + 1))
    
    def is_valid(x, current_set, sums):
        n_curr = len(current_set)
        
        # 1. Check if x is the mean of any subset of current_set
        for k in range(2, n_curr + 1):
            target_sum = k * x
            if (sums[k] >> target_sum) & 1:
                return False
        
        # 2. Check if any a in current_set is the mean of a subset of current_set U {x}
        for a in current_set:
            for j in range(1, n_curr):
                target_sum = (j + 1) * a - x
                if target_sum < 0:
                    continue
                if (sums[j] >> target_sum) & 1:
                    return False
        return True

    def add_element(x, current_set, sums):
        n_curr = len(current_set)
        current_set.append(x)
        for k in range(n_curr, 0, -1):
            sums[k] |= (sums[k-1] << x)

    def run_pass(candidates):
        current_set = []
        sums = [0] * 50
        sums[0] = 1
        
        for x in candidates:
            if time.time() > deadline - 2:
                break
            if is_valid(x, current_set, sums):
                add_element(x, current_set, sums)
                if len(current_set) >= target_size:
                    break
        return current_set

    # Try a few deterministic orders first
    orders = [
        candidates_base,
        list(reversed(candidates_base)),
    ]
    
    for order in orders:
        if time.time() > deadline - 5:
            break
        result = run_pass(order)
        if len(result) > len(best_set):
            best_set = result
            if len(best_set) >= target_size:
                return best_set

    # Then random orders
    while time.time() < deadline:
        candidates = candidates_base[:]
        random.shuffle(candidates)
        result = run_pass(candidates)
        if len(result) > len(best_set):
            best_set = result
            if len(best_set) >= target_size:
                return best_set

    return best_set
