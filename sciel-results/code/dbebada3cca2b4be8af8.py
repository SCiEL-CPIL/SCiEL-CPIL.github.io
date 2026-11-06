import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 17
    min_dist = 6
    max_size = 340 # Upper bound from Brouwer
    
    # Precompute popcounts for all 17-bit numbers to speed up distance checks
    # Using bit_count() directly in the inner loop is slower due to method lookup
    popcount = [0] * (1 << n)
    for i in range(1, 1 << n):
        popcount[i] = popcount[i >> 1] + (i & 1)
        
    # Initialize with the known good seed from the current program's strategy
    # The current program starts with 0 and greedily adds in natural order.
    # We will replicate this as the first pass, then do random restarts.
    
    best_codewords = [0]
    best_size = 1
    
    # Helper function to run a greedy pass given an order of candidates
    def run_greedy_pass(order, current_codewords=None):
        if current_codewords is None:
            codewords = [0]
        else:
            codewords = current_codewords.copy()
            
        for candidate in order:
            if time.time() > deadline - 2.0:
                break
            valid = True
            for cw in codewords:
                if popcount[candidate ^ cw] < min_dist:
                    valid = False
                    break
            if valid:
                codewords.append(candidate)
        return codewords

    # Pass 1: Natural order (0 to 2^17-1)
    # This matches the current program's behavior, ensuring we at least match the baseline
    natural_order = list(range(1, 1 << n))
    current_cw = run_greedy_pass(natural_order)
    
    if len(current_cw) > best_size:
        best_size = len(current_cw)
        best_codewords = current_cw
        
    if best_size >= max_size:
        return best_codewords

    # Subsequent passes: Randomized orders
    # We reuse the same list and shuffle it to avoid reallocation overhead
    candidates = list(range(1, 1 << n))
    
    while time.time() < deadline - 2.0:
        random.shuffle(candidates)
        current_cw = run_greedy_pass(candidates)
        
        if len(current_cw) > best_size:
            best_size = len(current_cw)
            best_codewords = current_cw
            
            if best_size >= max_size:
                break

    return best_codewords
