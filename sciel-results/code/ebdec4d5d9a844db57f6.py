import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    N = 128
    min_sum = 128
    max_sum = 256
    num_targets = max_sum - min_sum + 1  # 129
    
    # Pre-compute indices for faster cost calculation if needed, 
    # but np.convolve is fast enough for N=128.
    
    def get_counts(A_mask):
        """
        A_mask: numpy array of 0s and 1s of length N+1
        Returns: numpy array of counts for sums in [min_sum, max_sum]
        """
        # Convolution gives ordered pair counts
        conv = np.convolve(A_mask, A_mask, mode='full')
        
        # Extract target range
        # conv[s] is the count of ordered pairs (x,y) such that x+y=s
        # We need unordered pairs with x <= y
        # count_unordered(s) = (conv[s] + indicator(s/2 in A if s even else 0)) // 2
        
        counts = np.zeros(num_targets, dtype=np.int64)
        
        for i, s in enumerate(range(min_sum, max_sum + 1)):
            if s % 2 == 0:
                half = s // 2
                if 0 <= half <= N:
                    # conv[s] includes (x,y) and (y,x). 
                    # If x != y, they are distinct in conv.
                    # If x == y, it is (x,x), counted once in conv.
                    # Unordered: (x,y) with x<y is 1. (x,x) is 1.
                    # conv[s] = 2 * (# pairs x<y) + 1 * (# pairs x==y)
                    # We want # pairs x<y + # pairs x==y
                    # Let k = # pairs x<y, m = # pairs x==y (0 or 1)
                    # conv[s] = 2k + m
                    # target = k + m = (2k+m) + m = conv[s] + m
                    # m is 1 if s/2 in A, else 0.
                    target = (conv[s] + A_mask[half]) // 2
                else:
                    target = conv[s] // 2
            else:
                target = conv[s] // 2
            counts[i] = target
            
        return counts

    def get_cost(A_mask):
        """
        Returns cost. Lower is better.
        If any sum is 0, cost is high.
        Otherwise, cost is max count.
        """
        counts = get_counts(A_mask)
        if np.any(counts == 0):
            # Penalty proportional to number of missing sums
            return 1000 + int(np.sum(counts == 0))
        return int(np.max(counts))
    
    def random_valid_set():
        """
        Greedy construction of a valid set.
        """
        A = set()
        # Start with some spread out numbers
        # Try to cover [128, 256]
        
        # Heuristic: Pick numbers that are roughly spaced by 128/k
        # Let's try a random restart approach for initialization
        for _ in range(20):
            A_candidate = set()
            # Add 0 and 128 to help cover extremes?
            # Actually, sums are 128..256.
            # If x in A, y in A, x+y in [128, 256].
            
            # Greedy:
            uncovered = set(range(min_sum, max_sum + 1))
            
            # Add a few random elements to start
            for _ in range(5):
                x = random.randint(0, N)
                A_candidate.add(x)
                
            # Update uncovered
            for x in A_candidate:
                for y in A_candidate:
                    if x >= y:
                        s = x + y
                        if s in uncovered:
                            uncovered.discard(s)
            
            # Add elements to cover remaining
            while uncovered:
                best_x = None
                best_gain = -1
                for x in range(N + 1):
                    if x in A_candidate:
                        continue
                    gain = 0
                    for y in A_candidate:
                        if x >= y:
                            s = x + y
                            if s in uncovered:
                                gain += 1
                    # Also check x+x
                    if x + x in uncovered:
                        gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_gain == 0:
                    break
                A_candidate.add(best_x)
                for y in A_candidate:
                    if best_x >= y:
                        s = best_x + y
                        if s in uncovered:
                            uncovered.discard(s)
                s = 2 * best_x
                if s in uncovered:
                    uncovered.discard(s)
                
            if not uncovered:
                return A_candidate
                
        # If all failed, return a large set
        return set(range(0, N + 1, 4))

    # Phase 1: Initialize
    A_set = random_valid_set()
    A_mask = np.zeros(N + 1, dtype=np.int64)
    for x in A_set:
        A_mask[x] = 1
        
    current_cost = get_cost(A_mask)
    best_cost = current_cost
    best_mask = A_mask.copy()
    
    # Phase 2: Local Search
    # We want to minimize the max count.
    # Move: Flip a bit.
    
    # Simulated Annealing parameters
    temp = 1.0
    cooling = 0.999
    
    steps = 0
    max_steps = 500000 # Adjust based on time
    
    while time.time() < deadline - 2.0:
        # Choose a random index to flip
        idx = random.randint(0, N)
        
        # Flip
        A_mask[idx] = 1 - A_mask[idx]
        
        new_cost = get_cost(A_mask)
        
        # Acceptance criterion
        if new_cost <= current_cost:
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_mask = A_mask.copy()
                # If we reach a very low cost, we can stop or tighten
                if best_cost <= 2:
                    break
        else:
            # Probabilistic acceptance for worse moves (Simulated Annealing)
            # Only useful if we are stuck in a local minimum with high cost
            # But if we are valid (cost < 1000), we generally don't want to go worse
            # unless it helps escape.
            if current_cost < 1000 and new_cost < 1000:
                # Both valid, but new is worse
                # Accept with small probability to explore
                if random.random() < 0.01:
                    current_cost = new_cost
                else:
                    A_mask[idx] = 1 - A_mask[idx] # Revert
            else:
                # One or both invalid
                # If current is invalid and new is valid, accept
                if current_cost >= 1000 and new_cost < 1000:
                    current_cost = new_cost
                elif current_cost >= 1000 and new_cost >= 1000:
                    # Both invalid, accept if new is "less invalid" (fewer holes)
                    if new_cost < current_cost:
                        current_cost = new_cost
                    else:
                        A_mask[idx] = 1 - A_mask[idx] # Revert
                else:
                    # Current valid, new invalid -> Revert
                    A_mask[idx] = 1 - A_mask[idx]
        
        steps += 1
        if steps % 1000 == 0:
            temp *= cooling
            
    # Convert best_mask back to set
    best_A = [i for i in range(N + 1) if best_mask[i] == 1]
    
    return best_A
