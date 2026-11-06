import time
import random
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin from 90s limit
    
    n = 128
    target_low = 128
    target_high = 256
    target_range = range(target_low, target_high + 1)
    targets = list(target_range)
    num_targets = len(targets)
    
    best_A = None
    best_peak = float('inf')
    best_score = (float('inf'), float('inf')) # (peak, sum_squares)

    def get_multiplicities(A):
        mults = defaultdict(int)
        A_list = list(A)
        k = len(A_list)
        for i in range(k):
            ai = A_list[i]
            for j in range(i, k):
                aj = A_list[j]
                s = ai + aj
                if target_low <= s <= target_high:
                    mults[s] += 1
        return mults

    def evaluate(A):
        if not A:
            return (float('inf'), float('inf'), 0)
        
        mults = get_multiplicities(A)
        
        # Check coverage
        uncovered = 0
        peak = 0
        sum_squares = 0
        
        for t in target_range:
            m = mults.get(t, 0)
            if m == 0:
                uncovered += 1
            else:
                if m > peak:
                    peak = m
                sum_squares += m * m
        
        if uncovered > 0:
            return (float('inf'), float('inf'), uncovered)
            
        return (peak, sum_squares, 0)

    def try_improve(A, max_iters=1000):
        """Local search to reduce peak and sum of squares"""
        current_A = set(A)
        current_peak, current_sq, _ = evaluate(current_A)
        
        if current_peak == float('inf'):
            return current_A, current_peak, current_sq

        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() > deadline - 2:
                break
                
            A_list = list(current_A)
            random.shuffle(A_list)
            
            for a_out in A_list:
                if time.time() > deadline - 2:
                    break
                    
                # Try swapping a_out with a_in
                best_swap = None
                best_new_peak = current_peak
                best_new_sq = current_sq
                best_new_score = (current_peak, current_sq)
                
                # Candidate a_in: random samples + neighbors of existing elements
                candidates = set()
                # Add random candidates
                for _ in range(20):
                    candidates.add(random.randint(0, 128))
                # Add neighbors of existing elements (to maintain structure)
                for a in current_A:
                    candidates.add(max(0, a - 1))
                    candidates.add(min(128, a + 1))
                    candidates.add(max(0, a - 2))
                    candidates.add(min(128, a + 2))
                
                for a_in in candidates:
                    if a_in in current_A:
                        continue
                    
                    new_A = current_A - {a_out}
                    new_A.add(a_in)
                    
                    new_peak, new_sq, unc = evaluate(new_A)
                    
                    if unc > 0:
                        continue
                        
                    new_score = (new_peak, new_sq)
                    
                    # Accept if better peak, or same peak but better sq
                    if new_score < best_new_score:
                        best_new_score = new_score
                        best_swap = a_in
                        best_new_peak = new_peak
                        best_new_sq = new_sq
                        
                if best_swap is not None:
                    current_A.remove(a_out)
                    current_A.add(best_swap)
                    current_peak = best_new_peak
                    current_sq = best_new_sq
                    improved = True
                    break # Restart search with new state
                    
        return current_A, current_peak, current_sq

    # Phase 1: Greedy Construction to get a valid set
    # We want a set that covers all targets.
    # Start with a small random set and add elements that cover the most uncovered targets.
    
    def greedy_construct():
        A = set()
        # Initialize with a few random elements to break symmetry
        for _ in range(5):
            A.add(random.randint(0, 128))
        
        uncovered = set(target_range)
        
        # Iteratively add the element that covers the most uncovered targets
        # or reduces the peak if all are covered.
        
        for _ in range(30): # Max size 35
            if time.time() > deadline - 10:
                break
                
            if not uncovered:
                break
                
            best_elem = None
            best_gain = -1
            best_new_uncovered = uncovered
            
            # Try all possible elements
            for a in range(129):
                if a in A:
                    continue
                
                new_A = A | {a}
                # Calculate new multiplicities
                mults = defaultdict(int)
                A_list = list(new_A)
                k = len(A_list)
                for i in range(k):
                    ai = A_list[i]
                    for j in range(i, k):
                        s = ai + A_list[j]
                        if target_low <= s <= target_high:
                            mults[s] += 1
                
                new_uncovered = set()
                for t in target_range:
                    if mults.get(t, 0) == 0:
                        new_uncovered.add(t)
                
                gain = len(uncovered) - len(new_uncovered)
                
                # Prefer elements that cover more uncovered targets
                # Tie-breaker: lower peak
                if gain > best_gain:
                    best_gain = gain
                    best_elem = a
                    best_new_uncovered = new_uncovered
                elif gain == best_gain and gain > 0:
                    # Check peak
                    peak = max(mults.get(t, 0) for t in target_range)
                    # This is a bit expensive inside the loop, so we only do it if gain is same
                    # For now, just take the first one with max gain
                    pass

            if best_elem is None:
                break
                
            A.add(best_elem)
            uncovered = best_new_uncovered
            
            if not uncovered:
                break
                
        return A

    # Run several random restarts
    restarts = 0
    while time.time() < deadline - 5:
        restarts += 1
        
        # Generate initial valid set
        A_init = greedy_construct()
        
        if not A_init:
            continue
            
        # Check if valid
        peak, sq, unc = evaluate(A_init)
        
        if unc > 0:
            # Try to fix coverage by adding random elements
            for _ in range(10):
                A_init.add(random.randint(0, 128))
                peak, sq, unc = evaluate(A_init)
                if unc == 0:
                    break
            
            if unc > 0:
                continue
        
        # Local search to optimize
        A_opt, peak_opt, sq_opt = try_improve(A_init, max_iters=500)
        
        if peak_opt < best_peak:
            best_peak = peak_opt
            best_A = A_opt
            best_score = (peak_opt, sq_opt)
            
            # If we found a very good solution, we might stop early?
            # Theoretical min peak is likely small. 
            # With |A|=k, avg mult is ~ k^2/2 / 129.
            # If k=20, avg is ~ 200/129 ~ 1.5. Peak likely 3-5.
            # If k=16, avg is ~ 136/129 ~ 1. Peak likely 2-3.
            if best_peak <= 3:
                # Try to optimize further with more iterations
                pass

    # If no valid set found (unlikely), return a default valid set
    if best_A is None:
        # Fallback: [64, 128]
        return list(range(64, 129))
        
    return list(best_A)
