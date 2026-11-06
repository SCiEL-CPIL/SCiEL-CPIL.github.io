import numpy as np
import time

def construct():
    N = 51975
    # Get divisors > 1
    divisors = []
    for i in range(2, int(np.sqrt(N)) + 1):
        if N % i == 0:
            divisors.append(i)
            if i != N // i:
                divisors.append(N // i)
    if N % 1 != 0: # Always true
        pass
    # Add N itself if not added
    if N not in divisors:
        divisors.append(N)
    
    # Sort divisors for consistent processing, maybe not strictly necessary
    divisors.sort()
    
    # Initialize
    covered = np.zeros(N, dtype=np.uint8)
    used_moduli = set()
    classes = []
    
    start_time = time.time()
    timeout = 120.0 # Safety margin
    
    # Greedy Phase
    while time.time() - start_time < timeout - 10: # Leave 10s for local search
        best_gain = 0
        best_m = None
        best_r = None
        
        for m in divisors:
            if m in used_moduli:
                continue
            
            # Find best r for this m
            # We can vectorize the check for all r?
            # covered is length N.
            # For a given m, residues r=0..m-1.
            # We want to count zeros in covered[r::m] for each r.
            
            # Efficient way:
            # Reshape covered? No, strides are different.
            # Just loop r, it's small (max m=51975, but usually smaller).
            # Sum of m is 125k. Looping 125k times with numpy slice sum is fast.
            
            max_r_gain = 0
            max_r_val = -1
            
            # Optimization: If m is large, the slice is small.
            # If m is small, the slice is large.
            
            for r in range(m):
                # Count uncovered in this class
                # covered[r::m] is a view. 
                # sum of (1 - covered[r::m])
                gain = np.sum(1 - covered[r::m])
                if gain > max_r_gain:
                    max_r_gain = gain
                    max_r_val = r
            
            if max_r_gain > best_gain:
                best_gain = max_r_gain
                best_m = m
                best_r = max_r_val
        
        if best_gain == 0:
            break
            
        classes.append([best_m, best_r])
        used_moduli.add(best_m)
        covered[best_r::best_m] = 1
        
    # Calculate final score
    final_score = np.sum(1 - covered)
    
    # Local Search Phase (if time remains)
    # Try swapping existing classes with unused ones
    if time.time() - start_time < timeout - 2:
        current_uncovered = final_score
        improved = True
        while improved and time.time() - start_time < timeout - 2:
            improved = False
            # Try to swap each existing class with a new one
            # To be efficient, we can just try to replace the "worst" class?
            # Or iterate through all pairs?
            # Number of classes is small (~10-20).
            # Number of unused moduli is small.
            
            # Let's try a simple hill climb:
            # For each class in classes:
            #   Remove it, update covered (decrement).
            #   Find best new class among unused moduli.
            #   If new class improves total, swap.
            
            for i in range(len(classes)):
                if time.time() - start_time > timeout - 2:
                    break
                
                m_old, r_old = classes[i]
                # Temporarily remove
                # We need to recompute covered state without this class?
                # That's expensive.
                # Instead, we can just calculate the gain of adding a new class
                # given the current covered state, and compare it to the "loss" of removing the old one?
                # No, they interact.
                
                # Simpler Local Search:
                # Just re-run greedy but with a bias?
                # Or, try to add a new class if it reduces uncovered?
                # Greedy already stops when gain is 0.
                
                # Let's try swapping.
                # To swap, we must remove m_old.
                # We can't easily "uncover" just that part without knowing what other classes cover those spots.
                # So we must recalculate `covered` from scratch for the trial state.
                # Cost of recalculation: O(sum of lengths of slices) ~ 125k ops.
                # Number of swaps to try: len(classes) * (num_unused_moduli).
                # ~ 20 * 30 = 600 trials.
                # 600 * 125k = 75 million ops. Might be too slow in Python loop if not vectorized well.
                
                # Let's optimize the trial:
                # 1. Calculate `covered_without_old`:
                #    This is hard to do incrementally.
                #    Let's just rebuild `covered` for the candidate set.
                
                # Candidate set: classes with i-th removed, plus new (m_new, r_new).
                
                # We can precompute the "contribution" of each class?
                # No, overlaps make it complex.
                
                # Given time constraints, the Greedy phase is likely very strong.
                # Let's just do a few random restarts of the Greedy phase with different tie-breaking.
                
                break # Skip complex local search for now, rely on Greedy robustness
                
    return classes
