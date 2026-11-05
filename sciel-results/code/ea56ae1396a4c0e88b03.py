import time
import numpy as np

def construct():
    N = 51975
    divisors = [d for d in range(3, N + 1, 2) if N % d == 0]
    divisors.sort()
    
    counts = np.zeros(N, dtype=np.uint16)
    classes = {}
    
    # Greedy Init
    for m in divisors:
        best_r = 0
        best_gain = -1
        for r in range(m):
            # Count how many 0s in counts[r::m]
            # Using numpy sum on a slice
            gain = np.sum(counts[r::m] == 0)
            if gain > best_gain:
                best_gain = gain
                best_r = r
        classes[m] = best_r
        counts[best_r::m] += 1
        
    # Current uncovered count
    uncovered = np.sum(counts == 0)
    
    start_time = time.time()
    deadline = start_time + 140.0
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for m in divisors:
            r_old = classes[m]
            
            # 1. Remove old class
            # Iterate over indices
            idxs = range(r_old, N, m)
            # We need to update uncovered count
            # For each x in idxs:
            #   if counts[x] == 1: counts[x] becomes 0, uncovered += 1
            #   counts[x] -= 1
            
            # Vectorized approach for removal?
            # old_counts = counts[idxs]
            # newly_uncovered = np.sum(old_counts == 1)
            # counts[idxs] -= 1
            # uncovered += newly_uncovered
            
            # Let's use numpy for speed
            old_counts_slice = counts[r_old::m]
            newly_uncovered = np.sum(old_counts_slice == 1)
            counts[r_old::m] -= 1
            uncovered += newly_uncovered
            
            best_r = r_old
            best_uncovered = uncovered
            
            for r in range(m):
                if r == r_old:
                    continue
                
                # 2. Try new class r
                # Add class r to counts
                # For each x in r::m:
                #   if counts[x] == 0: counts[x] becomes 1, uncovered -= 1
                #   counts[x] += 1
                
                new_counts_slice = counts[r::m]
                newly_covered = np.sum(new_counts_slice == 0)
                # Temporarily add
                counts[r::m] += 1
                current_uncovered = uncovered - newly_covered
                
                if current_uncovered < best_uncovered:
                    best_uncovered = current_uncovered
                    best_r = r
                
                # Rollback add
                counts[r::m] -= 1
                
            # 3. Apply best
            if best_r != r_old:
                classes[m] = best_r
                # We need to add the best_r class permanently
                # But we rolled back the last trial.
                # So we add it now.
                new_counts_slice = counts[best_r::m]
                newly_covered = np.sum(new_counts_slice == 0)
                counts[best_r::m] += 1
                uncovered -= newly_covered
                improved = True
            else:
                # If no change, we need to re-add r_old
                # We removed it at the start of the loop.
                old_counts_slice = counts[r_old::m]
                newly_covered = np.sum(old_counts_slice == 0)
                counts[r_old::m] += 1
                uncovered -= newly_covered

    return [[m, classes[m]] for m in divisors]
