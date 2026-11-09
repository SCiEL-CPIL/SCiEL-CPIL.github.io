import numpy as np

def construct():
    n = 128
    targets = np.arange(128, 257)
    target_set = set(targets)
    num_targets = len(targets)
    
    # counts[i] corresponds to target 128+i
    counts = np.zeros(num_targets, dtype=np.int32)
    
    A = []
    A_set = set()
    
    candidates = list(range(0, 129))
    
    # Precompute target indices for fast lookup
    # target t is at index t - 128
    
    for _ in range(129): # Max iterations
        best_x = -1
        best_new_counts = None
        best_max_count = float('inf')
        best_newly_covered = -1
        
        for x in candidates:
            if x in A_set:
                continue
            
            # Calculate new counts
            new_counts = counts.copy()
            
            # Add pairs (x, a) for a in A
            for a in A:
                s = x + a
                if 128 <= s <= 256:
                    idx = s - 128
                    new_counts[idx] += 1
            
            # Add pair (x, x)
            s2 = x + x
            if 128 <= s2 <= 256:
                idx = s2 - 128
                new_counts[idx] += 1
            
            # Calculate metrics
            newly_covered = np.sum(new_counts > 0) - np.sum(counts > 0)
            max_count = np.max(new_counts)
            
            # Score: Prefer higher newly_covered, then lower max_count
            # We want to maximize newly_covered first.
            if newly_covered > best_newly_covered:
                best_newly_covered = newly_covered
                best_max_count = max_count
                best_x = x
                best_new_counts = new_counts
            elif newly_covered == best_newly_covered:
                if max_count < best_max_count:
                    best_max_count = max_count
                    best_x = x
                    best_new_counts = new_counts
        
        if best_x == -1:
            break
        
        A.append(best_x)
        A_set.add(best_x)
        counts = best_new_counts
        
        # Check if all covered
        if np.all(counts > 0):
            break
    
    return A
