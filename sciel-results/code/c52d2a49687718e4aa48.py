"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_counts(A_set):
    """Compute unordered pair representation counts for targets [128,256]."""
    v = np.zeros(129, dtype=np.int32)
    for a in A_set:
        v[a] = 1
    # Ordered convolution: c[k] = number of ordered pairs (i,j) in A with i+j=k
    c = np.convolve(v, v, mode='full')  # length 257, index k gives count for sum=k
    # For target t in [128, 256]:
    # Unordered count = (c[t] + (1 if t even and t//2 in A else 0)) // 2
    counts = np.zeros(257, dtype=np.int32)
    for t in range(128, 257):
        if t % 2 == 0:
            mid = t // 2
            if 0 <= mid <= 128:
                counts[t] = (c[t] + v[mid]) // 2
            else:
                counts[t] = c[t] // 2
        else:
            counts[t] = c[t] // 2
    return counts[128:257]  # 129 entries for targets 128..256

def _local_search(A_set, time_limit=80.0):
    """Local search to minimize peak multiplicity while maintaining coverage."""
    A_set = set(A_set)
    start_time = time.time()
    
    counts = _compute_counts(A_set)
    current_peak = int(np.max(counts))
    best_peak = current_peak
    best_A = set(A_set)
    
    # Try removing elements
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for a in sorted(list(A_set)):
            if (time.time() - start_time) > time_limit:
                break
            A_set.discard(a)
            if not A_set:
                A_set.add(a)
                continue
            counts = _compute_counts(A_set)
            if np.min(counts) > 0:  # still covers all targets
                new_peak = int(np.max(counts))
                if new_peak <= current_peak:
                    current_peak = new_peak
                    if current_peak < best_peak:
                        best_peak = current_peak
                        best_A = set(A_set)
                    improved = True
                else:
                    A_set.add(a)
            else:
                A_set.add(a)
    
    # Try adding elements from [0,128] not in A
    if (time.time() - start_time) < time_limit:
        for a in range(129):
            if (time.time() - start_time) > time_limit:
                break
            if a in A_set:
                continue
            A_set.add(a)
            counts = _compute_counts(A_set)
            new_peak = int(np.max(counts))
            if new_peak < current_peak:
                current_peak = new_peak
                if current_peak < best_peak:
                    best_peak = current_peak
                    best_A = set(A_set)
            else:
                A_set.discard(a)
    
    # Try swaps: remove one, add another
    if (time.time() - start_time) < time_limit:
        for a in sorted(list(A_set)):
            if (time.time() - start_time) > time_limit:
                break
            for b in range(129):
                if (time.time() - start_time) > time_limit:
                    break
                if b == a or b in A_set:
                    continue
                A_set.discard(a)
                A_set.add(b)
                counts = _compute_counts(A_set)
                if np.min(counts) > 0:
                    new_peak = int(np.max(counts))
                    if new_peak < current_peak:
                        current_peak = new_peak
                        if current_peak < best_peak:
                            best_peak = current_peak
                            best_A = set(A_set)
                    else:
                        A_set.discard(b)
                        A_set.add(a)
                else:
                    A_set.discard(b)
                    A_set.add(a)
    
    return best_A, best_peak

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill
    
    # Start with a structured initial set
    # Use a set that's spread out to reduce peak multiplicity
    # Try: all even numbers plus some odds to cover odd targets
    # Actually, let's start with the seed and improve
    A_set = set(range(64, 129))
    
    best_A, best_peak = _local_search(A_set, time_limit=deadline - (time.time() - start_time))
    
    # Try a few different starting points if time remains
    elapsed = time.time() - start_time
    if elapsed < deadline - 5.0:
        # Try starting with a more spread-out set
        A_set2 = set(range(0, 129, 2))  # all evens
        # Need to add some odds to cover odd targets
        # Odd targets need one even + one odd
        # Add a few odds
        for o in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]:
            A_set2.add(o)
        # Check coverage
        counts2 = _compute_counts(A_set2)
        if np.min(counts2) > 0:
            A_set2, peak2 = _local_search(A_set2, time_limit=min(30.0, deadline - (time.time() - start_time)))
            if peak2 < best_peak:
                best_A = A_set2
                best_peak = peak2
    
    # Final check: ensure validity
    final_counts = _compute_counts(best_A)
    if np.min(final_counts) <= 0:
        # Fallback to seed
        return list(range(64, 129))
    
    return list(best_A)
# EVOLVE-BLOCK-END
