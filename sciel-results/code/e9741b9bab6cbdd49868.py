import numpy as np
import bisect

def construct():
    N = 10_000_000
    sigma = np.zeros(N + 1, dtype=np.int64)
    for i in range(1, N + 1):
        sigma[i::i] += i
    
    # Use a dictionary to group numbers by their sigma value
    # This avoids the O(N log N) global sort
    sigma_map = {}
    for n in range(1, N + 1):
        s = int(sigma[n])
        if s in sigma_map:
            sigma_map[s].append(n)
        else:
            sigma_map[s] = [n]
            
    best_ratio = float('inf')
    best_a, best_b = 1, 1
    
    for s, group in sigma_map.items():
        if len(group) < 2:
            continue
        group.sort()
        size = len(group)
        for i in range(size):
            b = group[i]
            target = 1.5 * b
            # Binary search for target in group
            idx = bisect.bisect_left(group, int(target))
            # Check neighbors
            for j in [idx - 1, idx, idx + 1]:
                if 0 <= j < size:
                    a = group[j]
                    if a != b:
                        # Calculate exact rational error
                        # |a/b - 3/2| = |2a - 3b| / (2b)
                        num = abs(2 * a - 3 * b)
                        if num == 0:
                            continue # Invalid
                        den = 2 * b
                        err = num / den
                        if err < best_ratio:
                            best_ratio = err
                            best_a, best_b = a, b
                            
    return [best_a, best_b]
