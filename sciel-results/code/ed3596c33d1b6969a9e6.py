import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55 seconds limit to be safe
    
    MAX_N = 10000000
    
    # Sieve for sigma(n)
    sigma = [0] * (MAX_N + 1)
    for i in range(1, MAX_N + 1):
        for j in range(i, MAX_N + 1, i):
            sigma[j] += i
    
    # Group numbers by their sigma value
    # We only care about sigma values that appear multiple times
    sigma_groups = defaultdict(list)
    for n in range(1, MAX_N + 1):
        s = sigma[n]
        if s > 0:
            sigma_groups[s].append(n)
            
    best_ratio_diff = float('inf')
    best_a = None
    best_b = None
    
    # Target ratio is 3/2 = 1.5
    target = 1.5
    
    # Iterate through groups
    # To optimize, we can sort groups by size or just iterate
    # Since we want ratio close to 1.5, we can check pairs within each group
    # If a group has many elements, we can be smarter about which pairs to check
    # For now, simple pair check within small groups, and sampling for large groups?
    # Actually, for each w in a group, we want v such that v/w ~ 1.5
    # So v ~ 1.5 * w
    # We can use binary search or just check neighbors in the sorted list of the group
    
    # Let's sort the lists in the groups to allow binary search
    for s, ns in sigma_groups.items():
        if time.time() > deadline:
            break
        if len(ns) < 2:
            continue
        
        ns.sort()
        
        # For each w in ns, look for v in ns such that v/w is close to 1.5
        # This means v is close to 1.5 * w
        # We can use bisect to find the position
        import bisect
        
        for w in ns:
            target_v = 1.5 * w
            # Find insertion point
            pos = bisect.bisect_left(ns, target_v)
            
            # Check neighbors around pos
            # We check a small window around the target
            # The window size can be adjusted.
            # Since ns is sorted, the closest values to target_v are near pos
            # We check indices pos-2 to pos+2
            
            start_idx = max(0, pos - 5)
            end_idx = min(len(ns), pos + 5)
            
            for i in range(start_idx, end_idx):
                v = ns[i]
                if v == w:
                    continue
                # Calculate ratio
                # We want |v/w - 1.5|
                # Which is |2v - 3w| / (2w)
                # We want to minimize |2v - 3w| / w
                # Let's just compute the float difference for simplicity, 
                # but be careful with precision.
                # The task says "rational error are exact", but score is binary64 of that rational.
                # So we should compare exactly if possible, or use high precision.
                # |v/w - 3/2| = |2v - 3w| / (2w)
                # Let num = abs(2*v - 3*w)
                # Let den = 2*w
                # We want to minimize num/den
                
                num = abs(2 * v - 3 * w)
                den = 2 * w
                
                if num == 0:
                    # Ratio is exactly 1.5
                    # This is invalid per task description: "A raw value of exactly 0 cannot be scored"
                    # So we skip this? Or is it just that we can't return it?
                    # "Any broken rule makes the whole result invalid"
                    # So we must avoid num == 0.
                    continue
                
                # Compare with best
                # We store best_num, best_den to compare fractions exactly
                if best_a is None:
                    best_a = v
                    best_b = w
                    best_num = num
                    best_den = den
                else:
                    # Compare num/den < best_num/best_den
                    # num * best_den < best_num * den
                    if num * best_den < best_num * den:
                        best_a = v
                        best_b = w
                        best_num = num
                        best_den = den
                        
    if best_a is None:
        # Fallback if no pair found (should not happen)
        return [11, 6]
    
    # Ensure we return Python ints
    return [int(best_a), int(best_b)]
