import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 45  # Safety margin
    
    n = 5040
    target = 4979
    
    # Generate divisors
    divisors = [d for d in range(1, n + 1) if n % d == 0]
    divisors.sort(reverse=True)
    
    # Filter divisors > target (cannot be part of sum)
    divisors = [d for d in divisors if d <= target]
    N = len(divisors)
    
    # We want to minimize k.
    # Try k = 1, 2, ...
    for k in range(1, N + 1):
        if time.time() > deadline:
            break
            
        # Meet in the Middle
        # Split divisors into two halves
        # To optimize, we can split arbitrarily or based on values.
        # Let's split into first half and second half.
        mid = N // 2
        left = divisors[:mid]
        right = divisors[mid:]
        
        # Generate all sums of subsets of size 'i' from left
        # and 'k-i' from right.
        # We need to find i such that i + (k-i) = k.
        # Actually, we iterate over possible counts from left (c_l) and right (c_r) such that c_l + c_r = k.
        
        # Optimization:
        # For a fixed k, we can generate all sums of subsets of size c from a list.
        # Since N is small (~60), and k is likely small (target 4979, max divisor 2520, so k>=2),
        # we can just iterate k.
        
        # Let's use a set for sums from the right half for specific sizes.
        # But k varies.
        # Better: For each k, check if there exists a subset of size k summing to target.
        
        # Efficient MITM for fixed k:
        # 1. Generate all subsets of left with size c_l for c_l in [max(0, k-len(right)), min(len(left), k)]
        # 2. Generate all subsets of right with size c_r = k - c_l
        # 3. Check intersection of sums.
        
        # To avoid regenerating sets for every k, we can precompute all subset sums for left and right
        # grouped by size.
        # However, N=60, 2^30 is too big for precomputation of ALL subsets.
        # But we only need subsets of size up to k.
        # If k is small (e.g., < 10), we can generate combinations.
        
        found = False
        solution = None
        
        # Determine valid range for c_l
        min_cl = max(0, k - len(right))
        max_cl = min(len(left), k)
        
        for c_l in range(min_cl, max_cl + 1):
            c_r = k - c_l
            if c_r < 0 or c_r > len(right):
                continue
            
            # Get sums for left subsets of size c_l
            sums_l = set()
            for combo in itertools.combinations(left, c_l):
                sums_l.add(sum(combo))
            
            # Get sums for right subsets of size c_r
            # We check if any sum in right matches target - s_l
            for combo in itertools.combinations(right, c_r):
                s_r = sum(combo)
                needed = target - s_r
                if needed in sums_l:
                    # Reconstruct solution
                    # We need to find the actual elements, not just the sum.
                    # Since we only stored sums, we need to re-iterate or store tuples.
                    # Storing tuples (sum, indices) is better for reconstruction.
                    
                    # Let's redo this part to store indices or the combination itself.
                    # Given the small size, storing the combination tuple is fine.
                    
                    found = True
                    # We need to find the specific combinations that produced these sums.
                    # This requires a second pass or storing a map from sum to combo.
                    
                    # Let's optimize:
                    # Build map for left: sum -> list of combos (or just one combo is enough if we just need A solution)
                    # Actually, we just need ONE valid solution.
                    
                    break
            
            if found:
                # We found a sum match, but we need the actual numbers.
                # Let's implement a helper to find the specific combos.
                
                # Re-iterate to find the exact combos
                # For efficiency, we can build a dict: sum -> combo for left
                map_l = {}
                for combo in itertools.combinations(left, c_l):
                    s = sum(combo)
                    if s not in map_l:
                        map_l[s] = combo
                
                for combo in itertools.combinations(right, c_r):
                    s_r = sum(combo)
                    needed = target - s_r
                    if needed in map_l:
                        solution = list(map_l[needed]) + list(combo)
                        break
                
                if solution:
                    break
        
        if found and solution:
            # Verify
            if len(set(solution)) == len(solution) and sum(solution) == target:
                return solution
        
        # If not found for this k, continue to k+1

    # Fallback (should not happen for this instance)
    # Greedy fallback
    rem = target
    chosen = []
    for d in divisors:
        if d <= rem:
            chosen.append(d)
            rem -= d
            if rem == 0:
                break
    return chosen
