"""Construct maximum-size set with cardinality-separated subset sums."""
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    max_val = 256
    max_size = 60

    def try_add(elements, x, dp, current_size):
        """Try to add x to elements, check if cardinality separation holds."""
        if x in elements:
            return None

        new_dp = [dp[k] for k in range(len(dp))]
        # Extend dp if needed
        while len(new_dp) <= current_size + 1:
            new_dp.append(0)

        # Update: for k from current_size down to 0
        for k in range(current_size, -1, -1):
            new_dp[k+1] |= (dp[k] << x)

        # Check cardinality separation: for each pair (i,j) with i≠j, 
        # dp[i] & dp[j] must be 0 (no common sums)
        n_k = len(new_dp)
        for i in range(n_k):
            for j in range(i+1, n_k):
                if new_dp[i] & new_dp[j]:
                    return None

        return elements + [x], new_dp

    def verify_disjoint(dp, n):
        """Check that all dp[i] for i in 0..n are pairwise disjoint."""
        for i in range(n + 1):
            for j in range(i + 1, n + 1):
                if dp[i] & dp[j]:
                    return False
        return True

    def is_valid_exact(s):
        """Exact check using bitsets."""
        n = len(s)
        if n < 2:
            return True
        
        sums = [0] * (n + 1)
        sums[0] = 1
        
        for x in s:
            for k in range(n, 0, -1):
                sums[k] |= (sums[k-1] << x)
                
        for k in range(n + 1):
            for j in range(k + 1, n + 1):
                if sums[k] & sums[j]:
                    return False
        return True

    def is_valid_range(s):
        """
        Check if sorted s satisfies Max_k < Min_{k+1} for all k.
        This is a necessary condition for validity (if valid, this must hold).
        It's much faster than exact check.
        """
        n = len(s)
        if n < 2:
            return True
        s_sorted = sorted(s)
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] + s_sorted[i]
        
        for k in range(1, n):
            max_k = prefix[n] - prefix[n - k]
            min_k1 = prefix[k + 1]
            if max_k >= min_k1:
                return False
        return True

    # Phase 1: Greedy construction with multiple orderings
    best_elements = []
    best_dp = [1]

    orderings = [
        ("inc", list(range(1, max_val + 1))),
        ("dec", list(range(max_val, 0, -1))),
        ("sparse", [1, 3, 7, 15, 31, 63, 127, 255, 256] + list(range(2, 128))),
        ("prime", [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251] + list(range(4, 257, 2))),
    ]

    for ordering_name, ordering in orderings:
        if time.time() > deadline:
            break

        elements = []
        dp = [1]

        for x in ordering:
            if time.time() > deadline:
                break
            if len(elements) >= max_size:
                break
            result = try_add(elements, x, dp, len(elements))
            if result:
                elements, dp = result

        if len(elements) > len(best_elements):
            best_elements = list(elements)
            best_dp = dp

    # Try to add more elements to best set
    if len(best_elements) < max_size and time.time() < deadline:
        in_set = set(best_elements)
        for x in range(1, max_val + 1):
            if time.time() > deadline:
                break
            if x in in_set:
                continue
            result = try_add(best_elements, x, best_dp, len(best_elements))
            if result:
                best_elements, best_dp = result
                in_set.add(x)
                if len(best_elements) >= max_size:
                    break

    # Phase 2: Local Search / Simulated Annealing inspired by Inspiration 1
    # We use the exact check for moves, but we can also use the range check as a filter.
    
    current_set = best_elements[:]
    current_size = len(current_set)
    
    # Helper to check if we can add x
    def can_add(s, x):
        if x in s:
            return False
        # Quick filter
        if not is_valid_range(s + [x]):
            return False
        return is_valid_exact(s + [x])

    # Hill Climbing with Restarts
    # We keep best_set globally
    for restart in range(200):
        if time.time() > deadline - 2.0:
            break
            
        # Start from current best or a random valid set
        if restart == 0:
            s = current_set[:]
        else:
            # Generate a random dense cluster or random set
            # Try to find a valid set of random size
            # Simple approach: take random subset of 1..256, check validity
            # If valid, use it. Otherwise fallback to small set.
            # Better: Start with a small valid set and grow?
            # Let's just try random sets of size 10-30
            n_try = random.randint(10, 30)
            s = random.sample(range(1, max_val + 1), n_try)
            if not is_valid_exact(s):
                # Fallback to best_set if random fails, or try again?
                # To save time, just skip this restart or use best_set
                continue
            
        s_set = set(s)
        
        # Try to add elements greedily
        improved = True
        while improved:
            if time.time() > deadline - 2.0:
                break
            improved = False
            # Try all unused elements
            # To speed up, we can iterate in random order or sorted order
            # Sorted order often works well for "tight" sets
            candidates = [x for x in range(1, max_val + 1) if x not in s_set]
            # Shuffle to avoid bias
            random.shuffle(candidates)
            
            for x in candidates:
                if can_add(s, x):
                    s.append(x)
                    s_set.add(x)
                    improved = True
                    if len(s) > len(best_elements):
                        best_elements = s[:]
                    # Restart the while loop to check again
                    break 
            
        # Try swaps to free up space?
        # If we can't add, maybe swap a large element with a smaller one to allow more adds?
        # Simple Swap Move
        for _ in range(50):
            if time.time() > deadline - 2.0:
                break
            if len(s) < 2:
                break
            i = random.randint(0, len(s) - 1)
            # Pick a candidate to swap with
            candidates = [x for x in range(1, max_val + 1) if x not in s_set]
            if not candidates:
                break
            j = random.choice(candidates)
            
            new_s = s[:]
            new_s[i] = j
            new_s_set = set(new_s)
            if len(new_s_set) != len(new_s):
                continue
            
            # Quick filter
            if is_valid_range(new_s) and is_valid_exact(new_s):
                s = new_s
                s_set = new_s_set
                if len(s) > len(best_elements):
                    best_elements = s[:]
                
    # Final verification
    if not best_elements:
        return [1]

    # Verify: check all subset sums by cardinality
    n = len(best_elements)
    verify_dp = [0] * (n + 1)
    verify_dp[0] = 1
    for x in best_elements:
        for k in range(n, 0, -1):
            verify_dp[k] |= (verify_dp[k-1] << x)

    # Check disjointness
    for i in range(n + 1):
        for j in range(i + 1, n + 1):
            if verify_dp[i] & verify_dp[j]:
                # Invalid! Fall back to a known-valid small set
                return [1, 3, 7]

    # Ensure all are Python ints
    return [int(x) for x in best_elements]
