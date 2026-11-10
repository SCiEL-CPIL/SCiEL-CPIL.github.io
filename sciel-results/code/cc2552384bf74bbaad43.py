# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 85  # 85s safety margin
    
    def is_mean_of_subset(element, others):
        """Check if element is the arithmetic mean of any subset of others with size >= 2."""
        n = len(others)
        if n < 2:
            return False
        
        target = element
        others_set = set(others)
        
        # Fast path for k=2: Check if any pair sums to 2*target
        target_sum_2 = 2 * target
        # Use set for O(1) complement check
        seen = set()
        for v in others:
            complement = target_sum_2 - v
            if complement in seen:
                return True
            seen.add(v)
        
        # For k >= 3, use Bitset DP
        # We only need to check up to n elements
        # But k*target sum grows.
        # Max possible sum is sum(others).
        
        # Sort others for potential pruning?
        sorted_others = sorted(others)
        
        # DP: dp[j] is a bitset where bit s is 1 if sum s is achievable with j elements.
        # We cap sum at max_target_sum = n * target (worst case) or sum(others).
        max_sum = sum(others)
        
        # Initialize dp
        # dp[0] = 1 (bit 0 set)
        # We only care about sums that could be k*target for some k>=3.
        # So we can mask to relevant bits?
        # Actually, just run DP for all k up to n, and check if bit (k*target) is set.
        
        # Optimization: We can stop early if we find a match.
        
        # To save memory/time, we can use a list of ints.
        # Size of dp is n+1.
        # Size of int is max_sum bits.
        
        # Iterate k from 3 to n
        # This is O(n^2 * (max_sum/word_size)) roughly.
        # With n up to 48, max_sum up to 48*512 ~ 25000.
        # 25000 bits is ~ 400 bytes. Very fast.
        
        # However, we can be smarter.
        # For a fixed k, target_sum = k * target.
        # We check if there is a subset of size k summing to target_sum.
        
        # Let's do a single DP pass that tracks counts?
        # Standard subset sum DP with count constraint:
        # dp[j][s] = True if sum s achievable with j elements.
        # We can use a list of bitsets: dp[j] is a bitset of sums.
        
        dp = [0] * (n + 1)
        dp[0] = 1  # sum 0 with 0 elements
        
        for val in sorted_others:
            # Update dp in reverse order of j to avoid using same element twice
            # But we need to update dp[j] from dp[j-1].
            # If we iterate j from n down to 1:
            # dp[j] |= (dp[j-1] << val)
            
            # We should mask to max_sum to keep ints small?
            # Python ints are arbitrary precision, so shifting is fine, but memory grows.
            # We can mask to (1 << (max_sum + 1)) - 1.
            mask = (1 << (max_sum + 1)) - 1
            
            # We only need to update j up to the current number of processed elements?
            # No, we just iterate j from n down to 1.
            # But if j-1 > number of elements processed so far, dp[j-1] is 0.
            
            for j in range(n, 0, -1):
                if dp[j-1]:
                    shifted = (dp[j-1] << val) & mask
                    dp[j] |= shifted
        
        # Now check for any k >= 3 if bit (k*target) is set in dp[k]
        for k in range(3, n + 1):
            target_sum = k * target
            if target_sum > max_sum:
                break
            if (dp[k] >> target_sum) & 1:
                return True
        
        return False

    def can_add_element(element, current_set):
        """Check if element can be added to current_set without violating the constraint."""
        if not current_set:
            return True
        
        # Check 1: is element the mean of any subset of current_set?
        if is_mean_of_subset(element, current_set):
            return False
        
        # Check 2: for each a in current_set, does adding element create a new mean violation?
        # Violation if a is mean of subset T of (current_set U {element}) \ {a}
        # T must contain element.
        # Let T = {element} U U, where U subset of current_set \ {a}, |U| >= 1.
        # Sum(T) = |T| * a
        # element + Sum(U) = (|U| + 1) * a
        # Sum(U) = (|U| + 1) * a - element
        
        current_set_set = set(current_set)
        
        for a in current_set:
            others_without_a = [x for x in current_set if x != a]
            n = len(others_without_a)
            if n == 0:
                continue
            
            # We need to check if there exists U subset of others_without_a, 1 <= |U| <= n
            # such that Sum(U) = (|U| + 1) * a - element
            
            # Let k = |U| + 1. Then 2 <= k <= n + 1.
            # Target Sum S_k = k * a - element.
            # We need subset of size k-1 from others_without_a summing to S_k.
            
            # Bounds check
            # Min sum of k-1 smallest elements
            # Max sum of k-1 largest elements
            
            sorted_others = sorted(others_without_a)
            prefix_sums = [0]
            for v in sorted_others:
                prefix_sums.append(prefix_sums[-1] + v)
            
            # DP for subset sum with count
            # We can reuse the logic from is_mean_of_subset but with variable target sums.
            # Since a changes, target sums change.
            # We must run DP for each a?
            # Yes, because the target sum depends on a.
            
            # Optimization: If n is small, just brute force?
            # n is up to 48.
            
            # Let's use Bitset DP again.
            max_sum = sum(others_without_a)
            mask = (1 << (max_sum + 1)) - 1
            
            dp = [0] * (n + 1)
            dp[0] = 1
            
            for val in sorted_others:
                for j in range(n, 0, -1):
                    if dp[j-1]:
                        shifted = (dp[j-1] << val) & mask
                        dp[j] |= shifted
            
            # Now check for each possible k
            # k ranges from 2 to n+1
            # Size of U is k-1.
            # Target sum S = k * a - element.
            # Check if S is achievable with k-1 elements.
            
            for k in range(2, n + 2):
                size_u = k - 1
                if size_u > n:
                    break
                target_sum = k * a - element
                if target_sum < 0 or target_sum > max_sum:
                    continue
                if (dp[size_u] >> target_sum) & 1:
                    return False
        
        return True

    # Strategy: Greedy with multiple orderings
    best_set = []
    
    # Generate orderings
    # 1. Random
    # 2. Powers of 2 first
    # 3. Primes first
    # 4. Small to Large
    # 5. Large to Small
    
    def is_prime(x):
        if x < 2: return False
        for i in range(2, int(x**0.5) + 1):
            if x % i == 0: return False
        return True
        
    primes = [x for x in range(2, 513) if is_prime(x)]
    non_primes = [x for x in range(1, 513) if not is_prime(x)]
    powers_of_2 = [2**i for i in range(9) if 2**i <= 512]
    non_powers = [x for x in range(1, 513) if x not in set(powers_of_2)]
    
    orderings = [
        list(range(1, 513)),
        list(range(512, 0, -1)),
        powers_of_2 + non_powers,
        primes + non_primes,
        [x for x in range(1, 513) if x % 2 == 1] + [x for x in range(1, 513) if x % 2 == 0],
    ]
    
    # Add random orderings
    for _ in range(5):
        rand_order = list(range(1, 513))
        random.shuffle(rand_order)
        orderings.append(rand_order)
        
    for ordering in orderings:
        if time.time() > deadline - 5:
            break
        
        selected = []
        selected_set = set()
        
        for x in ordering:
            if time.time() > deadline - 5:
                break
            if x in selected_set:
                continue
            
            if can_add_element(x, selected):
                selected.append(x)
                selected_set.add(x)
        
        if len(selected) > len(best_set):
            best_set = selected[:]
    
    # Local Improvement: Try to add more
    if best_set and time.time() < deadline - 5:
        improved = best_set[:]
        improved_set = set(improved)
        remaining = [x for x in range(1, 513) if x not in improved_set]
        
        # Try adding in random order
        random.shuffle(remaining)
        
        for x in remaining:
            if time.time() > deadline - 5:
                break
            if can_add_element(x, improved):
                improved.append(x)
                improved_set.add(x)
        
        if len(improved) > len(best_set):
            best_set = improved[:]
            
    return best_set
# EVOLVE-BLOCK-END
