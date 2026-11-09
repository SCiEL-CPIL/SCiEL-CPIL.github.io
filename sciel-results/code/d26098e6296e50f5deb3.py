"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    """
    Closest strict underpacking of one by supplied unit fractions.
    Returns nonempty distinct zero-based indices into denominators [2,3,...,48]
    such that reciprocal sum is STRICTLY BELOW 1, minimizing positive deficit 1-sum.
    """
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s limit
    
    denominators = list(range(2, 49))  # [2, 3, ..., 48]
    n = len(denominators)  # 47 items
    values = [Fraction(1, d) for d in denominators]
    
    # Greedy initial solution
    selected, total = [], Fraction()
    for i, v in enumerate(values):
        if total + v < 1:
            selected.append(i)
            total += v
    best_indices = selected[:]
    best_total = total
    
    # Meet-in-the-middle approach
    # Split into two halves
    mid = n // 2  # 23 and 24
    left_items = list(range(mid))
    right_items = list(range(mid, n))
    
    # Generate all subset sums for the left half
    # 2^23 = 8388608 subsets
    left_sums = []  # list of (sum, bitmask)
    left_mask_bits = len(left_items)
    
    # Check time
    if time.time() - start_time > DEADLINE:
        return best_indices
    
    # Generate left subset sums
    # Use iterative approach to avoid recursion depth issues
    # For 23 items, 2^23 = 8.4M subsets - this might be too slow in Python
    # Let's use a smarter approach: generate sums incrementally
    
    # Actually, 2^23 is manageable with optimization
    left_subset_count = 1 << left_mask_bits
    left_sums = [Fraction()] * left_subset_count
    left_sums[0] = Fraction()
    
    # Build subset sums using DP: sum[i] = sum[i without lowest set bit] + value[lowest set bit]
    # This is O(2^n) but with Fraction operations
    # This might be too slow. Let's try a different approach.
    
    # Alternative: use integer arithmetic with common denominator
    # LCM of 2..48 is very large (~7.4e23), so let's compute it
    from math import gcd
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = 1
    for d in denominators:
        L = lcm(L, d)
        if time.time() - start_time > DEADLINE:
            return best_indices
    
    # Convert to integer weights
    weights = [L // d for d in denominators]
    target = L - 1  # We want sum <= L-1, i.e., sum < L
    
    # Now we're looking for the maximum sum <= target achievable by a subset
    # This is a 0/1 knapsack-like problem
    
    # Meet-in-the-middle with integer weights
    # Left half: indices 0..mid-1, Right half: indices mid..n-1
    left_weights = weights[:mid]
    right_weights = weights[mid:]
    
    # Generate all subset sums for left half
    # 2^23 = 8388608 - this is feasible
    left_count = 1 << len(left_weights)
    left_sums = [0] * left_count
    for mask in range(1, left_count):
        # Find lowest set bit
        lowbit = mask & (-mask)
        bit_idx = lowbit.bit_length() - 1
        prev = mask ^ lowbit
        left_sums[mask] = left_sums[prev] + left_weights[bit_idx]
        if time.time() - start_time > DEADLINE:
            # Return current best
            return best_indices
    
    # Sort left sums
    left_sums_sorted = sorted(left_sums)
    # Remove duplicates? No, we need to track which mask gives which sum
    # Actually, for finding max sum < target, we just need the values
    # But we need to reconstruct the subset, so let's keep track
    
    # For the right half, generate all subset sums
    right_count = 1 << len(right_weights)
    right_sums = [0] * right_count
    for mask in range(1, right_count):
        lowbit = mask & (-mask)
        bit_idx = lowbit.bit_length() - 1
        prev = mask ^ lowbit
        right_sums[mask] = right_sums[prev] + right_weights[bit_idx]
        if time.time() - start_time > DEADLINE:
            return best_indices
    
    # For each right subset sum, find the best left subset sum such that
    # left_sum + right_sum < L, i.e., left_sum < L - right_sum
    # We want to maximize left_sum + right_sum subject to left_sum + right_sum < L
    
    # Sort left_sums and use binary search
    import bisect
    
    best_total_sum = 0
    best_left_mask = 0
    best_right_mask = 0
    
    for r_mask in range(right_count):
        r_sum = right_sums[r_mask]
        if r_sum >= L:
            continue
        # Find largest left_sum < L - r_sum
        target_left = L - r_sum
        # bisect_left finds first element >= target_left
        # We want the largest element < target_left
        pos = bisect.bisect_left(left_sums_sorted, target_left)
        if pos > 0:
            best_l_sum = left_sums_sorted[pos - 1]
            total_sum = best_l_sum + r_sum
            if total_sum > best_total_sum:
                best_total_sum = total_sum
                best_right_mask = r_mask
                # Need to find which left mask gives best_l_sum
                # This is a problem - we lost the mask information
                # Let's store (sum, mask) pairs
        if time.time() - start_time > DEADLINE:
            return best_indices
    
    # The above approach has a problem: we need to recover the mask from the sum
    # Let's redo with (sum, mask) tracking
    
    # Regenerate with mask tracking
    left_sum_mask = []  # list of (sum, mask)
    for mask in range(left_count):
        left_sum_mask.append((left_sums[mask], mask))
    
    # Sort by sum
    left_sum_mask.sort(key=lambda x: x[0])
    
    # Extract sorted sums and masks
    sorted_sums = [x[0] for x in left_sum_mask]
    sorted_masks = [x[1] for x in left_sum_mask]
    
    best_total_sum = 0
    best_l_mask = 0
    best_r_mask = 0
    
    for r_mask in range(right_count):
        r_sum = right_sums[r_mask]
        if r_sum >= L:
            continue
        target_left = L - r_sum
        pos = bisect.bisect_left(sorted_sums, target_left)
        if pos > 0:
            cand_l_sum = sorted_sums[pos - 1]
            total_sum = cand_l_sum + r_sum
            if total_sum > best_total_sum:
                best_total_sum = total_sum
                best_l_mask = sorted_masks[pos - 1]
                best_r_mask = r_mask
        if time.time() - start_time > DEADLINE:
            return best_indices
    
    # Reconstruct the subset
    result_indices = []
    for i in range(len(left_weights)):
        if best_l_mask & (1 << i):
            result_indices.append(i)
    for i in range(len(right_weights)):
        if best_r_mask & (1 << i):
            result_indices.append(mid + i)
    
    if not result_indices:
        return best_indices
    
    # Verify: sum must be strictly below 1
    total = Fraction()
    for idx in result_indices:
        total += values[idx]
    
    if total >= 1:
        return best_indices
    
    if total > best_total:
        best_indices = result_indices
        best_total = total
    
    return best_indices
# EVOLVE-BLOCK-END
