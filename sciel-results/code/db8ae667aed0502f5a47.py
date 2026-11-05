"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    p = 1009
    n = 1008
    deadline = time.time() + 85  # Safety margin before 90s limit
    
    # Precompute modular inverses
    inv = [0] * (p)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    # Self-inverse elements: x^2 % p == 1
    # For prime p, solutions are x=1 and x=p-1
    self_inverse = {1, p - 1}
    
    best_len = 0
    best_start = 1
    
    # Two-pointer sliding window approach
    # For each end position, find the smallest valid start
    # An interval [start, end] is valid iff:
    #   1. No element in [start, end] is self-inverse
    #   2. For every x in [start, end], inv[x] is NOT in [start, x-1]
    #      (i.e., inv[x] < start or inv[x] > end)
    
    # Key insight: for a fixed end, the constraint is that for all x in [start, end],
    # inv[x] must not lie in [start, x-1]. This means inv[x] < start OR inv[x] > end.
    # Since inv[x] > x is possible, we need inv[x] > end as well.
    
    # Let's track: for each position x, the "forbidden start range" if x is in the window.
    # If x is in [start, end], then we need inv[x] < start OR inv[x] > end.
    # The constraint "inv[x] < start" means start > inv[x].
    # The constraint "inv[x] > end" is automatically satisfied if end < inv[x].
    
    # Actually, let me think differently. For a window [start, end]:
    # For each x in [start, end]:
    #   - x must not be self-inverse
    #   - inv[x] must not be in [start, x-1]
    #     which means: inv[x] < start OR inv[x] > x-1
    #     But since we're checking inv[x] in [start, end], and x <= end,
    #     inv[x] > x-1 is equivalent to inv[x] >= x
    #     So the constraint is: inv[x] < start OR inv[x] >= x
    #     Wait, that's not quite right either.
    
    # Let me reconsider. inv[x] is in [1, p-1]. For x in [start, end]:
    # We need inv[x] NOT in [start, x-1].
    # This fails iff start <= inv[x] <= x-1.
    # So the constraint for x is: NOT (start <= inv[x] <= x-1)
    # Which is: start > inv[x] OR inv[x] > x-1
    # i.e., start > inv[x] OR inv[x] >= x
    
    # For a fixed end, as start increases, the constraint start > inv[x] becomes easier to satisfy.
    # So for each end, the minimum valid start is determined by the maximum of inv[x] over all x in [start, end] where inv[x] < x.
    
    # Let me use a different approach: for each end, binary search or linear scan for the minimum valid start.
    
    # Actually, let's use the two-pointer method directly:
    # Maintain a window [left, right]. For each right, advance left until the window is valid.
    
    # To check validity efficiently, we need to know if there exists x in [left, right] such that
    # left <= inv[x] <= x-1.
    
    # Alternative: for each position x, define conflict_start[x] = inv[x] if inv[x] < x else 0.
    # Then x conflicts with the window [left, right] iff left <= conflict_start[x] (and x is in window).
    # Wait, that's not quite right. The conflict is: left <= inv[x] <= x-1.
    # So if inv[x] < x, the conflict occurs when left <= inv[x].
    # If inv[x] >= x, there's no conflict from x (since inv[x] can't be in [left, x-1] if inv[x] >= x).
    
    # So for each x, define:
    #   if inv[x] < x: min_start_for_x = inv[x] + 1 (start must be > inv[x])
    #   else: min_start_for_x = 1 (no constraint from this x)
    
    # For a window [left, right] to be valid, we need left >= min_start_for_x for all x in [left, right].
    # Also, no x in [left, right] can be self-inverse.
    
    # So the minimum valid left for a given right is:
    #   max over x in [left, right] of min_start_for_x, and also left must not include self-inverses.
    
    # This is getting complex. Let me just use a straightforward two-pointer with a data structure.
    
    # For each right, we need the maximum of min_start_for_x over x in [left, right].
    # As right increases, we add min_start_for_right.
    # As left increases, we remove min_start_for_left.
    
    # We can use a segment tree or a heap to track the max min_start_for_x in the window.
    
    # Given n=1008, even O(n^2) is fast enough. Let me just do the O(n^2) approach but with early termination.
    
    # Precompute min_start_for_x
    min_start = [1] * (n + 1)  # 1-indexed
    for x in range(1, n + 1):
        if x in self_inverse:
            min_start[x] = x + 1  # Force start to be after x (effectively excludes x)
        else:
            ix = inv[x]
            if ix < x:
                min_start[x] = ix + 1  # start must be > ix
            else:
                min_start[x] = 1
    
    # For each start, find the maximum end such that the interval is valid.
    # An interval [start, end] is valid iff for all x in [start, end]: min_start[x] <= start
    # (and x is not self-inverse, which is handled by min_start[x] = x+1 > start when x >= start)
    
    # Wait, if x is self-inverse and x >= start, then min_start[x] = x+1 > start, so the condition fails. Good.
    
    # So for a fixed start, the interval [start, end] is valid iff max(min_start[x] for x in [start, end]) <= start.
    # We want the largest end such that this holds.
    
    # As end increases, max(min_start[x]) is non-decreasing. So we can binary search or linearly extend.
    
    best_len = 0
    best_start = 1
    
    for start in range(1, n + 1):
        if time.time() > deadline:
            break
        
        # Find the maximum end such that max(min_start[x] for x in [start, end]) <= start
        # Start with end = start
        max_min = min_start[start]
        if max_min > start:
            continue  # Even [start, start] is invalid
        
        end = start
        while end < n:
            next_end = end + 1
            if min_start[next_end] > start:
                break
            max_min = max(max_min, min_start[next_end])
            if max_min > start:
                break
            end = next_end
        
        length = end - start + 1
        if length > best_len:
            best_len = length
            best_start = start
    
    if best_len == 0:
        # Should not happen, but return a valid fallback
        best_start = 2
        best_len = 1
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
