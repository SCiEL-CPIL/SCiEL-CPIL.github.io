import time
import random

_DEADLINE = 85.0

def _is_mean_of_subset(C, i):
    """
    Check if C[i] is the mean of any subset of at least 2 OTHER elements in C.
    Returns True if violation exists.
    """
    a = C[i]
    others = C[:i] + C[i+1:]
    n = len(others)
    if n < 2:
        return False
    
    # Map y -> y - a
    zs = [y - a for y in others]
    
    # Quick check: pairs
    # If y1 + y2 = 2a, then (y1-a) + (y2-a) = 0
    # So we check if there are two elements in zs that sum to 0.
    seen = set()
    for z in zs:
        if -z in seen:
            return True
        seen.add(z)
        
    # If no pair, check subsets of size >= 3.
    # We need to know if 0 is a subset sum of zs with cardinality >= 3.
    # Note: Since no pair sums to 0, any subset summing to 0 must have size >= 3.
    # Also, if there are zeros in zs, say z=0, then we need another element w such that w=0? No.
    # If z=0 is in zs, it means y=a. But y is from others, so y != a. So z cannot be 0.
    # Wait, y is from C, and a is C[i]. Since C has distinct elements, y != a, so z != 0.
    
    # So all z are non-zero.
    # We want to find a subset T of zs such that sum(T) = 0 and |T| >= 3.
    
    # Optimization: Separate positive and negative.
    pos = [z for z in zs if z > 0]
    neg = [z for z in zs if z < 0]
    
    if not pos or not neg:
        return False
        
    # DP over cardinality and sum.
    # Sum range: sum(neg) to sum(pos).
    # Let offset = -sum(neg) (so min sum becomes 0).
    # Actually, let's just use the standard bitset DP.
    
    max_neg_sum = sum(neg) # This is negative
    max_pos_sum = sum(pos)
    
    # The possible sums range from max_neg_sum to max_pos_sum.
    # We want to know if 0 is reachable.
    # Offset = -max_neg_sum.
    # Index for sum S is S - max_neg_sum.
    # Index for 0 is -max_neg_sum.
    
    # We only care about cardinalities >= 3.
    # We can run DP for k=1, 2, ..., n.
    # dp[k] is a bitset where bit j is set if sum (offset by max_neg_sum) j is reachable with k elements.
    
    # To save memory/time, we can just keep the bitset for each k.
    # n is at most 47.
    # Sum range is at most 512 * 47 ~ 24000. Bitset size is manageable.
    
    offset = -max_neg_sum
    max_bit = max_pos_sum - max_neg_sum
    
    # Initialize
    # dp[k] is an integer. Bit i is 1 if sum = i - offset is reachable.
    dp = [0] * (n + 1)
    dp[0] = 1 << offset  # Sum 0 with 0 elements? No, sum 0 is at index offset.
    # Wait, if sum is 0, index is 0 - max_neg_sum = offset.
    
    # Process each z
    for z in zs:
        if z > 0:
            shift = z
        else:
            shift = z # Negative shift is right shift?
            # Bitset shift:
            # If we have bitset B representing sums S, and we add z, new sums are S+z.
            # Index becomes (S+z) - offset = (S-offset) + z.
            # So we shift left by z if z>0, right by -z if z<0.
            
        # We must iterate k downwards to avoid using the same element twice.
        for k in range(n, 0, -1):
            if dp[k-1] == 0:
                continue
            if z > 0:
                dp[k] |= (dp[k-1] << z)
            else:
                dp[k] |= (dp[k-1] >> (-z))
                
    # Check if 0 is reachable with k >= 3
    target_bit = 1 << offset
    for k in range(3, n + 1):
        if dp[k] & target_bit:
            return True
            
    return False

def _is_valid_incremental(selected, x):
    """
    Check if x can be added to selected without violating constraints.
    1. x is not mean of subset of selected (size >= 2).
    2. No y in selected is mean of subset of selected U {x} containing x.
    """
    # Check 1
    if len(selected) >= 2:
        if _is_mean_of_subset(selected + [x], len(selected)):
            return False

    # Check 2
    # For each y in selected, check if y is mean of subset of (selected \ {y}) U {x} containing x.
    # Let S' = selected \ {y}.
    # We need subset T of S' such that sum(T) + x = (|T| + 1) * y.
    # sum(T) - |T|*y = y - x.
    # Let d_t = t - y.
    # sum(d_t) = y - x.
    # We need to check if (y-x) is a subset sum of {t-y for t in selected, t!=y}.
    # And the subset size must be such that total size >= 2.
    # Total size = |T| + 1.
    # So |T| >= 1.
    
    for y in selected:
        target = y - x
        if target == 0:
            # Then sum(T) = 0.
            # Need subset of {t-y} summing to 0 with size >= 1.
            # If there is t=y, but t!=y.
            # If there is t such that t-y=0, impossible.
            # So we need subset of {t-y} summing to 0.
            # Since y is in selected, and elements are distinct, t-y != 0.
            # So we need a subset of non-zero numbers summing to 0.
            # This is equivalent to checking if 0 is a subset sum.
            pass
        
        vals = [z - y for z in selected if z != y]
        if not vals:
            continue
            
        # Check if target is a subset sum of vals with size >= 1.
        # We can use a similar DP.
        # But this is called N times.
        # Optimization: Pre-compute something?
        
        # Simple DP for this specific check.
        n = len(vals)
        max_sum = sum(abs(v) for v in vals)
        if abs(target) > max_sum:
            continue
            
        offset = max_sum
        dp = [0] * (n + 1)
        dp[0] = 1 << offset
        
        for v in vals:
            for k in range(n, 0, -1):
                if dp[k-1] == 0:
                    continue
                if v >= 0:
                    dp[k] |= (dp[k-1] << v)
                else:
                    dp[k] |= (dp[k-1] >> (-v))
                    
        bit_idx = offset + target
        for k in range(1, n + 1):
            if dp[k] & (1 << bit_idx):
                return False
                
    return True

def construct():
    start_time = time.time()
    best_set = []
    
    candidates = list(range(1, 513))
    
    # Generate diverse orderings
    orders = []
    
    # 1. Spread: 512, 1, 511, 2, ...
    spread = []
    lo, hi = 1, 512
    while lo <= hi:
        spread.append(hi)
        if lo != hi:
            spread.append(lo)
        lo += 1
        hi -= 1
    orders.append(spread)
    orders.append(spread[::-1])
    
    # 2. Random shuffles
    rng = random.Random(42)
    for _ in range(100):
        r = candidates[:]
        rng.shuffle(r)
        orders.append(r)
        
    # 3. "Hard" numbers first?
    # Numbers with few divisors are less likely to be means?
    # Actually, being a mean depends on the set.
    
    for order in orders:
        if time.time() - start_time > _DEADLINE - 5:
            break
            
        selected = []
        sel_set = set()
        
        for x in order:
            if time.time() - start_time > _DEADLINE - 5:
                break
            if len(selected) >= 48:
                break
            if x in sel_set:
                continue
                
            # Fast pair check (O(N^2) but small N)
            valid = True
            # Check if x is mean of pair in selected
            # x = (a+b)/2 => a+b=2x
            # Use set for O(1) lookup of b = 2x-a
            for a in selected:
                b = 2 * x - a
                if b in sel_set and b != a:
                    valid = False
                    break
            
            if not valid:
                continue
                
            # Check if any a in selected is mean of pair (x, b) where b in selected
            # a = (x+b)/2 => b = 2a-x
            for a in selected:
                b = 2 * a - x
                if b in sel_set and b != a and b != x:
                    valid = False
                    break
            
            if not valid:
                continue
                
            # Full check
            if _is_valid_incremental(selected, x):
                selected.append(x)
                sel_set.add(x)
                
        if len(selected) > len(best_set):
            best_set = selected[:]
            # If we hit 48, we can stop? No, maybe there's a better set?
            # The score is cardinality. 48 is max.
            if len(best_set) == 48:
                break

    if best_set:
        # Final verification
        C = best_set[:]
        for i in range(len(C)):
            if _is_mean_of_subset(C, i):
                return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        return best_set
    else:
        return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
