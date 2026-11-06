import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 40  # 40s safety margin
    
    # Search for k terms, starting from k=8 (minimum possible for denom >= 5)
    for k in range(8, 15):
        if time.time() > deadline:
            break
        result = _dfs(k, 5, Fraction(1, 1), [])
        if result:
            return result
            
    # Fallback: Use the seed logic or a known larger solution
    # The seed logic finds a solution. Let's implement a simpler fallback.
    # Actually, just return the seed's result if we fail.
    target, reached = 2520, {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]
    raise RuntimeError("no representation")

def _dfs(k, start_denom, remaining, current):
    if k == 0:
        if remaining == 0:
            return current
        return None
    
    # Pruning:
    # 1. If remaining <= 0, fail
    if remaining <= 0:
        return None
        
    # 2. Max possible sum with k terms starting from start_denom
    # We need to check if we can reach 'remaining' with k terms >= start_denom.
    # Max sum is achieved by smallest denominators: start_denom, start_denom+1, ...
    max_possible = sum(Fraction(1, start_denom + i) for i in range(k))
    if remaining > max_possible:
        return None
        
    # Min possible sum?
    # Min sum is achieved by largest denominators. But they are bounded by 10^7.
    # However, for small k and large remaining, min sum is not a tight constraint initially.
    # But if remaining is very small, we might need very large denominators.
    # If remaining < 1 / (start_denom + k - 1), it might be possible, but if remaining is too small 
    # to be represented by k distinct reciprocals?
    # Actually, if remaining < 1/(10^7) * k, it's impossible.
    
    # 3. Determine range for first denominator d
    # 1/d <= remaining => d >= ceil(1/remaining)
    # Also d >= start_denom
    min_d = max(start_denom, int(1 / remaining) + 1 if 1 % remaining != 0 else int(1 / remaining))
    # Wait, if remaining is Fraction, 1/remaining is Fraction.
    min_d = max(start_denom, int(1 / remaining) + (1 if (1 % remaining) != 0 else 0))
    # Actually, simpler: d >= ceil(1/remaining)
    # 1/remaining might be huge.
    # d * remaining >= 1 => d >= 1/remaining
    
    # Also, we need to leave enough sum for the remaining k-1 terms.
    # Sum of k-1 terms starting from d+1 must be >= remaining - 1/d.
    # This is implicitly handled by the next recursive call's max_possible check.
    
    # Upper bound for d:
    # We need 1/d + (k-1)/ (d+1) >= remaining ? No.
    # We need remaining - 1/d to be representable by k-1 terms > d.
    # So remaining - 1/d > 0 => 1/d < remaining => d > 1/remaining.
    # Also, the sum of the remaining k-1 terms must be at least (k-1) / (10^7)? No.
    # The sum of the remaining k-1 terms must be at least the sum of the smallest k-1 reciprocals > d? No, that's max.
    # The sum of the remaining k-1 terms must be at least ... well, they can be very small.
    # But they must be distinct.
    
    # A better upper bound:
    # If 1/d is too small, we might not be able to reach 'remaining' with the remaining k-1 terms?
    # No, if 1/d is small, remaining - 1/d is large, which is easier to reach with large terms?
    # No, we want to reach a specific sum.
    # If d is too large, 1/d is small, remaining - 1/d is close to remaining.
    # The max sum of k-1 terms starting from d+1 is sum(1/(d+i) for i in 1..k-1).
    # We need remaining - 1/d <= max_possible_next.
    
    max_d = 10**7
    # Optimization: if d is too large, 1/d is very small.
    # remaining - 1/d approx remaining.
    # max_possible_next approx (k-1)/(d+1).
    # If remaining > (k-1)/(d+1), then it's impossible?
    # No, max_possible_next is the MAX sum. If remaining - 1/d > max_possible_next, then we can't reach it.
    # So we need remaining - 1/d <= max_possible_next.
    # This gives a lower bound on d? No, upper bound?
    # As d increases, 1/d decreases, so remaining - 1/d increases.
    # max_possible_next decreases as d increases.
    # So there is a range of valid d.
    
    # Let's just iterate d from min_d upwards, but cap it.
    # If d is very large, 1/d is tiny.
    # We can stop if 1/d < remaining / k ? No.
    
    # Simple iteration:
    for d in range(min_d, min_d + 1000): # Limit search width for speed, or use tighter bound
        if d > 10**7:
            break
        if 1 / d > remaining:
            continue
        next_remaining = remaining - Fraction(1, d)
        if next_remaining < 0:
            continue
        
        # Check if next_remaining can be formed by k-1 terms > d
        # Max sum of k-1 terms starting from d+1:
        max_next = sum(Fraction(1, d + 1 + i) for i in range(k - 1))
        if next_remaining > max_next:
            # If next_remaining is too big, we need a smaller d (larger 1/d) to reduce next_remaining?
            # No, smaller d means larger 1/d, so next_remaining = remaining - 1/d is SMALLER.
            # If next_remaining > max_next, it means we subtracted too little?
            # remaining - 1/d > max_next => 1/d < remaining - max_next.
            # So d > 1 / (remaining - max_next).
            # This implies d is too SMALL. We should increase d?
            # Wait. If d is small, 1/d is large. next_remaining is small.
            # If next_remaining > max_next, it means next_remaining is LARGE.
            # So 1/d must be SMALL. So d must be LARGE.
            # So we should INCREASE d.
            # But we are iterating upwards. So we just continue.
            continue
            
        # Min sum of k-1 terms starting from d+1?
        # They can be arbitrarily large, so min sum is near 0.
        # But they must be distinct integers.
        
        res = _dfs(k - 1, d + 1, next_remaining, current + [d])
        if res:
            return res
            
    return None
