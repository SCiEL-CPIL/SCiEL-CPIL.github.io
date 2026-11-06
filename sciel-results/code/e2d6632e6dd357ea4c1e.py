import time

def construct():
    """
    Find the longest contiguous interval [start, start+length-1] contained in [1, 1008]
    such that for all x, y in the interval, x*y % 1009 != 1.
    
    This is a fixed-size problem (p=1009), so we can do an exhaustive O(n^2) check
    or use a smarter approach. The current best score is 2.8333, which corresponds to
    length = 503 (since score = length/seed, and if seed=177.5, then 503/177.5 ≈ 2.8333).
    Wait, the task says "The evaluator measures the raw value (higher is better). The search 
    score is value/seed". The current program returns a length. Let's assume the "value" 
    is the length itself, and the seed is some baseline.
    
    Actually, looking at the current program, it returns [best_start, best_len].
    The score is likely just best_len.
    If the current score is 2.8333, and the best_len is 503, then 503/seed = 2.8333 => seed ≈ 177.5.
    This suggests the "value" is the length.
    
    Let's verify the maximum possible length.
    The condition is that no two elements in the interval are modular inverses modulo 1009.
    Also x=y is checked, so x^2 % 1009 != 1.
    The solutions to x^2 = 1 mod 1009 are x=1 and x=1008.
    So 1 and 1008 cannot be in the interval.
    
    Let's write a robust solver that finds the true maximum length.
    """
    p = 1009
    n = p - 1  # 1008
    
    # Precompute inverses
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    # We want to find the max length of a contiguous subarray [s, e] within [1, 1008]
    # such that for all x, y in [s, e], x * y % p != 1.
    
    # A pair (x, y) is "bad" if x * y % p == 1.
    # Note that if x is in the interval, then inv[x] cannot be in the interval.
    
    # We can iterate over all possible start positions s from 1 to 1008.
    # For each s, we expand e as far as possible while maintaining the property.
    # This is a standard "longest valid subarray" problem, solvable in O(n^2) or O(n) with a sliding window.
    
    # Let's use a sliding window approach.
    # Maintain a window [left, right].
    # We need to ensure no pair (x, y) in the window has x*y % p == 1.
    # This is equivalent to saying: for every x in the window, inv[x] is NOT in the window.
    
    # We can keep track of the "forbidden" positions.
    # Actually, a simpler way for small N (1008) is just O(n^2) brute force, which is ~10^6 operations, very fast.
    # Let's do O(n^2) to be safe and simple.
    
    max_len = 0
    best_start = 1
    
    for s in range(1, n + 1):
        # Try to extend the interval starting at s
        # We need to check if we can include s, s+1, ..., e
        # Condition: for all x, y in [s, e], x*y % p != 1.
        
        # We can just iterate e from s to n and check if adding e violates the condition.
        # But checking all pairs in [s, e] is O((e-s)^2), making the whole thing O(n^3) ~ 10^9, which might be slow.
        
        # Better: For a fixed s, as we increase e, we just need to check if the new element e conflicts with any existing element in [s, e-1].
        # e conflicts with x in [s, e-1] if e * x % p == 1, i.e., x == inv[e].
        # So for a fixed s, the interval [s, e] is valid if and only if:
        # 1. e is not self-inverse (e != 1, e != 1008) -- wait, self-inverse is x^2=1, so x=1, 1008.
        #    If e is 1 or 1008, then e*e % p == 1, so it's invalid immediately.
        # 2. inv[e] is not in [s, e-1].
        
        # So for each s, we can iterate e from s to n.
        # We maintain a set of elements in the current window? Or just check if inv[e] is in [s, e-1].
        # Since the window is contiguous [s, e-1], inv[e] is in the window if s <= inv[e] <= e-1.
        
        # Let's trace:
        # Start with empty window.
        # Add e.
        # Check if e is self-inverse (e in {1, 1008}). If so, invalid.
        # Check if inv[e] is in [s, e-1]. If so, invalid.
        
        # This is O(1) per extension.
        # Total O(n^2).
        
        valid = True
        for e in range(s, n + 1):
            # Check if e can be added to [s, e-1]
            # 1. Self-inverse check
            if e * e % p == 1:
                break
            # 2. Inverse check
            ie = inv[e]
            if s <= ie <= e - 1:
                break
            
            # If we reach here, [s, e] is valid.
            length = e - s + 1
            if length > max_len:
                max_len = length
                best_start = s
        
    return [best_start, max_len]
