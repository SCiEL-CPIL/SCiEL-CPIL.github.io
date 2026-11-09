import time

def construct():
    start_time = time.time()
    limit = 512
    max_size = 48
    S = []
    
    # We iterate in decreasing order
    # This is a heuristic: large numbers are less likely to be the mean of a subset of larger numbers
    # (which don't exist yet if we go decreasing, wait, we are adding to S, so S contains larger numbers).
    # If we add x (small) to S (large), we check if x is mean of S.
    # Mean of large numbers is large. x is small. So x is unlikely to be the mean.
    # We also check if a in S becomes mean of S U {x}.
    # This seems safer.
    
    for x in range(limit, 0, -1):
        if len(S) >= max_size:
            break
        if time.time() - start_time > 55:
            break
            
        # Check 1: Is x the mean of a subset of S?
        valid = True
        if len(S) >= 2:
            bits_S = 1
            for v in S:
                bits_S |= bits_S << v
            
            max_sum_S = sum(S)
            for k in range(2, len(S) + 1):
                target = k * x
                if target > max_sum_S:
                    break
                if (bits_S >> target) & 1:
                    valid = False
                    break
        
        if not valid:
            continue
            
        # Check 2: Does adding x make any a in S the mean of a subset of S U {x}?
        # The subset must contain x.
        # sum(T) + x = k * a  => sum(T) = k * a - x
        # T is a subset of S \ {a}.
        # |T| = k - 1. Since subset size >= 2, k >= 2.
        # Max k is len(S) (since T can be S\{a}, size len(S)-1, so k-1 = len(S)-1 => k=len(S)).
        # Wait, T can be any subset of S\{a}.
        # The size of the full subset is |T| + 1.
        # The condition is "subset of at least two OTHER elements".
        # So the full subset size M >= 2.
        # If the full subset is T U {x}, then M = |T| + 1.
        # So |T| >= 1.
        # k = M. So k >= 2.
        
        for a in S:
            # Compute subset sums of S \ {a}
            bits_S_minus_a = 1
            for v in S:
                if v != a:
                    bits_S_minus_a |= bits_S_minus_a << v
            
            max_sum_minus_a = sum(S) - a
            
            for k in range(2, len(S) + 1):
                target = k * a - x
                if target < 0:
                    continue
                if target > max_sum_minus_a:
                    continue
                if (bits_S_minus_a >> target) & 1:
                    valid = False
                    break
            
            if not valid:
                break
        
        if valid:
            S.append(x)
            
    return S
