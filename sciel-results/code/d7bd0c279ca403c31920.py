"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
# EVOLVE-BLOCK-START

def _is_valid_interval(start, length, p=1009):
    """Check that for every x,y in [start, start+length-1], x*y % p != 1."""
    end = start + length - 1
    if start < 1 or end > 1008 or length <= 0:
        return False
    present = set(range(start, end + 1))
    for x in present:
        # self-inverse check: x*x % p == 1
        if (x * x) % p == 1:
            return False
        # inverse pair check: if x^-1 is in present, invalid
        inv = pow(x, p - 2, p)
        if inv in present:
            return False
    return True

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    best_start = 2
    best_length = 1
    
    # Precompute modular inverses for all values 1..1008
    p = 1009
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    # Self-inverse values (x*x % p == 1): x=1 and x=1008
    self_inv = {1, p - 1}
    
    # Try all possible starts and find the maximum valid length
    for start in range(1, 1009):
        if time.time() > deadline:
            break
        # Find the maximum length for this start
        # We can extend the interval as long as no new element creates a forbidden pair
        # with any existing element, or is self-inverse
        length = 0
        for end in range(start, 1009):
            x = end
            # Check if x is self-inverse
            if x in self_inv:
                break
            # Check if x's inverse is in [start, end]
            if inv[x] >= start and inv[x] <= end:
                break
            # Also need to check: for all existing y in [start, end-1], 
            # inv[y] != x. But since we check inv[x] in [start,end], and inv is bijective,
            # this is symmetric. Actually we need: for any y in [start, end-1], y*x % p != 1
            # which means x != inv[y]. Since inv is a bijection, x = inv[y] iff y = inv[x].
            # So checking inv[x] in [start, end] is sufficient.
            length = end - start + 1
        
        if length > best_length:
            best_length = length
            best_start = start
    
    return [best_start, best_length]

# EVOLVE-BLOCK-END
