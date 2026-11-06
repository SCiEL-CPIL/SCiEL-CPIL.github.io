import time
from fractions import Fraction

_DEADLINE = 85.0
_T0 = time.time()

def _deadline_ok():
    return (time.time() - _T0) < _DEADLINE

def _validate(dens):
    if not (1 <= len(dens) <= 256):
        return False
    if len(set(dens)) != len(dens):
        return False
    for d in dens:
        if d < 5 or d > 10000000:
            return False
    s = Fraction(0, 1)
    for d in dens:
        s += Fraction(1, d)
    return s == 1

def _search(rem, k, min_den, out):
    if not _deadline_ok():
        return False
    
    if k == 0:
        return rem == 0

    if k == 1:
        # rem must be 1/d
        if rem.numerator == 1:
            d = rem.denominator
            if min_den <= d <= 10000000:
                out.append(d)
                return True
        return False

    # d_min: 1/d < rem => d > 1/rem
    # Also d >= min_den
    inv_rem = Fraction(1, 1) / rem
    d_min = max(min_den, inv_rem.numerator // inv_rem.denominator + 1)
    
    # d_max: rem - 1/d <= sum_{i=1}^{k-1} 1/(d+i)
    # Approx: rem <= k/d => d <= k/rem
    # We can use this as an upper bound to limit the loop
    # Let's be safe and add a small buffer or just use the approximation
    # d <= k / rem
    d_max_approx = (k * rem.denominator) // rem.numerator
    
    # We should probably cap d_max to something reasonable if the approximation is loose,
    # but for Egyptian fractions, this bound is quite tight.
    # Also d must be at least d_min.
    
    # Iterate d from d_min to d_max_approx
    # If d_max_approx is very large, we might need to be careful, but k is small (<=8)
    # and rem is a fraction, so d_max_approx is usually small.
    
    for d in range(d_min, d_max_approx + 1):
        if not _deadline_ok():
            return False
            
        # Check if 1/d < rem
        # This is guaranteed by d > 1/rem, i.e., d >= d_min
        # But let's double check to be safe against integer division issues
        if Fraction(1, d) >= rem:
            continue
            
        new_rem = rem - Fraction(1, d)
        
        # Pruning: Check if new_rem can be represented by k-1 fractions
        # Max sum of k-1 fractions with denominators > d is approx (k-1)/(d+1)
        # If new_rem > (k-1)/(d+1), it's impossible.
        # Actually, the max sum is sum_{i=1}^{k-1} 1/(d+i)
        # This sum is slightly less than (k-1)/(d+1)
        # Using the approximation (k-1)/(d+1) is a safe upper bound for pruning?
        # No, sum 1/(d+i) < sum 1/(d+1) = (k-1)/(d+1).
        # So if new_rem > (k-1)/(d+1), then new_rem > max_possible_sum, so prune.
        
        max_possible = Fraction(k - 1, d + 1)
        if new_rem > max_possible:
            continue
            
        out.append(d)
        if _search(new_rem, k - 1, d + 1, out):
            return True
        out.pop()
        
    return False

def construct():
    # The minimum number of terms is 8
    for k in range(8, 16):
        if not _deadline_ok():
            break
        out = []
        if _search(Fraction(1, 1), k, 5, out):
            if len(out) == k and _validate(out):
                return out
    
    # Fallback if search fails (should not happen for k=8)
    raise RuntimeError("No representation found within deadline")
