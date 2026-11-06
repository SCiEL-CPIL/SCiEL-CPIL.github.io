import time

_START_TIME = time.time()

def priority(vector, n):
    """
    Priority function for greedy cap set construction.
    
    Strategy:
    1. Primary signal: Prefer vectors with fewer 1s. Vectors with many 1s are 
       more likely to be the 'middle' element of a 3-term arithmetic progression 
       (x, x+1, x+2 in each coordinate where the middle is 1). By prioritizing 
       vectors with fewer 1s, we reduce the chance of blocking future additions 
       that would create APs centered on 1s.
       
    2. Secondary signal: Deterministic hash-based perturbation to break symmetry 
       and explore different orderings. This helps escape local minima caused by 
       lexicographic bias.
       
    3. Deadline guard: Return a neutral value if time is running out to avoid 
       computation overhead in critical moments.
    """
    # Deadline guard: if we've used 90% of our time budget, return a simple priority
    # to avoid any potential slowdown from complex calculations in the final moments
    elapsed = time.time() - _START_TIME
    if elapsed > 297.0:  # 3s safety margin before 300s limit
        # Simple fallback: prefer fewer 1s, no hash
        ones = sum(1 for x in vector if x == 1)
        return -float(ones)
    
    # Count number of 1s in the vector
    ones = sum(1 for x in vector if x == 1)
    
    # Count number of 0s (optional, but can help distinguish vectors with same # of 1s)
    # Actually, for cap sets, the key insight is that 1s are problematic as middle elements.
    # Vectors with all 0s or all 2s are "safe" in the sense that they can't be the middle
    # of a 3-AP in any coordinate (since 0-1=-1 and 2+1=3 are out of {0,1,2} range for 
    # the neighbors to be valid).
    
    # Primary: minimize number of 1s (negative so higher priority = fewer 1s)
    # This is the main structural improvement from inspiration 2
    
    # Secondary: deterministic hash for tie-breaking and symmetry breaking
    # Using FNV-1a style hash for good distribution
    h = 1469598103934665603
    for v in vector:
        h = (h ^ v) * 1099511628211
    # Final avalanche for better bit mixing
    h = (h ^ (h >> 32)) & 0xFFFFFFFF
    
    # Combine: primary signal dominates, hash provides fine-grained ordering
    # Scale hash to a small range so it only affects tie-breaking
    hash_score = (h / 0xFFFFFFFF) * 0.1  # Range [0, 0.1)
    
    # Return: -ones + small_hash
    # Vectors with fewer 1s get higher priority (less negative)
    # Among vectors with same # of 1s, hash determines order
    return -float(ones) + hash_score
