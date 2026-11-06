import time
import math

BASE = 51975

def _get_divisors(n):
    divs = []
    for i in range(2, int(math.isqrt(n)) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    return sorted(divs)

_DIVISORS = _get_divisors(BASE)
# Ensure all divisors > 1 are present. 
# The loop above gets factors. 
# Let's generate all divisors properly.
def _all_divisors(n):
    divs = set()
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            divs.add(i)
            divs.add(n // i)
    divs.discard(1)
    return sorted(divs)

_DIVISORS = _all_divisors(BASE)

# Precompute masks
# masks[m] will be a list of length m, where masks[m][r] is the bitmask for class (m, r)
_MASKS = {}
for m in _DIVISORS:
    masks_m = [0] * m
    for r in range(m):
        mask = 0
        # Set bits at r, r+m, r+2m, ...
        # Using a loop is O(N/m). 
        # For small m, this is large. For large m, small.
        # Total operations sum over m of m * (N/m) = N * len(divisors) ~ 51975 * 47 ~ 2.4M operations.
        # This is fine.
        for i in range(r, BASE, m):
            mask |= (1 << i)
        masks_m[r] = mask
    _MASKS[m] = masks_m

def _popcount(x):
    return x.bit_count()

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Greedy Initialization
    covered = 0
    classes = {} # m -> r
    
    # Precompute list of (m, r) to consider? 
    # Or just loop through divisors and residues.
    
    # To speed up greedy, we can just iterate through all available (m, r) 
    # and pick the one with max gain.
    # But we can only pick one r per m.
    # So for each m, find best r given current covered.
    # Then pick the m with the highest gain among all m.
    
    available_moduli = set(_DIVISORS)
    
    while time.time() < deadline:
        best_m = None
        best_r = None
        best_gain = 0
        best_new_covered = 0
        
        for m in available_moduli:
            masks_m = _MASKS[m]
            # Find best r for this m
            # We want max popcount(masks_m[r] & ~covered)
            # Which is popcount(masks_m[r]) - popcount(masks_m[r] & covered)
            # popcount(masks_m[r]) is constant for fixed m, r. It's BASE//m or (BASE-1)//m + 1.
            # Actually, number of terms is (BASE - 1 - r) // m + 1.
            # Let's just compute the intersection with covered to find the one with minimum overlap.
            
            min_overlap = BASE
            best_r_m = 0
            for r in range(m):
                overlap = _popcount(masks_m[r] & covered)
                if overlap < min_overlap:
                    min_overlap = overlap
                    best_r_m = r
            
            # Gain is total_bits - min_overlap
            # total_bits for (m, r) is the number of residues covered by that class alone.
            # This varies slightly with r.
            # Let's just compute gain directly.
            gain = _popcount(masks_m[best_r_m] & ~covered)
            
            if gain > best_gain:
                best_gain = gain
                best_m = m
                best_r = best_r_m
                best_new_covered = covered | masks_m[best_r]
        
        if best_m is None or best_gain == 0:
            break
            
        classes[best_m] = best_r
        covered = best_new_covered
        available_moduli.remove(best_m)
        
        # If we used all moduli, break
        if not available_moduli:
            break

    # Local Search
    # We have a set of classes. We want to optimize residues.
    # We also want to ensure we haven't left out a modulus that could help.
    # But greedy added all that helped. 
    # Local search can swap residues.
    
    current_classes = classes.copy()
    current_covered = covered
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Optimize residues for existing classes
        for m in list(current_classes.keys()):
            # Compute union of all other classes
            others_mask = 0
            for m2, r2 in current_classes.items():
                if m2 != m:
                    others_mask |= _MASKS[m2][r2]
            
            # Find best r for m given others_mask
            masks_m = _MASKS[m]
            best_r = current_classes[m]
            best_gain = _popcount(masks_m[best_r] & ~others_mask)
            
            for r in range(m):
                if r == best_r:
                    continue
                gain = _popcount(masks_m[r] & ~others_mask)
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            
            if best_r != current_classes[m]:
                # Check if this actually improves total coverage
                new_covered = others_mask | masks_m[best_r]
                if _popcount(new_covered) > _popcount(current_covered):
                    current_classes[m] = best_r
                    current_covered = new_covered
                    improved = True
        
        # 2. Try adding a new class
        # Only if we have available moduli
        if available_moduli:
             best_m_add = None
             best_r_add = None
             best_gain_add = 0
             
             for m in available_moduli:
                 masks_m = _MASKS[m]
                 # Find best r for this m given current_covered
                 best_r_m = 0
                 best_gain_m = 0
                 for r in range(m):
                     gain = _popcount(masks_m[r] & ~current_covered)
                     if gain > best_gain_m:
                         best_gain_m = gain
                         best_r_m = r
                 
                 if best_gain_m > best_gain_add:
                     best_gain_add = best_gain_m
                     best_m_add = m
                     best_r_add = best_r_m
             
             if best_m_add is not None and best_gain_add > 0:
                 current_classes[best_m_add] = best_r_add
                 current_covered |= _MASKS[best_m_add][best_r_add]
                 available_moduli.remove(best_m_add)
                 improved = True

    # Convert to required format
    result = []
    for m, r in current_classes.items():
        result.append([m, r])
        
    return result
