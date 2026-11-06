"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s deadline to stay under 60s limit
    
    # Use the base-3 (ternary) construction: numbers with only digits 0 and 1 in base 3
    # This set is known to be 3-term AP free, hence also 4-term AP free.
    # For n=512, this gives a good baseline.
    
    def ternary_01_set(n):
        """Generate all numbers in [1, n] whose base-3 representation contains only 0s and 1s."""
        result = set()
        # Max number of digits needed: 3^6 = 729 > 512, so 6 digits suffice
        for mask in range(1, 1 << 6):  # skip 0 (mask=0)
            val = 0
            for bit in range(6):
                if mask & (1 << bit):
                    val += 3 ** bit
            if 1 <= val <= n:
                result.add(val)
        return result
    
    n = 512
    selected = ternary_01_set(n)
    
    # Verify no 4-term AP exists (should be guaranteed, but let's be safe)
    def has_4ap(S):
        S_set = set(S)
        for a in S:
            for d in range(1, (n - a) // 3 + 1):
                if (a + d) in S_set and (a + 2*d) in S_set and (a + 3*d) in S_set:
                    return True
        return False
    
    if has_4ap(selected):
        # Fallback to greedy if ternary construction somehow has an issue
        selected = set()
        for x in range(1, 513):
            if all(not all(x - j*d in selected for j in (1,2,3)) for d in range(1, (x-1)//3+1)):
                selected.add(x)
    
    # Try to improve by adding elements greedily while maintaining 4-AP-free property
    # Check if we can add any elements not in the ternary set
    for x in range(1, 513):
        if time.time() > deadline:
            break
        if x in selected:
            continue
        
        # Check if adding x would create a 4-term AP
        can_add = True
        for a in list(selected):
            # x could be the 2nd, 3rd, or 4th term of a 4-AP with existing elements
            # Case 1: x is 4th term: a, a+d, a+2d, x => d = (x-a)/3, need d integer
            if (x - a) % 3 == 0:
                d = (x - a) // 3
                if d > 0 and (a + d) in selected and (a + 2*d) in selected:
                    can_add = False
                    break
            # Case 2: x is 3rd term: a, a+d, x, x+d => d = (x-a)/2, need d integer
            if (x - a) % 2 == 0:
                d = (x - a) // 2
                if d > 0 and (a + d) in selected and (x + d) <= n and (x + d) in selected:
                    can_add = False
                    break
            # Case 3: x is 2nd term: a, x, x+d, x+2d => d = x-a, need x+d and x+2d in set
            d = x - a
            if d > 0 and (x + d) <= n and (x + 2*d) <= n:
                if (x + d) in selected and (x + 2*d) in selected:
                    can_add = False
                    break
        
        if can_add:
            selected.add(x)
    
    return sorted(selected)
# EVOLVE-BLOCK-END
